#!/usr/bin/env python3
"""
Update whale_sightings.json from curated sources.

Deterministic:
- No "search the whole web"
- Pulls from curated URLs in config/sources.yml
- Parses pages with:
  - orcanetwork_recent_sightings (custom)
  - generic_dated_page (generic HTML parsing)
  - rss_feed (RSS/Atom parsing via feedparser)

Outputs JSON array entries that match your Swift model:
id, name, species, info, date, latitude, longitude, area, source, behaviors
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz

# Requires: pip install feedparser
import feedparser


ALLOWED_SPECIES = {"Orca", "Humpback", "Sperm whale", "Great White Shark", "Blue Whale"}

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
)

# -------------------------
# Helpers
# -------------------------

def now_local(tz_name: str) -> datetime:
    return datetime.now(tz.gettz(tz_name))

def to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

def species_key(species: str) -> str:
    return {
        "Orca": "orca",
        "Humpback": "humpback",
        "Sperm whale": "sperm",
        "Great White Shark": "white",
        "Blue Whale": "blue",
    }[species]

def clamp_nudge(lat: float, lon: float, dlat: float, dlon: float) -> Tuple[float, float]:
    # limit each component to [-0.05, 0.05]
    dlat = max(-0.05, min(0.05, dlat))
    dlon = max(-0.05, min(0.05, dlon))
    return lat + dlat, lon + dlon

def infer_behaviors(text: str) -> List[str]:
    t = (text or "").lower()
    out = ["reported"]
    if any(k in t for k in ["hunt", "predation", "prey", "kill"]):
        out.append("hunting")
    if any(k in t for k in ["feed", "feeding", "lunge", "krill"]):
        out.append("feeding")
    if "breach" in t:
        out.append("breaching")
    if any(k in t for k in ["call", "hydrophone", "vocal"]):
        out.append("vocalizing")
    if any(k in t for k in ["travel", "headed", "northbound", "southbound", "moving"]):
        out.append("traveling")

    # de-dupe, preserve order
    seen = set()
    deduped = []
    for b in out:
        if b not in seen:
            deduped.append(b)
            seen.add(b)
    return deduped

def normalize_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fetch_text(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=45)
    r.raise_for_status()
    return normalize_text(r.text)

def within_window(dt: datetime, now_dt: datetime, max_days: int) -> bool:
    return (now_dt.date() - dt.date()).days <= max_days

def parse_date_from_url(url: str, tz_name: str) -> Optional[datetime]:
    # patterns like /2026/1/14/ or /2026-01-14/
    m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})(?:/|$)", url)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, tzinfo=tz.gettz(tz_name))
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", url)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, tzinfo=tz.gettz(tz_name))
    return None

def best_effort_parse_date(s: str, tz_name: str, today: datetime) -> Optional[datetime]:
    s = (s or "").strip()

    # ISO
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, tzinfo=tz.gettz(tz_name))

    # mm/dd/yyyy or m/d/yy
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        return datetime(y, mo, d, tzinfo=tz.gettz(tz_name))

    # dd.mm.yyyy
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{2,4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        return datetime(y, mo, d, tzinfo=tz.gettz(tz_name))

    # Month name day (optional year)
    month_re = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    m = re.match(rf"^({month_re})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?$", s)
    if m:
        month_name = m.group(1)
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        dt = datetime.strptime(f"{year} {month_name} {day}", "%Y %B %d")
        dt = dt.replace(tzinfo=tz.gettz(tz_name))
        # handle year boundary if inferred date is "future"
        if dt.date() > today.date() and not m.group(3):
            dt = dt.replace(year=year - 1)
        return dt

    return None

def detect_species(text: str, force: Optional[List[str]] = None) -> List[str]:
    if force:
        out = [s for s in force if s in ALLOWED_SPECIES]
        return out

    t = (text or "").lower()
    hits: List[str] = []

    if any(k in t for k in ["orca", "killer whale", "killer whales", "bigg", "southern resident"]):
        hits.append("Orca")
    if "humpback" in t:
        hits.append("Humpback")
    if "sperm whale" in t or "spermwhale" in t:
        hits.append("Sperm whale")
    if any(k in t for k in ["great white", "white shark"]):
        hits.append("Great White Shark")
    if "blue whale" in t or "bluewhale" in t:
        hits.append("Blue Whale")

    # de-dupe preserve order
    seen = set()
    out: List[str] = []
    for s in hits:
        if s in ALLOWED_SPECIES and s not in seen:
            out.append(s)
            seen.add(s)
    return out

def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def try_extract_entry_latlon(entry: Any) -> Optional[Tuple[float, float]]:
    """
    Attempts common RSS/GeoRSS fields feedparser exposes.
    Returns (lat, lon) or None.
    """
    # Common fields:
    # entry.get("geo_lat"), entry.get("geo_long")
    lat = safe_float(entry.get("geo_lat"))
    lon = safe_float(entry.get("geo_long"))
    if lat is not None and lon is not None:
        return lat, lon

    # GeoRSS point sometimes appears as "georss_point": "lat lon"
    gp = entry.get("georss_point") or entry.get("georss:point")
    if isinstance(gp, str):
        m = re.search(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", gp.strip())
        if m:
            lat = safe_float(m.group(1))
            lon = safe_float(m.group(2))
            if lat is not None and lon is not None:
                return lat, lon

    # Some feeds include "where" or other nested structures; skip for now (deterministic)
    return None


@dataclass
class Candidate:
    date: datetime
    species: str
    name: str
    info: str
    area: str
    source: str
    latitude: float
    longitude: float
    behaviors: List[str]
    source_key: str


# -------------------------
# Generic “dated sections” extraction (HTML)
# -------------------------

def extract_dated_sections(text: str, tz_name: str, today: datetime) -> List[Tuple[datetime, str]]:
    """
    Finds date markers in text and splits into sections.
    Returns list of (datetime, section_text).
    """
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",                 # 2026-01-14
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",           # 1/14/2026
        r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",         # 14.01.2026
        rf"\b(?:{'|'.join(MONTHS)})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b",  # January 14, 2026 or January 14
    ]

    matches: List[Tuple[int, int, str]] = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            matches.append((m.start(), m.end(), m.group(0)))

    if not matches:
        return []

    matches.sort(key=lambda x: x[0])

    cleaned: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, val in matches:
        if s < last_end:
            continue
        cleaned.append((s, e, val))
        last_end = e

    out: List[Tuple[datetime, str]] = []
    for i, (s, e, dstr) in enumerate(cleaned):
        dt = best_effort_parse_date(dstr, tz_name, today)
        if not dt:
            continue
        section_start = e
        section_end = cleaned[i + 1][0] if i + 1 < len(cleaned) else len(text)
        section = text[section_start:section_end].strip()
        if len(section) < 20:
            continue
        out.append((dt, section))

    return out

def build_candidates_from_generic_html(
    cfg: Dict[str, Any],
    session: requests.Session,
    tz_name: str,
    source_key: str,
) -> List[Candidate]:
    url = cfg["url"]
    area = (cfg.get("area", "") or "").strip()
    lat0 = float(cfg.get("latitude", 0.0))
    lon0 = float(cfg.get("longitude", 0.0))
    max_items = int(cfg.get("max_items", 10))

    text = fetch_text(session, url)
    today = now_local(tz_name)

    # optional force_species
    force_species: Optional[List[str]] = None
    fs = cfg.get("force_species")
    if fs:
        if isinstance(fs, list):
            force_species = [str(x) for x in fs]
        else:
            force_species = [str(fs)]

    sections = extract_dated_sections(text, tz_name, today)

    candidates: List[Candidate] = []

    if sections:
        for dt, section in sections:
            species_list = detect_species(section, force=force_species)
            for species in species_list[:2]:
                nud = cfg.get("uncertain_offshore_nudge") or {}
                lat, lon = clamp_nudge(lat0, lon0, float(nud.get("dlat", 0.0)), float(nud.get("dlon", 0.0)))

                name = f"{species} sighting ({area})" if area else f"{species} sighting"
                info = f"Parsed from {source_key} on {to_date_str(dt)}."
                candidates.append(
                    Candidate(
                        date=dt,
                        species=species,
                        name=name,
                        info=info,
                        area=area,
                        source=url,
                        latitude=lat,
                        longitude=lon,
                        behaviors=infer_behaviors(section),
                        source_key=source_key,
                    )
                )
    else:
        # single page fallback
        dt = parse_date_from_url(url, tz_name) or today
        species_list = detect_species(text, force=force_species)
        for species in species_list[:2]:
            nud = cfg.get("uncertain_offshore_nudge") or {}
            lat, lon = clamp_nudge(lat0, lon0, float(nud.get("dlat", 0.0)), float(nud.get("dlon", 0.0)))

            name = f"{species} sighting ({area})" if area else f"{species} sighting"
            info = f"Parsed from {source_key} (single-page) on {to_date_str(dt)}."
            candidates.append(
                Candidate(
                    date=dt,
                    species=species,
                    name=name,
                    info=info,
                    area=area,
                    source=url,
                    latitude=lat,
                    longitude=lon,
                    behaviors=infer_behaviors(text),
                    source_key=source_key,
                )
            )

    candidates.sort(key=lambda c: c.date, reverse=True)
    return candidates[:max_items]


# -------------------------
# RSS/Atom parser
# -------------------------

def parse_rss_feed(cfg: Dict[str, Any], session: requests.Session, tz_name: str, source_key: str) -> List[Candidate]:
    """
    Parses an RSS/Atom feed and produces one candidate per feed entry (up to max_items).
    Species is detected from title/summary unless force_species is provided.
    Date is taken from published/updated if available.
    Lat/Lon:
      - uses GeoRSS fields when present
      - otherwise falls back to cfg latitude/longitude (offshore)
    """
    url = cfg["url"]
    area = (cfg.get("area", "") or "").strip()
    lat0 = float(cfg.get("latitude", 0.0))
    lon0 = float(cfg.get("longitude", 0.0))
    max_items = int(cfg.get("max_items", 10))

    # optional force_species
    force_species: Optional[List[str]] = None
    fs = cfg.get("force_species")
    if fs:
        if isinstance(fs, list):
            force_species = [str(x) for x in fs]
        else:
            force_species = [str(fs)]

    # Fetch feed ourselves (more reliable with headers/timeouts)
    r = session.get(url, timeout=45)
    r.raise_for_status()
    feed = feedparser.parse(r.content)

    today = now_local(tz_name)
    candidates: List[Candidate] = []

    for entry in feed.entries[: max_items * 2]:  # small buffer; we trim later
        title = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or entry.get("description") or "").strip()
        link = (entry.get("link") or url).strip()

        blob = (title + "\n" + summary).strip()

        species_list = detect_species(blob, force=force_species)
        if not species_list:
            continue

        # datetime: published_parsed / updated_parsed -> struct_time
        dt: Optional[datetime] = None
        st = entry.get("published_parsed") or entry.get("updated_parsed")
        if st:
            # treat as UTC then attach tz; this is good enough for date filtering
            dt = datetime.fromtimestamp(time.mktime(st), tz=tz.UTC).astimezone(tz.gettz(tz_name))
        else:
            # try from link/url, else today
            dt = parse_date_from_url(link, tz_name) or parse_date_from_url(url, tz_name) or today

        # coordinates
        latlon = try_extract_entry_latlon(entry)
        if latlon:
            lat, lon = latlon
        else:
            nud = cfg.get("uncertain_offshore_nudge") or {}
            lat, lon = clamp_nudge(lat0, lon0, float(nud.get("dlat", 0.0)), float(nud.get("dlon", 0.0)))

        for species in species_list[:1]:  # RSS entries are usually one topic; keep 1 per entry
            name = f"{species} sighting ({area})" if area else f"{species} sighting"
            info = f"From RSS feed {source_key} on {to_date_str(dt)}."
            candidates.append(
                Candidate(
                    date=dt,
                    species=species,
                    name=name,
                    info=info,
                    area=area,
                    source=link or url,
                    latitude=float(lat),
                    longitude=float(lon),
                    behaviors=infer_behaviors(blob),
                    source_key=source_key,
                )
            )

    candidates.sort(key=lambda c: c.date, reverse=True)
    return candidates[:max_items]


# -------------------------
# Specific parser: Orca Network
# -------------------------

def parse_orcanetwork_recent_sightings(cfg: Dict[str, Any], session: requests.Session, tz_name: str, source_key: str) -> List[Candidate]:
    url = cfg["url"]
    area_default = (cfg.get("area", "") or "").strip()
    lat_fallback = float(cfg.get("latitude", 0.0))
    lon_fallback = float(cfg.get("longitude", 0.0))
    max_items = int(cfg.get("max_items", 18))

    html = session.get(url, timeout=45).text
    text = normalize_text(html)

    month_re = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    heading_pat = re.compile(rf"^(?P<month>{month_re})\s+(?P<day>\d{{1,2}})\s*$", re.MULTILINE)

    matches = list(heading_pat.finditer(text))
    if not matches:
        return []

    today = now_local(tz_name)
    year = today.year

    # very rough location -> safe offshore coords
    loc_coords = {
        "admiralty inlet": (48.15, -122.80),
        "haro strait": (48.50, -123.20),
        "puget sound": (47.90, -122.55),
        "san juans": (48.55, -123.05),
        "possession sound": (47.95, -122.35),
        "hood canal": (47.70, -122.95),
        "saratoga passage": (48.10, -122.45),
    }

    def best_coords(block: str) -> Tuple[float, float, bool]:
        b = block.lower()
        for k, (la, lo) in loc_coords.items():
            if k in b:
                return la, lo, False
        return lat_fallback, lon_fallback, True

    out: List[Candidate] = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        month_name = m.group("month")
        day = int(m.group("day"))

        dt = datetime.strptime(f"{year} {month_name} {day}", "%Y %B %d").replace(tzinfo=tz.gettz(tz_name))
        if dt.date() > today.date():
            dt = dt.replace(year=year - 1)

        species_hits: List[str] = []
        if re.search(r"KILLER\s+WHALES|SOUTHERN\s+RESIDENTS|ORCA\b", block, re.IGNORECASE):
            species_hits.append("Orca")
        if re.search(r"HUMPBACK", block, re.IGNORECASE):
            species_hits.append("Humpback")
        if re.search(r"SPERM\s+WHALE", block, re.IGNORECASE):
            species_hits.append("Sperm whale")
        if re.search(r"GREAT\s+WHITE|WHITE\s+SHARK", block, re.IGNORECASE):
            species_hits.append("Great White Shark")
        if re.search(r"BLUE\s+WHALE", block, re.IGNORECASE):
            species_hits.append("Blue Whale")

        for species in species_hits[:2]:
            if species not in ALLOWED_SPECIES:
                continue

            lat, lon, uncertain = best_coords(block)
            if uncertain and "uncertain_offshore_nudge" in cfg:
                nud = cfg["uncertain_offshore_nudge"]
                lat, lon = clamp_nudge(lat, lon, float(nud.get("dlat", 0.0)), float(nud.get("dlon", 0.0)))

            area = area_default or "Salish Sea (offshore)"
            name = f"{species} sighting ({area})"
            info = f"Orca Network recent sightings indicates {species} on {to_date_str(dt)}."

            out.append(
                Candidate(
                    date=dt,
                    species=species,
                    name=name,
                    info=info,
                    area=area,
                    source=url,
                    latitude=float(lat),
                    longitude=float(lon),
                    behaviors=infer_behaviors(block),
                    source_key=source_key,
                )
            )

    out.sort(key=lambda c: c.date, reverse=True)
    return out[:max_items]


# -------------------------
# Parsers mapping
# -------------------------

def parse_generic_html(cfg: Dict[str, Any], session: requests.Session, tz_name: str, source_key: str) -> List[Candidate]:
    return build_candidates_from_generic_html(cfg, session, tz_name, source_key)

PARSERS = {
    # Custom
    "orcanetwork_recent_sightings": parse_orcanetwork_recent_sightings,

    # Generic HTML aliases
    "generic_dated_page": parse_generic_html,
    "generic_article": parse_generic_html,
    "article_basic": parse_generic_html,

    # Backward-compat names you used earlier in YAML
    "eaglewing_daily_sighting_report": parse_generic_html,
    "victoriawhalewatching_captains_log": parse_generic_html,
    "montereybay_sightings": parse_generic_html,
    "dolphinsafari_sightings_log": parse_generic_html,
    "hawaiianadventures_weekly_whale_report": parse_generic_html,
    "whalewatchingakureyri_blog": parse_generic_html,
    "elding_whale_diary": parse_generic_html,
    "seatrips_whale_diary": parse_generic_html,
    "whalewatchwesternaustralia_daily": parse_generic_html,
    "pirsa_shark_sightings": parse_generic_html,

    # RSS
    "rss_feed": parse_rss_feed,
}


# -------------------------
# Selection to hit target_total / ratios / recency
# -------------------------

def compute_species_targets(cfg: Dict[str, Any], total: int) -> Dict[str, int]:
    ratios = cfg.get("species_targets") or {}
    wanted: Dict[str, int] = {}
    for sp, frac in ratios.items():
        try:
            sp = str(sp)
            frac = float(frac)
        except Exception:
            continue
        if sp not in ALLOWED_SPECIES:
            continue
        wanted[sp] = int(round(total * frac))

    for sp in ALLOWED_SPECIES:
        wanted.setdefault(sp, 0)

    s = sum(wanted.values())
    if s == 0:
        return wanted

    if s != total:
        order = ["Orca", "Humpback", "Great White Shark", "Sperm whale", "Blue Whale"]
        diff = total - s
        i = 0
        while diff != 0 and i < 1000:
            sp = order[i % len(order)]
            wanted[sp] = max(0, wanted.get(sp, 0) + (1 if diff > 0 else -1))
            diff = total - sum(wanted.values())
            i += 1

    return wanted

def select_candidates(cfg: Dict[str, Any], candidates: List[Candidate], tz_name: str) -> List[Candidate]:
    total = int(cfg.get("target_total", 80))
    min_recent_days = int(cfg.get("min_recent_days", 7))
    min_recent_fraction = float(cfg.get("min_recent_fraction", 0.5))

    now_dt = now_local(tz_name)
    wanted = compute_species_targets(cfg, total)

    by_species: Dict[str, List[Candidate]] = {sp: [] for sp in ALLOWED_SPECIES}
    for c in candidates:
        if c.species in by_species:
            by_species[c.species].append(c)
    for sp in by_species:
        by_species[sp].sort(key=lambda x: x.date, reverse=True)

    def is_recent(c: Candidate) -> bool:
        return (now_dt.date() - c.date.date()).days <= min_recent_days

    chosen: List[Candidate] = []
    chosen_keys = set()

    def pick_one(c: Candidate) -> None:
        k = (to_date_str(c.date), c.species, c.area, c.source)
        if k in chosen_keys:
            return
        chosen.append(c)
        chosen_keys.add(k)

    need_recent = int((total * min_recent_fraction) + 0.9999)  # ceil
    recent_count = 0

    species_order = sorted(wanted.keys(), key=lambda s: wanted[s], reverse=True)

    # A) Fill recent respecting targets
    progressed = True
    while recent_count < need_recent and progressed:
        progressed = False
        for sp in species_order:
            if wanted.get(sp, 0) <= 0:
                continue
            for c in by_species[sp]:
                if not is_recent(c):
                    break
                k = (to_date_str(c.date), c.species, c.area, c.source)
                if k in chosen_keys:
                    continue
                pick_one(c)
                wanted[sp] -= 1
                recent_count += 1
                progressed = True
                break

    # B) Fill remaining by species targets (newest first)
    for sp in species_order:
        while wanted.get(sp, 0) > 0:
            found = False
            for c in by_species[sp]:
                k = (to_date_str(c.date), c.species, c.area, c.source)
                if k in chosen_keys:
                    continue
                pick_one(c)
                wanted[sp] -= 1
                found = True
                break
            if not found:
                break

    # C) If still short, fill from any remaining newest
    if len(chosen) < total:
        remaining: List[Candidate] = []
        for sp in by_species:
            for c in by_species[sp]:
                k = (to_date_str(c.date), c.species, c.area, c.source)
                if k not in chosen_keys:
                    remaining.append(c)
        remaining.sort(key=lambda c: c.date, reverse=True)
        for c in remaining:
            if len(chosen) >= total:
                break
            pick_one(c)

    chosen.sort(key=lambda c: c.date, reverse=True)
    return chosen[:total]


# -------------------------
# JSON output
# -------------------------

def build_entries(candidates: List[Candidate]) -> List[Dict[str, Any]]:
    candidates = sorted(candidates, key=lambda c: c.date, reverse=True)

    entries: List[Dict[str, Any]] = []
    counters: Dict[Tuple[str, str], int] = {}

    for c in candidates:
        date_str = to_date_str(c.date)
        region_base = slugify((c.area or "").split("(")[0])[:20]
        region = region_base if region_base else "region"
        skey = species_key(c.species)

        k = (date_str, skey)
        counters[k] = counters.get(k, 0) + 1
        idx = counters[k]
        entry_id = f"{date_str}-{region}-{skey}-{idx:02d}"

        entries.append(
            {
                "id": entry_id,
                "name": c.name,
                "species": c.species,
                "info": c.info,
                "date": date_str,
                "latitude": float(c.latitude),
                "longitude": float(c.longitude),
                "area": c.area,
                "source": c.source,
                "behaviors": c.behaviors,
            }
        )

    entries.sort(key=lambda e: e["date"], reverse=True)
    return entries


# -------------------------
# Config + main
# -------------------------

def load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"YAML parse error in {path}: {e}") from e

def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "config", "sources.yml")
    cfg = load_config(cfg_path)

    tz_name = cfg.get("timezone", "America/Vancouver")
    now_dt = now_local(tz_name)
    max_days = int(cfg.get("max_days", 14))

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "WhaleSightingsBot/1.0 (GitHub Actions; contact: actions@users.noreply.github.com)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    all_candidates: List[Candidate] = []

    for s in cfg.get("sources", []):
        source_key = str(s.get("key", "unknown")).strip() or "unknown"
        parser_name = s.get("parser")
        url = s.get("url", "")

        if not parser_name or parser_name not in PARSERS:
            print(f"SKIP: {source_key} (unknown parser: {parser_name}) url={url}")
            continue

        try:
            items = PARSERS[parser_name](s, session, tz_name, source_key)
            items = [c for c in items if within_window(c.date, now_dt, max_days)]
            all_candidates.extend(items)
            print(f"OK: {source_key} parser={parser_name} -> {len(items)} candidates")
        except Exception as e:
            print(f"WARN: {source_key} failed: {e}")

    filtered = [c for c in all_candidates if within_window(c.date, now_dt, max_days)]
    selected = select_candidates(cfg, filtered, tz_name)
    entries = build_entries(selected)

    out_path = os.path.join(root, "whale_sightings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(entries)} entries to whale_sightings.json")

if __name__ == "__main__":
    main()
