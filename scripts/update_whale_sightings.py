#!/usr/bin/env python3
"""
Update whale_sightings.json from curated sources.

Deterministic:
- No "search the whole web"
- Pulls from curated URLs in config/sources.yml
- Parses pages with a small set of parsers (including a robust generic parser + RSS/Atom feeds)

Outputs JSON array entries that match your Swift model:
id, name, species, info, date, latitude, longitude, area, source, behaviors
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz

# NEW: RSS/Atom parsing
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
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return normalize_text(r.text)

def within_window(dt: datetime, now_dt: datetime, max_days: int) -> bool:
    return (now_dt.date() - dt.date()).days <= max_days

def parse_date_from_url(url: str, tz_name: str) -> Optional[datetime]:
    # patterns like /2026/1/14/ or /2026-01-14/
    m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})(?:/|$)", url or "")
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, tzinfo=tz.gettz(tz_name))
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", url or "")
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, tzinfo=tz.gettz(tz_name))
    return None

def best_effort_parse_date(s: str, tz_name: str, today: datetime) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None

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
        if dt.date() > today.date() and not m.group(3):
            dt = dt.replace(year=year - 1)
        return dt

    return None

def detect_species(text: str, force: Optional[List[str]] = None) -> List[str]:
    if force:
        return [s for s in force if s in ALLOWED_SPECIES]

    t = (text or "").lower()
    hits: List[str] = []

    if any(k in t for k in ["orca", "killer whale", "killer whales", "bigg", "southern resident"]):
        hits.append("Orca")
    if "humpback" in t:
        hits.append("Humpback")
    if "sperm whale" in t or re.search(r"\bsperm\b", t):
        hits.append("Sperm whale")
    if any(k in t for k in ["great white", "white shark"]):
        hits.append("Great White Shark")
    if "blue whale" in t:
        hits.append("Blue Whale")

    # de-dupe preserve order
    seen = set()
    out: List[str] = []
    for s in hits:
        if s in ALLOWED_SPECIES and s not in seen:
            out.append(s)
            seen.add(s)
    return out

def _force_species_list(cfg: Dict[str, Any]) -> Optional[List[str]]:
    fs = cfg.get("force_species")
    if fs is None:
        return None
    if isinstance(fs, list):
        return [str(x) for x in fs]
    return [str(fs)]

def _apply_nudge(cfg: Dict[str, Any], lat0: float, lon0: float) -> Tuple[float, float]:
    nud = cfg.get("uncertain_offshore_nudge") or {}
    return clamp_nudge(lat0, lon0, float(nud.get("dlat", 0.0)), float(nud.get("dlon", 0.0)))

def _safe_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _parsed_struct_to_dt(tstruct: Any, tz_name: str) -> Optional[datetime]:
    """
    feedparser gives time.struct_time in *_parsed fields.
    Convert to timezone-aware datetime in tz_name.
    """
    if not tstruct:
        return None
    try:
        dt_utc = datetime(
            tstruct.tm_year, tstruct.tm_mon, tstruct.tm_mday,
            tstruct.tm_hour, tstruct.tm_min, tstruct.tm_sec,
            tzinfo=tz.tzutc(),
        )
        return dt_utc.astimezone(tz.gettz(tz_name))
    except Exception:
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
# Generic “dated sections” extraction
# -------------------------

def extract_dated_sections(text: str, tz_name: str, today: datetime) -> List[Tuple[datetime, str]]:
    """
    Finds date markers in text and splits into sections.
    Returns list of (datetime, section_text).
    """
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",
        rf"\b(?:{'|'.join(MONTHS)})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b",
    ]

    matches: List[Tuple[int, int, str]] = []
    for pat in patterns:
        for m in re.finditer(pat, text or ""):
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
        section = (text or "")[section_start:section_end].strip()
        if len(section) < 20:
            continue
        out.append((dt, section))

    return out

def build_candidates_from_generic_source(
    cfg: Dict[str, Any],
    session: requests.Session,
    tz_name: str,
    source_key: str,
) -> List[Candidate]:
    url = cfg["url"]
    area = _safe_str(cfg.get("area"))
    lat0 = float(cfg.get("latitude", 0.0))
    lon0 = float(cfg.get("longitude", 0.0))
    max_items = int(cfg.get("max_items", 10))

    text = fetch_text(session, url)
    today = now_local(tz_name)

    force_species = _force_species_list(cfg)

    sections = extract_dated_sections(text, tz_name, today)
    candidates: List[Candidate] = []

    if sections:
        for dt, section in sections:
            sp = detect_species(section, force=force_species)
            for species in sp[:2]:
                name = f"{species} sighting ({area})" if area else f"{species} sighting"
                info = f"Parsed from {source_key} on {to_date_str(dt)}."
                lat, lon = _apply_nudge(cfg, lat0, lon0)
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
        dt = parse_date_from_url(url, tz_name) or today
        sp = detect_species(text, force=force_species)
        for species in sp[:2]:
            name = f"{species} sighting ({area})" if area else f"{species} sighting"
            info = f"Parsed from {source_key} (single-page) on {to_date_str(dt)}."
            lat, lon = _apply_nudge(cfg, lat0, lon0)
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
# NEW: RSS/Atom parser
# -------------------------

def parse_rss_feed(cfg: Dict[str, Any], session: requests.Session, tz_name: str, source_key: str) -> List[Candidate]:
    """
    Parses RSS/Atom feed into candidates.

    Add a source like:
      - key: monterey_feed
        parser: rss_feed
        url: https://example.com/feed/
        area: Monterey Bay (offshore), CA, USA
        latitude: 36.80
        longitude: -122.10
        max_items: 12
        force_species: ["Humpback"]  # optional
    """
    feed_url = cfg["url"]
    area = _safe_str(cfg.get("area"))
    lat0 = float(cfg.get("latitude", 0.0))
    lon0 = float(cfg.get("longitude", 0.0))
    max_items = int(cfg.get("max_items", 12))
    force_species = _force_species_list(cfg)

    # Fetch ourselves (so headers/timeout are consistent), then feedparser.parse bytes
    r = session.get(feed_url, timeout=30)
    r.raise_for_status()

    parsed = feedparser.parse(r.content)
    entries = parsed.entries or []

    today = now_local(tz_name)
    candidates: List[Candidate] = []
    seen = set()

    for e in entries:
        title = _safe_str(getattr(e, "title", None) or e.get("title"))
        summary = _safe_str(getattr(e, "summary", None) or e.get("summary") or e.get("description"))
        link = _safe_str(getattr(e, "link", None) or e.get("link")) or feed_url

        text = (title + "\n" + summary).strip()

        # Date: prefer parsed struct_time, then try string fields, then URL
        dt = (
            _parsed_struct_to_dt(e.get("published_parsed"), tz_name)
            or _parsed_struct_to_dt(e.get("updated_parsed"), tz_name)
            or _parsed_struct_to_dt(e.get("created_parsed"), tz_name)
        )

        if not dt:
            # try string fields like "published" / "updated"
            dt = (
                best_effort_parse_date(_safe_str(e.get("published")), tz_name, today)
                or best_effort_parse_date(_safe_str(e.get("updated")), tz_name, today)
                or parse_date_from_url(link, tz_name)
                or today
            )

        sp = detect_species(text, force=force_species)
        if not sp:
            continue

        lat, lon = _apply_nudge(cfg, lat0, lon0)

        # at most 2 species per feed item
        for species in sp[:2]:
            dedupe_key = (link, to_date_str(dt), species)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            name = f"{species} sighting ({area})" if area else f"{species} sighting"
            info = f"From RSS feed {source_key} on {to_date_str(dt)}: {title or 'Update'}"

            candidates.append(
                Candidate(
                    date=dt,
                    species=species,
                    name=name,
                    info=info,
                    area=area,
                    source=link,
                    latitude=float(lat),
                    longitude=float(lon),
                    behaviors=infer_behaviors(text),
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
    area_default = _safe_str(cfg.get("area"))
    lat_fallback = float(cfg.get("latitude", 0.0))
    lon_fallback = float(cfg.get("longitude", 0.0))
    max_items = int(cfg.get("max_items", 18))

    html = session.get(url, timeout=30).text
    text = normalize_text(html)

    month_re = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    heading_pat = re.compile(rf"^(?P<month>{month_re})\s+(?P<day>\d{{1,2}})\s*$", re.MULTILINE)

    matches = list(heading_pat.finditer(text))
    if not matches:
        return []

    today = now_local(tz_name)
    year = today.year

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
        b = (block or "").lower()
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

        dt = datetime.strptime(f"{year} {month_name} {day}", "%Y %B %d")
        dt = dt.replace(tzinfo=tz.gettz(tz_name))
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
                lat, lon = _apply_nudge(cfg, lat, lon)

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

def parse_generic(cfg: Dict[str, Any], session: requests.Session, tz_name: str, source_key: str) -> List[Candidate]:
    return build_candidates_from_generic_source(cfg, session, tz_name, source_key)

PARSERS = {
    # Orca Network
    "orcanetwork_recent_sightings": parse_orcanetwork_recent_sightings,

    # RSS/Atom
    "rss_feed": parse_rss_feed,
    "atom_feed": parse_rss_feed,

    # Your other YAML-listed parsers -> generic for now
    "eaglewing_daily_sighting_report": parse_generic,
    "victoriawhalewatching_captains_log": parse_generic,
    "montereybay_sightings": parse_generic,
    "dolphinsafari_sightings_log": parse_generic,
    "hawaiianadventures_weekly_whale_report": parse_generic,
    "whalewatchingakureyri_blog": parse_generic,
    "elding_whale_diary": parse_generic,
    "seatrips_whale_diary": parse_generic,
    "whalewatchwesternaustralia_daily": parse_generic,
    "pirsa_shark_sightings": parse_generic,
    "article_basic": parse_generic,
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
            if sp in wanted:
                wanted[sp] += 1 if diff > 0 else -1
                diff = total - sum(wanted.values())
            i += 1

        for sp in list(wanted.keys()):
            wanted[sp] = max(0, wanted[sp])

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

    need_recent = int((total * min_recent_fraction) + 0.9999)
    recent_count = 0

    species_order = sorted(wanted.keys(), key=lambda s: wanted[s], reverse=True)

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
        source_key = _safe_str(s.get("key")) or "unknown"
        parser_name = s.get("parser")
        url = _safe_str(s.get("url"))

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
