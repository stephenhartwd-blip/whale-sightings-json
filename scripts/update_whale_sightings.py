#!/usr/bin/env python3
"""
Update whale_sightings.json from curated sources.

Deterministic:
- Does NOT "search the whole web"
- Polls a curated list of sources in config/sources.yml

NEW:
- sources.yml can specify `url:` or `urls:` (list)
- Adds `generic_rss` parser (RSS/Atom)
- Adds target_total + species_targets selection with best-effort recency constraints
- More verbose logging for GitHub Actions
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz, parser as dateparser

ALLOWED_SPECIES = {"Orca", "Humpback", "Sperm whale", "Great White Shark", "Blue Whale"}


# -----------------------
# Helpers
# -----------------------

def now_local(tz_name: str) -> datetime:
    return datetime.now(tz.gettz(tz_name))


def to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def clamp_nudge(lat: float, lon: float, dlat: float, dlon: float) -> Tuple[float, float]:
    """Apply at most 0.05 degrees movement per component."""
    dlat = max(-0.05, min(0.05, dlat))
    dlon = max(-0.05, min(0.05, dlon))
    return lat + dlat, lon + dlon


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


def infer_species(text: str, force_species: Optional[str] = None) -> Optional[str]:
    """
    Best-effort species inference from text, unless force_species is set.
    Returns one of ALLOWED_SPECIES or None.
    """
    if force_species:
        fs = force_species.strip()
        return fs if fs in ALLOWED_SPECIES else None

    t = (text or "").lower()

    # Great white shark
    if "great white" in t or "white shark" in t:
        return "Great White Shark"

    # Blue whale
    if "blue whale" in t:
        return "Blue Whale"

    # Sperm whale
    if "sperm whale" in t or re.search(r"\bsperm\b", t):
        return "Sperm whale"

    # Humpback
    if "humpback" in t:
        return "Humpback"

    # Orca
    if "orca" in t or "killer whale" in t or "bigg" in t or "southern resident" in t:
        return "Orca"

    return None


def infer_behaviors(text: str) -> List[str]:
    t = (text or "").lower()
    out = ["reported"]
    if any(k in t for k in ["hunt", "predation", "prey", "kill"]):
        out.append("hunting")
    if any(k in t for k in ["feed", "feeding", "lunge", "krill"]):
        out.append("feeding")
    if any(k in t for k in ["breach", "breaching"]):
        out.append("breaching")
    if any(k in t for k in ["call", "hydrophone", "vocal"]):
        out.append("vocalizing")
    if any(k in t for k in ["travel", "headed", "northbound", "southbound", "migrat"]):
        out.append("traveling")

    # de-dupe while preserving order
    seen = set()
    deduped: List[str] = []
    for b in out:
        if b not in seen:
            deduped.append(b)
            seen.add(b)
    return deduped


def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def localname(tag: str) -> str:
    """Strip XML namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def first_child_text(node: ET.Element, want: str) -> Optional[str]:
    want = want.lower()
    for child in list(node):
        if localname(child.tag).lower() == want:
            if child.text:
                return child.text.strip()
            return None
    return None


def first_child_attr(node: ET.Element, want: str, attr: str) -> Optional[str]:
    want = want.lower()
    for child in list(node):
        if localname(child.tag).lower() == want:
            v = child.attrib.get(attr)
            return v.strip() if v else None
    return None


def parse_date_any(s: Optional[str], tz_name: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        # Ensure tz-aware in the chosen tz
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.gettz(tz_name))
        else:
            dt = dt.astimezone(tz.gettz(tz_name))
        return dt
    except Exception:
        return None


# -----------------------
# Data model
# -----------------------

@dataclass(frozen=True)
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

    def dedupe_key(self) -> Tuple[str, str, str, str, float, float]:
        # rounding makes dedupe stable across tiny float variations
        return (
            to_date_str(self.date),
            self.species,
            self.source,
            self.area,
            round(self.latitude, 4),
            round(self.longitude, 4),
        )


# -----------------------
# Parsers
# -----------------------

def parse_orcanetwork_recent_sightings(cfg: Dict[str, Any], session: requests.Session, tz_name: str) -> List[Candidate]:
    """
    Parses Orca Network 'Recent Whale Sightings' page.
    Still mostly Salish Sea, but it's a solid local anchor source.
    """
    url = cfg["url"]
    html = session.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

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
        b = block.lower()
        for k, (la, lo) in loc_coords.items():
            if k in b:
                return la, lo, False
        return safe_float(cfg.get("latitude"), 48.45), safe_float(cfg.get("longitude"), -123.05), True

    max_items = int(cfg.get("max_items", 9999))
    default_area = str(cfg.get("area", "")).strip() or "Salish Sea (offshore)"

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

        # Determine which species appear in this day's block (simple approach)
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

        # Create one candidate per species found that day
        for sp in species_hits:
            if sp not in ALLOWED_SPECIES:
                continue

            lat, lon, uncertain = best_coords(block)
            if uncertain and "uncertain_offshore_nudge" in cfg:
                nud = cfg["uncertain_offshore_nudge"]
                lat, lon = clamp_nudge(lat, lon, safe_float(nud.get("dlat"), 0.0), safe_float(nud.get("dlon"), 0.0))

            name = f"{sp} sighting ({default_area})"
            info = f"Orca Network recent sightings page lists a {sp.lower()} report for {default_area} on {to_date_str(dt)}."

            out.append(
                Candidate(
                    date=dt,
                    species=sp,
                    name=name,
                    info=info,
                    area=f"{default_area} (offshore)",
                    source=url,
                    latitude=float(lat),
                    longitude=float(lon),
                    behaviors=infer_behaviors(block),
                )
            )

    out.sort(key=lambda c: (c.date, c.species), reverse=True)
    return out[:max_items]


def parse_generic_rss(cfg: Dict[str, Any], session: requests.Session, tz_name: str) -> List[Candidate]:
    """
    Generic RSS/Atom feed parser.

    Config fields used:
      - url (expanded from url/urls in main)
      - area, latitude, longitude (fallback/offshore)
      - force_species (optional): one of ALLOWED_SPECIES
      - max_items (optional)
      - uncertain_offshore_nudge (optional)
    """
    url = cfg["url"]
    resp = session.get(url, timeout=30)
    content = resp.content.decode("utf-8", errors="ignore").strip()
    if not content:
        return []

    max_items = int(cfg.get("max_items", 9999))
    default_area = str(cfg.get("area", "")).strip() or "Unknown (offshore)"
    fallback_lat = safe_float(cfg.get("latitude"), 0.0)
    fallback_lon = safe_float(cfg.get("longitude"), 0.0)
    force_species = cfg.get("force_species")

    def maybe_nudge(lat: float, lon: float) -> Tuple[float, float]:
        if "uncertain_offshore_nudge" in cfg:
            nud = cfg["uncertain_offshore_nudge"]
            return clamp_nudge(lat, lon, safe_float(nud.get("dlat"), 0.0), safe_float(nud.get("dlon"), 0.0))
        return lat, lon

    try:
        root = ET.fromstring(content)
    except Exception as e:
        print(f"WARN: generic_rss could not parse XML for {url}: {e}")
        return []

    root_name = localname(root.tag).lower()

    out: List[Candidate] = []

    # RSS: <rss><channel><item>...
    if root_name == "rss" or root.find(".//channel") is not None:
        items = root.findall(".//item")
        for item in items:
            title = first_child_text(item, "title") or ""
            link = first_child_text(item, "link") or url
            desc = first_child_text(item, "description") or first_child_text(item, "content") or ""
            pub = first_child_text(item, "pubDate") or first_child_text(item, "date") or first_child_text(item, "published")

            dt = parse_date_any(pub, tz_name)
            if dt is None:
                continue

            blob = f"{title}\n{desc}".strip()
            sp = infer_species(blob, force_species=force_species)
            if sp is None or sp not in ALLOWED_SPECIES:
                continue

            lat, lon = maybe_nudge(fallback_lat, fallback_lon)

            name = f"{sp} sighting ({default_area})"
            info = f"Feed item: {title}".strip() if title else f"Feed report for {default_area} on {to_date_str(dt)}."

            out.append(
                Candidate(
                    date=dt,
                    species=sp,
                    name=name,
                    info=info,
                    area=f"{default_area} (offshore)",
                    source=link if link else url,
                    latitude=float(lat),
                    longitude=float(lon),
                    behaviors=infer_behaviors(blob),
                )
            )

    # Atom: <feed><entry>...
    elif root_name == "feed":
        entries = [n for n in root.iter() if localname(n.tag).lower() == "entry"]
        for ent in entries:
            title = first_child_text(ent, "title") or ""
            link = first_child_attr(ent, "link", "href") or first_child_text(ent, "link") or url
            summary = first_child_text(ent, "summary") or first_child_text(ent, "content") or ""
            updated = first_child_text(ent, "updated") or first_child_text(ent, "published")

            dt = parse_date_any(updated, tz_name)
            if dt is None:
                continue

            blob = f"{title}\n{summary}".strip()
            sp = infer_species(blob, force_species=force_species)
            if sp is None or sp not in ALLOWED_SPECIES:
                continue

            lat, lon = maybe_nudge(fallback_lat, fallback_lon)

            name = f"{sp} sighting ({default_area})"
            info = f"Feed item: {title}".strip() if title else f"Feed report for {default_area} on {to_date_str(dt)}."

            out.append(
                Candidate(
                    date=dt,
                    species=sp,
                    name=name,
                    info=info,
                    area=f"{default_area} (offshore)",
                    source=link if link else url,
                    latitude=float(lat),
                    longitude=float(lon),
                    behaviors=infer_behaviors(blob),
                )
            )

    else:
        # Unknown feed shape
        return []

    out.sort(key=lambda c: (c.date, c.species, c.source), reverse=True)
    return out[:max_items]


PARSERS = {
    "orcanetwork_recent_sightings": parse_orcanetwork_recent_sightings,
    "generic_rss": parse_generic_rss,
}


# -----------------------
# Config + selection logic
# -----------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def within_window(dt: datetime, now_dt: datetime, max_days: int) -> bool:
    return (now_dt.date() - dt.date()).days <= max_days


def is_recent(dt: datetime, now_dt: datetime, min_recent_days: int) -> bool:
    return (now_dt.date() - dt.date()).days <= min_recent_days


def expand_source_urls(source_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Allow either:
      - url: "https://..."
      - urls: ["https://...", "https://..."]
    Expands into multiple dicts each with a concrete `url`.
    """
    urls: List[str] = []
    if isinstance(source_cfg.get("urls"), list):
        urls.extend([u for u in source_cfg.get("urls", []) if isinstance(u, str) and u.strip()])
    if isinstance(source_cfg.get("url"), str) and source_cfg["url"].strip():
        urls.append(source_cfg["url"].strip())

    urls = list(dict.fromkeys(urls))  # de-dupe, preserve order
    if not urls:
        return []

    if len(urls) == 1:
        sc = dict(source_cfg)
        sc["url"] = urls[0]
        return [sc]

    out: List[Dict[str, Any]] = []
    base_key = str(source_cfg.get("key", "source"))
    for i, u in enumerate(urls, start=1):
        sc = dict(source_cfg)
        sc["url"] = u
        sc["key"] = f"{base_key}-{i}"
        out.append(sc)
    return out


def target_counts_from_ratios(target_total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    # Only allow known species keys
    cleaned = {k: float(v) for k, v in ratios.items() if k in ALLOWED_SPECIES}
    if not cleaned:
        return {}

    # Initial floor allocation
    counts: Dict[str, int] = {k: int(math.floor(target_total * cleaned[k])) for k in cleaned.keys()}
    used = sum(counts.values())
    remainder = target_total - used
    if remainder <= 0:
        return counts

    # Distribute remainder by highest fractional part
    fracs = sorted(
        cleaned.keys(),
        key=lambda k: (target_total * cleaned[k] - math.floor(target_total * cleaned[k]), cleaned[k]),
        reverse=True,
    )
    idx = 0
    while remainder > 0 and fracs:
        k = fracs[idx % len(fracs)]
        counts[k] = counts.get(k, 0) + 1
        remainder -= 1
        idx += 1

    return counts


def select_candidates(
    candidates: List[Candidate],
    tz_name: str,
    target_total: int,
    species_targets: Dict[str, float],
    min_recent_days: int,
    min_recent_fraction: float,
) -> List[Candidate]:
    """
    Best-effort:
    - Try to hit target_total
    - Try to match species ratios
    - Try to ensure enough are within min_recent_days (fraction)
    """
    now_dt = now_local(tz_name)
    if target_total <= 0:
        return []

    # de-dupe candidates
    deduped_map: Dict[Tuple[str, str, str, str, float, float], Candidate] = {}
    for c in candidates:
        deduped_map[c.dedupe_key()] = c
    candidates = list(deduped_map.values())

    # stable sort newest first
    candidates.sort(key=lambda c: (c.date, c.species, c.source, c.area, c.name), reverse=True)

    recent_pool = [c for c in candidates if is_recent(c.date, now_dt, min_recent_days)]
    old_pool = [c for c in candidates if not is_recent(c.date, now_dt, min_recent_days)]

    min_recent_count = int(math.ceil(target_total * float(min_recent_fraction)))
    min_recent_count = max(0, min(target_total, min_recent_count))

    # Prepare per-species pools
    def pool_by_species(pool: List[Candidate]) -> Dict[str, List[Candidate]]:
        d: Dict[str, List[Candidate]] = {s: [] for s in ALLOWED_SPECIES}
        for c in pool:
            if c.species in d:
                d[c.species].append(c)
        for s in d:
            d[s].sort(key=lambda c: (c.date, c.source, c.area, c.name), reverse=True)
        return d

    recent_by = pool_by_species(recent_pool)
    all_by = pool_by_species(recent_pool + old_pool)

    picked: List[Candidate] = []
    picked_counts: Dict[str, int] = {s: 0 for s in ALLOWED_SPECIES}

    # Helper to pop N from a species pool
    def pop_n(pools: Dict[str, List[Candidate]], sp: str, n: int) -> List[Candidate]:
        if n <= 0:
            return []
        take = pools.get(sp, [])[:n]
        pools[sp] = pools.get(sp, [])[n:]
        return take

    # --- Step 1: satisfy minimum recent quota (best effort)
    if min_recent_count > 0:
        recent_targets = target_counts_from_ratios(min_recent_count, species_targets)

        # First pass: try to hit ratios from recent pool
        for sp, need in recent_targets.items():
            got = pop_n(recent_by, sp, need)
            for c in got:
                picked.append(c)
                picked_counts[sp] += 1

        # Fill remaining recent slots from any recent (newest first)
        while len(picked) < min_recent_count:
            # find next best recent candidate across species
            next_c: Optional[Candidate] = None
            next_sp: Optional[str] = None
            for sp in ALLOWED_SPECIES:
                if recent_by.get(sp):
                    cand = recent_by[sp][0]
                    if next_c is None or cand.date > next_c.date:
                        next_c = cand
                        next_sp = sp
            if next_c is None or next_sp is None:
                break
            recent_by[next_sp] = recent_by[next_sp][1:]
            picked.append(next_c)
            picked_counts[next_sp] += 1

    # --- Step 2: fill to target_total using overall ratio targets
    remaining = target_total - len(picked)
    if remaining <= 0:
        return picked[:target_total]

    total_targets = target_counts_from_ratios(target_total, species_targets)

    # subtract what we already picked
    needed_by_species: Dict[str, int] = {}
    for sp, tgt in total_targets.items():
        needed_by_species[sp] = max(0, int(tgt) - int(picked_counts.get(sp, 0)))

    # Use overall pool (recent leftovers + old) by species
    for sp, need in needed_by_species.items():
        got = pop_n(all_by, sp, need)
        for c in got:
            picked.append(c)
            picked_counts[sp] += 1

    # Fill any remaining slots with newest available across all species
    while len(picked) < target_total:
        next_c = None
        next_sp = None
        for sp in ALLOWED_SPECIES:
            if all_by.get(sp):
                cand = all_by[sp][0]
                if next_c is None or cand.date > next_c.date:
                    next_c = cand
                    next_sp = sp
        if next_c is None or next_sp is None:
            break
        all_by[next_sp] = all_by[next_sp][1:]
        picked.append(next_c)
        picked_counts[next_sp] += 1

    return picked[:target_total]


def build_entries(candidates: List[Candidate], tz_name: str) -> List[Dict[str, Any]]:
    """Convert candidates into final JSON entries with id + required field order."""
    # stable newest->oldest
    candidates = sorted(candidates, key=lambda c: (c.date, c.species, c.source, c.area, c.name), reverse=True)

    entries: List[Dict[str, Any]] = []
    counters: Dict[Tuple[str, str], int] = {}

    for c in candidates:
        date_str = to_date_str(c.date)
        region_base = slugify(c.area.split("(")[0])[:20]
        region = region_base if region_base else "region"
        skey = species_key(c.species)
        k = (date_str, skey)
        counters[k] = counters.get(k, 0) + 1
        idx = counters[k]
        entry_id = f"{date_str}-{region}-{skey}-{idx:02d}"

        # IMPORTANT: keep field order exactly as your Swift expects
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


# -----------------------
# Main
# -----------------------

def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "config", "sources.yml")
    cfg = load_config(cfg_path)

    tz_name = cfg.get("timezone", "America/Vancouver")
    now_dt = now_local(tz_name)

    max_days = int(cfg.get("max_days", 14))
    min_recent_days = int(cfg.get("min_recent_days", 7))
    min_recent_fraction = float(cfg.get("min_recent_fraction", 0.50))

    target_total = int(cfg.get("target_total", 80))
    species_targets = cfg.get("species_targets", {}) or {}

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "WhaleSightingsBot/1.1 (GitHub Actions)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    all_candidates: List[Candidate] = []

    sources_in = cfg.get("sources", []) or []
    expanded_sources: List[Dict[str, Any]] = []
    for s in sources_in:
        expanded_sources.extend(expand_source_urls(s))

    if not expanded_sources:
        print("WARN: No sources found in config/sources.yml (need url: or urls:).")
        # still write empty file to keep pipeline consistent
        out_path = os.path.join(root, "whale_sightings.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("Wrote 0 entries to whale_sightings.json")
        return

    print(f"Loaded {len(expanded_sources)} source entries (after expanding urls).")

    for s in expanded_sources:
        key = str(s.get("key", "source"))
        url = str(s.get("url", "")).strip()
        parser_name = s.get("parser")

        if not parser_name or parser_name not in PARSERS:
            print(f"SKIP: {key} -> unknown parser '{parser_name}' (url={url})")
            continue

        try:
            items = PARSERS[parser_name](s, session, tz_name)
            print(f"SRC: {key} ({parser_name}) url={url} -> {len(items)} candidates")
            all_candidates.extend(items)
        except Exception as e:
            print(f"WARN: source {key} failed: {e}")

    # Filter to date window
    windowed = [c for c in all_candidates if within_window(c.date, now_dt, max_days)]
    print(f"Window filter: {len(all_candidates)} -> {len(windowed)} within last {max_days} days")

    # Select subset to hit target_total/ratios/recency
    selected = select_candidates(
        windowed,
        tz_name=tz_name,
        target_total=target_total,
        species_targets=species_targets,
        min_recent_days=min_recent_days,
        min_recent_fraction=min_recent_fraction,
    )

    if len(selected) < target_total:
        print(
            f"WARN: Only {len(selected)} entries available after selection (target_total={target_total}). "
            f"Add more sources (or increase max_days)."
        )

    # Output
    entries = build_entries(selected, tz_name)
    out_path = os.path.join(root, "whale_sightings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # quick stats for logs
    by_species: Dict[str, int] = {}
    recent_count = 0
    for c in selected:
        by_species[c.species] = by_species.get(c.species, 0) + 1
        if is_recent(c.date, now_dt, min_recent_days):
            recent_count += 1

    print(f"Wrote {len(entries)} entries to whale_sightings.json")
    print(f"Recent (<= {min_recent_days}d): {recent_count}/{len(entries)}")
    print("Species counts:", dict(sorted(by_species.items(), key=lambda kv: kv[0])))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)
