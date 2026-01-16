#!/usr/bin/env python3
"""Update whale_sightings.json from curated sources.

This is intentionally deterministic:
- It does NOT "search the whole web".
- It polls a curated list of known sources and parses them.

Add more sources + parsers over time to reach your ~80 entries and global distribution.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz

ALLOWED_SPECIES = {"Orca", "Humpback", "Sperm whale", "Great White Shark", "Blue Whale"}

# --- Helpers

def now_local(tz_name: str) -> datetime:
    return datetime.now(tz.gettz(tz_name))


def to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def clamp_nudge(lat: float, lon: float, dlat: float, dlon: float) -> Tuple[float, float]:
    """Apply at most 0.05 degrees total movement."""
    # limit each component to [-0.05, 0.05]
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


def infer_behaviors(text: str) -> List[str]:
    t = text.lower()
    out = ["reported"]
    if any(k in t for k in ["hunt", "predation", "prey", "kill"]):
        out.append("hunting")
    if any(k in t for k in ["feed", "lunge", "krill"]):
        out.append("feeding")
    if any(k in t for k in ["breach", "breaching"]):
        out.append("breaching")
    if any(k in t for k in ["call", "hydrophone", "vocal"]):
        out.append("vocalizing")
    if any(k in t for k in ["travel", "headed", "northbound", "southbound"]):
        out.append("traveling")
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for b in out:
        if b not in seen:
            deduped.append(b)
            seen.add(b)
    return deduped


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


# --- Parsers

def parse_orcanetwork_recent_sightings(cfg: Dict[str, Any], session: requests.Session, tz_name: str) -> List[Candidate]:
    """Parses Orca Network 'Recent Whale Sightings' page.

    Notes:
    - The page contains multiple species; we filter to allowed ones.
    - Dates appear as "January 12" headings; we infer year from *today*.
    """

    url = cfg["url"]
    html = session.get(url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    # Get visible text with separators so regex is easier
    text = soup.get_text("\n")

    # Find blocks that start with a month-day heading.
    # Example lines in the page:
    # "January 12" then "BIGG'S KILLER WHALES ..." then bullet items.

    month_re = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    heading_pat = re.compile(rf"^(?P<month>{month_re})\s+(?P<day>\d{{1,2}})\s*$", re.MULTILINE)

    # Split into sections by headings
    matches = list(heading_pat.finditer(text))
    if not matches:
        return []

    today = now_local(tz_name)
    year = today.year

    # Map some common location phrases to safer offshore coords
    # (These are intentionally mid-channel/offshore to avoid land pins.)
    loc_coords = {
        "admiralty inlet": (48.15, -122.80),
        "har o strait": (48.50, -123.20),
        "haro strait": (48.50, -123.20),
        "puget sound": (47.90, -122.55),
        "san juans": (48.55, -123.05),
        "san juan": (48.55, -123.05),
        "possession sound": (47.95, -122.35),
        "hood canal": (47.70, -122.95),
        "saratoga passage": (48.10, -122.45),
    }

    def best_coords(block: str) -> Tuple[float, float, bool]:
        b = block.lower()
        for k, (la, lo) in loc_coords.items():
            if k in b:
                return la, lo, False
        # fallback
        return float(cfg["latitude"]), float(cfg["longitude"]), True

    out: List[Candidate] = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        month_name = m.group("month")
        day = int(m.group("day"))
        # Build a date in local tz
        dt = datetime.strptime(f"{year} {month_name} {day}", "%Y %B %d")
        dt = dt.replace(tzinfo=tz.gettz(tz_name))

        # If the inferred date is "in the future" relative to today (year boundary), subtract a year.
        if dt.date() > today.date():
            dt = dt.replace(year=year - 1)

        # Identify allowed species in the block
        # Orca Network labels include "KILLER WHALES" / "SOUTHERN RESIDENTS" etc.
        species_hits: List[Tuple[str, str]] = []  # (species, snippet)

        # Orca
        if re.search(r"KILLER\s+WHALES|SOUTHERN\s+RESIDENTS|ORCA\b", block, re.IGNORECASE):
            species_hits.append(("Orca", block))

        # Humpback
        if re.search(r"HUMPBACK", block, re.IGNORECASE):
            species_hits.append(("Humpback", block))

        # Sperm whale
        if re.search(r"SPERM\s+WHALE", block, re.IGNORECASE):
            species_hits.append(("Sperm whale", block))

        # Great white shark (unlikely here)
        if re.search(r"GREAT\s+WHITE|WHITE\s+SHARK", block, re.IGNORECASE):
            species_hits.append(("Great White Shark", block))

        # Blue whale optional
        if re.search(r"BLUE\s+WHALE", block, re.IGNORECASE):
            species_hits.append(("Blue Whale", block))

        # Create at most 2 entries per date from this source to avoid flooding
        for species, snippet in species_hits[:2]:
            if species not in ALLOWED_SPECIES:
                continue

            lat, lon, uncertain = best_coords(snippet)
            if uncertain and "uncertain_offshore_nudge" in cfg:
                nud = cfg["uncertain_offshore_nudge"]
                lat, lon = clamp_nudge(lat, lon, float(nud.get("dlat", 0.0)), float(nud.get("dlon", 0.0)))

            loc_name = cfg.get("area", "")
            # Try to extract a short location label
            loc_line = None
            for line in snippet.splitlines():
                if "-" in line and any(mon in line for mon in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
                    # Example: "Mon, Jan 12 - Admiralty Inlet - We saw them..."
                    loc_line = line
                    break
            if loc_line:
                parts = [p.strip() for p in loc_line.split("-")]
                if len(parts) >= 2:
                    loc_name = parts[1] or loc_name

            name = f"{species} sighting ({loc_name})" if loc_name else f"{species} sighting"
            # Clean up
            name = re.sub(r"\s+", " ", name).strip()

            info = f"Orca Network recent sightings page lists a {species.lower()} report for {loc_name} on {to_date_str(dt)}."

            out.append(
                Candidate(
                    date=dt,
                    species=species,
                    name=name,
                    info=info,
                    area=f"{loc_name} (offshore), Salish Sea region",
                    source=url,
                    latitude=lat,
                    longitude=lon,
                    behaviors=infer_behaviors(snippet),
                )
            )

    # Cap
    max_items = int(cfg.get("max_items", 9999))
    out.sort(key=lambda c: c.date, reverse=True)
    return out[:max_items]


PARSERS = {
    "orcanetwork_recent_sightings": parse_orcanetwork_recent_sightings,
}


# --- Main

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def within_window(dt: datetime, now_dt: datetime, max_days: int) -> bool:
    return (now_dt.date() - dt.date()).days <= max_days


def build_entries(candidates: List[Candidate], tz_name: str) -> List[Dict[str, Any]]:
    """Convert candidates into final JSON entries with id + required field order."""

    now_dt = now_local(tz_name)
    # Sort newest -> oldest
    candidates = sorted(candidates, key=lambda c: c.date, reverse=True)

    entries: List[Dict[str, Any]] = []
    # Ensure unique IDs
    counters: Dict[Tuple[str, str], int] = {}

    for c in candidates:
        date_str = to_date_str(c.date)
        # Use a short region slug derived from the area text (stable-ish).
        region_base = slugify(c.area.split("(")[0])[:20]
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

    # Final sort newest->oldest (stable)
    entries.sort(key=lambda e: e["date"], reverse=True)
    return entries


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "config", "sources.yml"))
    tz_name = cfg.get("timezone", "America/Vancouver")

    now_dt = now_local(tz_name)
    max_days = int(cfg.get("max_days", 14))

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "WhaleSightingsBot/1.0 (GitHub Actions; contact: you@example.com)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    all_candidates: List[Candidate] = []

    for s in cfg.get("sources", []):
        parser_name = s.get("parser")
        if not parser_name or parser_name not in PARSERS:
            continue
        try:
            items = PARSERS[parser_name](s, session, tz_name)
            all_candidates.extend(items)
        except Exception as e:
            print(f"WARN: source {s.get('key')} failed: {e}")

    # Filter to window
    filtered = [c for c in all_candidates if within_window(c.date, now_dt, max_days)]

    entries = build_entries(filtered, tz_name)

    out_path = os.path.join(root, "whale_sightings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(entries)} entries to whale_sightings.json")


if __name__ == "__main__":
    main()
