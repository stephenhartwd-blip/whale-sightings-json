#!/usr/bin/env python3
"""
notify_sightings.py
Uploads whale_sightings.json and whale_trails.json to Firestore
after each GitHub Actions run, enabling real-time updates in the app.

Firestore structure:
  sightings/current  -> { whales: [...], updatedAt: timestamp, count: N }
  sightings/trails   -> { trails: {...}, updatedAt: timestamp, count: N }
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from google.cloud import firestore
from google.oauth2 import service_account

SIGHTINGS_PATH = Path(__file__).parent.parent / "whale_sightings.json"
TRAILS_PATH    = Path(__file__).parent.parent / "whale_trails.json"

# Timmy watch — released into Skagerrak May 2 2026, GPS failed, unconfirmed since
TIMMY_LAT_MIN, TIMMY_LAT_MAX = 54.0, 70.0
TIMMY_LNG_MIN, TIMMY_LNG_MAX = 4.0, 15.0
TIMMY_CHAT_ID = "6909590199"
TIMMY_SEEN_PATH  = Path(__file__).parent.parent / "timmy_seen.json"
TIMMY_CANDIDATES_PATH = Path(__file__).parent.parent / "timmy_candidates.json"


def _load_timmy_seen() -> set:
    if TIMMY_SEEN_PATH.exists():
        try:
            return set(json.loads(TIMMY_SEEN_PATH.read_text()))
        except Exception:
            return set()
    return set()


def _save_timmy_seen(seen: set) -> None:
    TIMMY_SEEN_PATH.write_text(json.dumps(sorted(seen)))


def _is_timmy_candidate(s: dict) -> bool:
    lat, lng = s.get("latitude"), s.get("longitude")
    if lat is None or lng is None:
        return False
    if "humpback" not in (s.get("species") or "").lower():
        return False
    return (TIMMY_LAT_MIN <= lat <= TIMMY_LAT_MAX and
            TIMMY_LNG_MIN <= lng <= TIMMY_LNG_MAX)


def _log_timmy_candidate(s: dict) -> None:
    candidates: list = []
    if TIMMY_CANDIDATES_PATH.exists():
        try:
            candidates = json.loads(TIMMY_CANDIDATES_PATH.read_text())
        except Exception:
            candidates = []
    candidates.append({**s, "flagged_at": datetime.now(timezone.utc).isoformat()})
    TIMMY_CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2))


def _send_timmy_alert(s: dict) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("[TIMMY] No TELEGRAM_BOT_TOKEN — alert skipped")
        return
    lat  = s.get("latitude", "?")
    lon  = s.get("longitude", "?")
    area = s.get("area") or "unknown location"
    date = s.get("date") or "unknown date"
    source = s.get("source_key") or s.get("source") or "unknown"
    sid  = s.get("id", "?")
    msg = (
        "🚨 POSSIBLE TIMMY SIGHTING\n"
        f"📍 {lat}, {lon} ({area})\n"
        f"🕐 {date}\n"
        f"📡 Source: {source}\n"
        f"🆔 ID: {sid}\n\n"
        "This could be the celebrity humpback released May 2. Verify before posting."
    )
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": TIMMY_CHAT_ID, "text": msg},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[TIMMY] Alert sent → {TIMMY_CHAT_ID} ({area})")
    except Exception as e:
        print(f"[TIMMY] Alert failed: {e}")


def check_timmy(whales: list) -> None:
    seen = _load_timmy_seen()
    new_candidates = [s for s in whales
                      if _is_timmy_candidate(s) and s.get("id") not in seen]
    if not new_candidates:
        print("[TIMMY] No new candidates in watch zone (54–70°N, 4–15°E)")
        return
    print(f"[TIMMY] 🚨 {len(new_candidates)} new candidate(s)!")
    for s in new_candidates:
        _log_timmy_candidate(s)
        _send_timmy_alert(s)
        seen.add(s.get("id"))
    _save_timmy_seen(seen)


def get_firestore_client(sa_json_str: str) -> firestore.Client:
    info  = json.loads(sa_json_str)
    creds = service_account.Credentials.from_service_account_info(info)
    return firestore.Client(project=info["project_id"], credentials=creds)


def main() -> None:
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        print("No FIREBASE_SERVICE_ACCOUNT_JSON — skipping")
        return

    try:
        db = get_firestore_client(sa_json)
    except Exception as e:
        print(f"ERROR creating Firestore client: {e}")
        sys.exit(1)

    # Upload sightings
    if not SIGHTINGS_PATH.exists():
        print(f"ERROR: {SIGHTINGS_PATH} not found")
        sys.exit(1)

    try:
        whales = json.loads(SIGHTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR reading sightings JSON: {e}")
        sys.exit(1)

    check_timmy(whales)

    try:
        db.collection("sightings").document("current").set({
            "whales":    whales,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "count":     len(whales),
        })
        print(f"Uploaded {len(whales)} sightings to Firestore sightings/current")
    except Exception as e:
        print(f"ERROR uploading sightings: {e}")
        sys.exit(1)

    # Upload trails
    if not TRAILS_PATH.exists():
        print("No whale_trails.json — skipping trails upload")
        return

    try:
        trails = json.loads(TRAILS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR reading trails JSON: {e}")
        return

    try:
        db.collection("sightings").document("trails").set({
            "trails":    trails,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "count":     len(trails),
        })
        print(f"Uploaded {len(trails)} whale trails to Firestore sightings/trails")
    except Exception as e:
        print(f"ERROR uploading trails: {e}")


if __name__ == "__main__":
    main()
