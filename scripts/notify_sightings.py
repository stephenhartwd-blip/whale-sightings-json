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
from pathlib import Path

from google.cloud import firestore
from google.oauth2 import service_account

SIGHTINGS_PATH = Path(__file__).parent.parent / "whale_sightings.json"
TRAILS_PATH    = Path(__file__).parent.parent / "whale_trails.json"


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
