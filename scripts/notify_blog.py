#!/usr/bin/env python3
"""
notify_blog.py - Checks Whale Tracker blog RSS and sends FCM push notification on new posts.
Runs every 15 min via GitHub Actions. Requires FIREBASE_SERVICE_ACCOUNT_JSON secret.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import feedparser, requests

BLOG_RSS_URL = "https://www.whaletrackerapp.com/blog-feed.xml"
GUID_FILE    = Path(__file__).parent.parent / "last_blog_post_guid.txt"
FCM_SCOPE    = "https://www.googleapis.com/auth/firebase.messaging"

def get_access_token(sa_json_str):
    from google.oauth2 import service_account
    import google.auth.transport.requests
    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json_str), scopes=[FCM_SCOPE])
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token

def send_fcm(project_id, token, title, body, url):
    resp = requests.post(
        f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"message": {"topic": "whale_alerts",
              "notification": {"title": title, "body": body},
              "apns": {"payload": {"aps": {"sound": "default", "badge": 1}}},
              "data": {"type": "blog_post", "url": url}}},
        timeout=15)
    ok = resp.status_code == 200
    print(("OK" if ok else "FAIL") + f" FCM {resp.status_code}: {title!r}")
    return ok

def main():
    print(f"Checking blog RSS: {BLOG_RSS_URL}")
    feed = feedparser.parse(BLOG_RSS_URL)
    if not feed.entries:
        print("No entries — skipping"); return

    latest   = feed.entries[0]
    guid     = (latest.get("id") or latest.get("link") or "").strip()
    title    = (latest.get("title") or "New whale update").strip()
    link     = (latest.get("link") or "https://www.whaletrackerapp.com/blog").strip()
    last_guid = GUID_FILE.read_text(encoding="utf-8").strip() if GUID_FILE.exists() else ""

    print(f"Latest: {title!r}")
    if guid == last_guid:
        print("No new post — done"); return

    print(f"New post: {title!r}")
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        print("No FIREBASE_SERVICE_ACCOUNT_JSON — saving guid only")
        GUID_FILE.write_text(guid, encoding="utf-8"); return

    sa_info    = json.loads(sa_json)
    project_id = sa_info.get("project_id", "")
    if not project_id:
        print("ERROR: project_id missing"); sys.exit(1)

    token = get_access_token(sa_json)
    ok    = send_fcm(project_id, token, "New on Whale Tracker", title, link)
    if ok:
        GUID_FILE.write_text(guid, encoding="utf-8")
        print("Done.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
