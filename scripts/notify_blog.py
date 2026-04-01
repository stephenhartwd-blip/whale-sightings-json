#!/usr/bin/env python3
"""
notify_blog.py - Checks Whale Tracker blog RSS and sends FCM push notification on new posts.
Runs every 15 min via GitHub Actions. Requires FIREBASE_SERVICE_ACCOUNT_JSON secret.

This version sends:
- notification title/body for all users
- data payload with:
    type      = "blog_post"
    articleId = latest post GUID
    title     = latest post title
    url       = latest post URL

That lets the app:
- show the push headline to everyone
- later open the article in-app by articleId
- still keep url as a fallback
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import feedparser
import requests

BLOG_RSS_URL = "https://www.whaletrackerapp.com/blog-feed.xml"
GUID_FILE = Path(__file__).parent.parent / "last_blog_post_guid.txt"
FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"


def get_access_token(sa_json_str: str) -> str:
    from google.oauth2 import service_account
    import google.auth.transport.requests

    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json_str),
        scopes=[FCM_SCOPE],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def send_fcm(
    project_id: str,
    token: str,
    push_title: str,
    post_title: str,
    url: str,
    article_id: str,
) -> bool:
    payload = {
        "message": {
            "topic": "whale_alerts",
            "notification": {
                "title": push_title,
                "body": post_title,
            },
            "apns": {
                "payload": {
                    "aps": {
                        "sound": "default",
                        "badge": 1,
                    }
                }
            },
            "data": {
                "type": "blog_post",
                "articleId": article_id,
                "title": post_title,
                "url": url,
            },
        }
    }

    resp = requests.post(
        f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=15,
    )

    ok = resp.status_code == 200
    print(("OK" if ok else "FAIL") + f" FCM {resp.status_code}: {post_title!r}")

    if not ok:
        print(resp.text)

    return ok


def main() -> None:
    print(f"Checking blog RSS: {BLOG_RSS_URL}")
    feed = feedparser.parse(BLOG_RSS_URL)

    if not feed.entries:
        print("No entries — skipping")
        return

    latest = feed.entries[0]

    guid = (latest.get("id") or latest.get("guid") or latest.get("link") or "").strip()
    title = (latest.get("title") or "New whale update").strip()
    link = (latest.get("link") or "https://www.whaletrackerapp.com/blog").strip()

    if not guid:
        print("ERROR: Latest post has no guid/id/link to use as articleId")
        sys.exit(1)

    last_guid = GUID_FILE.read_text(encoding="utf-8").strip() if GUID_FILE.exists() else ""

    print(f"Latest: {title!r}")
    print(f"GUID:   {guid!r}")

    if guid == last_guid:
        print("No new post — done")
        return

    print(f"New post detected: {title!r}")

    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        print("No FIREBASE_SERVICE_ACCOUNT_JSON — saving guid only")
        GUID_FILE.write_text(guid, encoding="utf-8")
        return

    try:
        sa_info = json.loads(sa_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: FIREBASE_SERVICE_ACCOUNT_JSON is invalid JSON: {e}")
        sys.exit(1)

    project_id = sa_info.get("project_id", "").strip()
    if not project_id:
        print("ERROR: project_id missing from service account JSON")
        sys.exit(1)

    token = get_access_token(sa_json)

    ok = send_fcm(
        project_id=project_id,
        token=token,
        push_title="New on Whale Tracker",
        post_title=title,
        url=link,
        article_id=guid,
    )

    if ok:
        GUID_FILE.write_text(guid, encoding="utf-8")
        print("Done.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
