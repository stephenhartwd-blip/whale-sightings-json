#!/usr/bin/env python3
"""
notify_blog.py - Checks Whale Tracker blog RSS, writes the latest post to Firestore,
and sends an FCM push notification on new posts.

Runs every 15 min via GitHub Actions.
Requires FIREBASE_SERVICE_ACCOUNT_JSON secret.

This version:
- checks the Whale Tracker blog RSS feed
- prevents duplicate sends with last_blog_post_guid.txt
- writes the latest post into Firestore collection: blog_posts
- uses the RSS GUID as the Firestore document ID
- sends an FCM push with:
    type      = "blog_post"
    articleId = latest post GUID
    title     = latest post title
    url       = latest post URL
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import feedparser
import requests
from google.cloud import firestore
from google.oauth2 import service_account
import google.auth.transport.requests

BLOG_RSS_URL = "https://www.whaletrackerapp.com/blog-feed.xml"
GUID_FILE = Path(__file__).parent.parent / "last_blog_post_guid.txt"
FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"


def get_access_token(sa_json_str: str) -> str:
    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json_str),
        scopes=[FCM_SCOPE],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def get_firestore_client(sa_json_str: str) -> firestore.Client:
    info = json.loads(sa_json_str)
    creds = service_account.Credentials.from_service_account_info(info)
    return firestore.Client(project=info["project_id"], credentials=creds)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "article"


def clean_html_to_text(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_preview_and_content(entry) -> tuple[str, str]:
    summary_html = (
        entry.get("summary")
        or entry.get("description")
        or ""
    ).strip()

    preview = clean_html_to_text(summary_html)

    # First live version:
    # use summary for both preview and content so the app has real data now.
    # Later we can upgrade this to scrape the full article page.
    content = preview

    return preview, content


def upsert_blog_post(
    db: firestore.Client,
    article_id: str,
    title: str,
    subtitle: str,
    preview: str,
    content: str,
    image_url: str,
    source_url: str,
) -> None:
    slug = slugify(title)

    db.collection("blog_posts").document(article_id).set(
        {
            "title": title,
            "subtitle": subtitle,
            "preview": preview,
            "content": content,
            "imageURL": image_url,
            "sourceURL": source_url,
            "slug": slug,
            "isPublished": True,
            "createdAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    print(f"Saved Firestore blog post: {article_id!r}")


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

    preview, content = extract_preview_and_content(latest)
    subtitle = ""
    image_url = ""

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

    # 1) Write the article to Firestore first
    try:
        db = get_firestore_client(sa_json)
        upsert_blog_post(
            db=db,
            article_id=guid,
            title=title,
            subtitle=subtitle,
            preview=preview,
            content=content,
            image_url=image_url,
            source_url=link,
        )
    except Exception as e:
        print(f"ERROR writing blog post to Firestore: {e}")
        sys.exit(1)

    # 2) Then send the push
    try:
        token = get_access_token(sa_json)
    except Exception as e:
        print(f"ERROR getting FCM access token: {e}")
        sys.exit(1)

    ok = send_fcm(
        project_id=project_id,
        token=token,
        push_title="New on Whale Tracker",
        post_title=title,
        url=link,
        article_id=guid,
    )

    # 3) Only mark as sent if push succeeded
    if ok:
        GUID_FILE.write_text(guid, encoding="utf-8")
        print("Done.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
