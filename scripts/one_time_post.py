#!/usr/bin/env python3
"""one_time_post.py - Posts the founding article to Firestore blog_posts."""
import json, os, sys
from google.cloud import firestore
from google.oauth2 import service_account

sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
if not sa_json:
    print("No FIREBASE_SERVICE_ACCOUNT_JSON")
    sys.exit(1)

info  = json.loads(sa_json)
creds = service_account.Credentials.from_service_account_info(info)
db    = firestore.Client(project=info["project_id"], credentials=creds)

SLUG  = "we-built-this-because-we-actually-care"
TITLE = "We Built This Because We Actually Care"
PREVIEW = "There\'s a moment that changes you. It happened somewhere off the coast of British Columbia — watching a pod of orcas move through the Salish Sea at dusk, completely silent except for the water."

CONTENT = """There's a moment that changes you. It happened to us somewhere off the coast of British Columbia — watching a pod of orcas move through the Salish Sea at dusk, completely silent except for the water. No app open. No notifications. Just the kind of quiet that makes you realize these animals have been navigating these waters for thousands of years, and somehow we've made it harder for them to do that.

That moment is why Whale Tracker exists.

We're a small independent team based in Vancouver. We're not a wildlife organization, we're not funded by a government program, and we don't have a fleet of research vessels. What we have is a deep, genuine love for the ocean and the animals in it — and the technical skills to build something that might actually help.

We started with a simple frustration: whale sightings were scattered everywhere. Forums, Facebook groups, whale watch company blogs, iNaturalist observations, research databases. If you wanted to know where whales had been spotted recently, you had to check six different places and piece it together yourself. That felt wrong. So we fixed it.

Whale Tracker pulls from over 60 sources around the world — whale watching operators in Iceland, shark spotters in Cape Town, citizen scientists reporting through iNaturalist, acoustic detection systems in the Salish Sea, even the same data feed that powers the alert system used by commercial vessels to avoid striking whales in BC waters. Every 15 minutes, that data updates automatically. The map you're looking at right now reflects the actual state of whale activity across the planet as of the last quarter hour.

But the data was never really the point.

The point is the connection. Every sighting on this map is a moment — someone was out on the water, or standing on a headland, or leaning over the railing of a ferry, and they saw something. They pulled out their phone and told the world. That network of people paying attention, caring enough to report what they saw — that's what makes this work. And that's what we're trying to honor.

We care about these animals for reasons that are hard to put neatly into a sentence. Orcas grieve their dead. Humpbacks sing songs that change year over year, spreading across ocean basins like cultural transmission. Sperm whales have the largest brains of any creature that has ever lived on this planet. Great white sharks have been swimming these waters since before the dinosaurs disappeared. They were here long before us, and they deserve better than what we've given them.

Ship strikes kill tens of thousands of whales every year. Most of those deaths are entirely preventable — a captain slows down, changes course by two degrees, and a whale lives. That's it. That's the whole equation. The problem has never been that mariners don't care. It's that they often don't know a whale is there until it's too late. Real-time information changes that calculation completely.

That's where we're going.

Whale Tracker started as a consumer app — a beautiful, fast way for anyone who loves the ocean to stay connected to what's happening out there. That remains at the heart of what we do. But we're building toward something larger: a maritime safety layer that puts the same real-time whale presence data directly in front of the people operating large vessels. Ferries, freighters, tankers, tugboats. The ones whose routes intersect with whale habitat every single day.

We're calling that platform Pelagic. It's early. But the foundation is being laid right now, and apps like this one — built on open data, real community science, and genuine care — are how we get there.

If you downloaded Whale Tracker, you already understand why this matters. You're part of it. Every time the app loads, it's pulling from a network of people around the world who were paying attention. You're one of those people now.

We'll keep building. We'll keep improving the data, expanding the coverage, making the alerts smarter and the map more useful. We'll tell you about named individual whales — animals with histories, personalities, documented family relationships — so that when you see a sighting near you, it isn't just a species and a coordinate. It's someone.

Thank you for being here at the beginning.

— The Whale Tracker Team, Vancouver, BC"""

db.collection("blog_posts").document(SLUG).set({
    "title":       TITLE,
    "subtitle":    "Who we are, why we built this, and where we're going",
    "preview":     PREVIEW,
    "content":     CONTENT,
    "imageURL":    "",
    "sourceURL":   "https://www.whaletrackerapp.com/blog",
    "slug":        SLUG,
    "isPublished": True,
    "createdAt":   firestore.SERVER_TIMESTAMP,
})
print(f"Posted: {TITLE}")
