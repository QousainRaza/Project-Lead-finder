"""Generate a SYNTHETIC demo dataset so the pipeline and dashboard run without internet.

Everything here is made up: names, phones, and the `converted` outcome (simulated from a
hidden rule plus noise). Metrics computed on this file only show that the pipeline works.
Run collect.py + enrich.py on real data for real results.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 420

areas = {"Hayatabad": (33.9786, 71.4283), "University Town": (34.0049, 71.4879), "Saddar": (34.0052, 71.5420),
         "Gulberg": (33.9921, 71.4650), "Regi": (33.9660, 71.4850), "Ring Road": (33.9540, 71.5240)}
cats = {"clinic": ["Care", "Shifa", "Noor", "Family", "Dental", "City", "Medicare", "Life", "Sehat", "Rehman"],
        "real_estate": ["Estate", "Properties", "Builders", "Realtors", "Associates", "Homes", "Developers", "Land"],
        "academy": ["Academy", "Institute", "Learning Hub", "Coaching Center", "Scholars", "Tutors"]}
prefix = ["Al", "Pak", "Khyber", "Frontier", "Royal", "Green", "Star", "Prime", "Elite", "Capital", "Unity", "Zam"]

rows = []
for i in range(N):
    cat = rng.choice(list(cats))
    area = rng.choice(list(areas))
    lat, lon = areas[area]
    name = f"{rng.choice(prefix)} {rng.choice(cats[cat])} {rng.choice(['', '', 'Group', 'Plus', 'Centre'])}".strip()
    has_site = rng.random() < 0.55
    has_phone = rng.random() < 0.85
    has_email = rng.random() < 0.25
    booking = int(has_site and rng.random() < 0.18)
    whatsapp = int(has_site and rng.random() < 0.35)
    form = int(has_site and rng.random() < 0.5)
    chat = int(has_site and rng.random() < 0.1)
    mobile = int(has_site and rng.random() < 0.7)
    https = int(has_site and rng.random() < 0.65)
    social = int(rng.integers(0, 4)) if rng.random() < 0.7 else 0
    stale = int(rng.integers(0, 7)) if has_site else 5
    slug = name.lower().replace(" ", "")
    rows.append({
        "name": name, "category": cat, "city": "Peshawar",
        "phone": f"0{rng.integers(300, 349)}{rng.integers(1000000, 9999999)}" if has_phone else "",
        "email": f"info@{slug}.pk" if has_email else "",
        "website": f"https://{slug}{i}.pk" if has_site else "",
        "address": f"{rng.integers(1, 120)} {area}, Peshawar",
        "lat": lat + rng.normal(0, 0.006), "lon": lon + rng.normal(0, 0.006), "osm_id": f"synthetic/{i}",
        "has_website": int(has_site), "has_https": https, "mobile_friendly": mobile, "has_booking": booking,
        "has_whatsapp": whatsapp, "has_form": form, "has_chat": chat, "social_count": social,
        "has_email": int(has_email), "has_phone": int(has_phone), "stale_years": stale,
        "word_count": int(rng.integers(80, 2500)) if has_site else 0,
        "site_title": "", "services": "", "fetch_status": "ok" if has_site else "no website",
    })

df = pd.DataFrame(rows)

# Simulated outcome: a hidden pattern (needs help AND reachable) plus noise and an interaction the rules miss.
logit = (-4.7 + 1.3 * (1 - df.has_website) + 0.9 * (1 - df.has_booking) + 0.5 * (1 - df.has_whatsapp)
         + 0.35 * (df.stale_years >= 3) + 0.4 * (1 - df.mobile_friendly) + 1.1 * df.has_phone
         + 0.9 * ((df.category == "clinic") & (df.has_booking == 0)) - 0.5 * df.has_chat
         + rng.normal(0, 0.8, len(df)))
df["converted"] = (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)
df["synthetic"] = True
df.to_csv("data/enriched.csv", index=False)
print(f"Wrote data/enriched.csv  ({len(df)} synthetic businesses, conversion rate {df.converted.mean():.1%})")
