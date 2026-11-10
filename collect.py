"""Step 1: collect businesses from OpenStreetMap (free, no API key).

Usage:
    python collect.py --city Peshawar --category clinic
    python collect.py --city Islamabad --category real_estate --out data/raw_islamabad.csv
"""
import argparse
import time

import pandas as pd
import requests

OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "leadfinder-research/1.0 (student project)"}

# category -> list of OSM tag filters
CATEGORIES = {
    "clinic": ['["amenity"~"clinic|doctors|dentist"]', '["healthcare"~"clinic|doctor|dentist"]'],
    "real_estate": ['["office"="estate_agent"]', '["shop"="estate_agent"]'],
    "school": ['["amenity"~"school|college|kindergarten"]'],
    "academy": ['["amenity"="language_school"]', '["office"="educational_institution"]'],
    "gym": ['["leisure"="fitness_centre"]', '["leisure"="sports_centre"]'],
    "restaurant": ['["amenity"~"restaurant|cafe|fast_food"]'],
    "salon": ['["shop"~"hairdresser|beauty"]'],
}


NOMINATIM = "https://nominatim.openstreetmap.org/search"


def get_bbox(city: str, country: str):
    """Look up the city's bounding box (south, west, north, east) with OSM's free geocoder."""
    r = requests.get(NOMINATIM, params={"q": f"{city}, {country}", "format": "json", "limit": 1},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    hits = r.json()
    if not hits:
        raise SystemExit(f"Could not find '{city}, {country}'. Check the spelling.")
    south, north, west, east = (float(x) for x in hits[0]["boundingbox"])
    print(f"Found {hits[0]['display_name']}")
    return south, west, north, east


def build_query(bbox, category: str) -> str:
    box = ",".join(str(x) for x in bbox)
    parts = []
    for f in CATEGORIES[category]:
        for kind in ("node", "way"):
            parts.append(f"{kind}{f}({box});")
    return f'[out:json][timeout:90];({"".join(parts)});out center tags;'


def fetch(bbox, category: str, retries: int = 3) -> list:
    query = build_query(bbox, category)
    for attempt in range(retries):
        try:
            r = requests.post(OVERPASS, data={"data": query}, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                return r.json().get("elements", [])
            print(f"Overpass returned {r.status_code}, retrying...")
        except requests.RequestException as e:
            print(f"Request failed ({e}), retrying...")
        time.sleep(5 * (attempt + 1))
    raise SystemExit("Could not reach Overpass. Try again in a minute.")


def to_row(el: dict, city: str, category: str):
    t = el.get("tags", {})
    name = t.get("name") or t.get("name:en")
    if not name:
        return None
    lat = el.get("lat") or el.get("center", {}).get("lat")
    lon = el.get("lon") or el.get("center", {}).get("lon")
    addr = ", ".join(x for x in [t.get("addr:housenumber"), t.get("addr:street"), t.get("addr:suburb"), t.get("addr:city")] if x)
    return {
        "name": name.strip(),
        "category": category,
        "city": city,
        "phone": t.get("phone") or t.get("contact:phone") or t.get("contact:mobile") or "",
        "email": t.get("email") or t.get("contact:email") or "",
        "website": t.get("website") or t.get("contact:website") or t.get("url") or "",
        "address": addr,
        "lat": lat,
        "lon": lon,
        "osm_id": f"{el['type']}/{el['id']}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--country", default="Pakistan")
    ap.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    ap.add_argument("--out", default="data/raw.csv")
    a = ap.parse_args()

    bbox = get_bbox(a.city, a.country)
    time.sleep(1)  # Nominatim asks for at most 1 request per second
    elements = fetch(bbox, a.category)
    rows = [r for r in (to_row(e, a.city, a.category) for e in elements) if r]
    if not rows:
        raise SystemExit(f"No named {a.category} businesses found in {a.city}. Try another category or a bigger city.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["name", "address"]).reset_index(drop=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {len(df)} businesses to {a.out}")
    print(f"  with website: {(df.website != '').sum()}   with phone: {(df.phone != '').sum()}")


if __name__ == "__main__":
    main()
