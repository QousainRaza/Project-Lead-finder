"""Step 2: visit each business website and extract signals about how well it
handles online customers. Respects robots.txt and rate limits itself.

Usage:
    python enrich.py --inp data/raw.csv --out data/enriched.csv
Optional: set GROQ_API_KEY (free tier) to also get an LLM summary of services.
"""
import argparse
import json
import os
import re
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

UA = "leadfinder-research/1.0 (student project)"
THIS_YEAR = datetime.now().year

BOOKING = re.compile(r"book (an )?(appointment|now|online|a (visit|table|class|session))|schedule (an )?appointment|calendly|appointy|setmore|practo|oladoc|marham", re.I)
CHAT = re.compile(r"tawk\.to|tidio|crisp\.chat|livechat|intercom|drift\.com|zopim|chatra", re.I)
WHATSAPP = re.compile(r"wa\.me/|api\.whatsapp\.com|whatsapp://|web\.whatsapp\.com", re.I)
SOCIAL = re.compile(r"facebook\.com|instagram\.com|linkedin\.com|twitter\.com|x\.com/|youtube\.com|tiktok\.com", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

FEATURE_COLS = ["has_website", "has_https", "mobile_friendly", "has_booking", "has_whatsapp",
                "has_form", "has_chat", "social_count", "has_email", "has_phone",
                "stale_years", "word_count"]


def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not u.startswith("http"):
        u = "http://" + u
    return u


def allowed_by_robots(url: str) -> bool:
    try:
        p = urlparse(url)
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(UA, url)
    except Exception:
        return True


def empty_features(has_phone: int, has_email: int) -> dict:
    return {"has_website": 0, "has_https": 0, "mobile_friendly": 0, "has_booking": 0, "has_whatsapp": 0,
            "has_form": 0, "has_chat": 0, "social_count": 0, "has_email": has_email, "has_phone": has_phone,
            "stale_years": 5, "word_count": 0, "site_title": "", "services": "", "fetch_status": "no website"}


def analyse_site(url: str, has_phone: int, has_email: int) -> dict:
    f = empty_features(has_phone, has_email)
    url = normalize_url(url)
    if not url:
        return f
    if not allowed_by_robots(url):
        f["fetch_status"] = "blocked by robots.txt"
        f["has_website"] = 1
        return f
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=12, allow_redirects=True)
        r.raise_for_status()
    except requests.RequestException:
        f["fetch_status"] = "site unreachable"
        return f  # a dead website counts as no website

    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    years = [int(y) for y in re.findall(r"(?:©|&copy;|copyright)[^0-9]{0,20}((?:19|20)\d{2})", html, re.I)]
    stale = max(0, THIS_YEAR - max(years)) if years else 3
    f.update({
        "has_website": 1,
        "has_https": int(r.url.startswith("https")),
        "mobile_friendly": int(bool(soup.find("meta", attrs={"name": "viewport"}))),
        "has_booking": int(bool(BOOKING.search(html))),
        "has_whatsapp": int(bool(WHATSAPP.search(html))),
        "has_form": int(bool(soup.find("form"))),
        "has_chat": int(bool(CHAT.search(html))),
        "social_count": len(set(m.group(0).lower() for m in SOCIAL.finditer(html))),
        "has_email": int(has_email or bool(EMAIL.search(text))),
        "stale_years": stale,
        "word_count": len(text.split()),
        "site_title": (soup.title.string or "").strip()[:120] if soup.title and soup.title.string else "",
        "fetch_status": "ok",
        "_text": text[:3000],
    })
    return f


def llm_services(text: str) -> str:
    """Optional: short list of services via Groq's free tier."""
    key = os.environ.get("GROQ_API_KEY")
    if not key or not text:
        return ""
    body = {"model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [{"role": "user", "content":
                          'From this business website text, reply with JSON only: {"services": "max 12 words"}.\n\n' + text}],
            "response_format": {"type": "json_object"}, "temperature": 0}
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, timeout=15,
                          headers={"Authorization": f"Bearer {key}"})
        return str(json.loads(r.json()["choices"][0]["message"]["content"]).get("services", ""))[:120]
    except Exception:
        return ""


def enrich_row(row: dict) -> dict:
    has_phone = int(bool(str(row.get("phone") or "").strip()))
    has_email = int(bool(str(row.get("email") or "").strip()))
    feats = analyse_site(str(row.get("website") or ""), has_phone, has_email)
    text = feats.pop("_text", "")
    feats["services"] = llm_services(text)
    time.sleep(0.4)  # be polite
    return {**row, **feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw.csv")
    ap.add_argument("--out", default="data/enriched.csv")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    df = pd.read_csv(a.inp, dtype=str).fillna("")
    rows = df.to_dict("records")
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(enrich_row, rows))
    res = pd.DataFrame(out)
    res.to_csv(a.out, index=False)
    print(f"Enriched {len(res)} businesses -> {a.out}")
    print(res["fetch_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
