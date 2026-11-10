# Lead Finder
#URL: "https://leadfinderaa.streamlit.app/"

Automated lead generation and research pipeline. It finds local businesses, checks how well each one handles online customers, and ranks who to contact first.

```
collect.py  ->  enrich.py  ->  score.py  ->  app.py (dashboard)
OpenStreetMap   website signals   rules + ML     ranked leads, map, CSV
```

## What each step does

| Step | File | What it does |
|---|---|---|
| 1. Collect | `collect.py` | Pulls businesses for a city and category from OpenStreetMap (free, no API key). |
| 2. Enrich | `enrich.py` | Visits each website (respects robots.txt, rate limited) and extracts signals: online booking, WhatsApp link, contact form, live chat, HTTPS, mobile friendly, social links, how outdated the site is. Optional LLM summary of services with a free Groq key. |
| 3. Score | `score.py` | Rule score (works with no labels, gives a reason per business). When a `converted` column exists, trains a gradient boosting model and compares it to the rule score and to random using 5-fold cross-validation (precision@k and AUC). |
| 4. Dashboard | `app.py` | Filters, ranked table with reasons, map, CSV export, first-message draft, model results. |

## Run it

```bash
pip install -r requirements.txt

# Real data (needs internet)
python collect.py --city Peshawar --category clinic
python enrich.py
python score.py
streamlit run app.py

# Demo without internet (synthetic data)
python make_sample.py
python score.py
streamlit run app.py
```

Categories: clinic, real_estate, school, academy, gym, restaurant, salon.
Optional: `export GROQ_API_KEY=...` before `enrich.py` for service summaries.

## About the results

The bundled demo data is **synthetic**. Names, phone numbers and the `converted` outcome are generated, and the outcome comes from a hidden rule I wrote plus noise. Metrics on it only show that the pipeline runs end to end. They are not evidence that the model works on real businesses.

To get real results:
1. Run steps 1 to 3 on a real city. With no labels, the rule score is used.
2. Contact the top leads and record who became a client in a `converted` column (1 or 0).
3. Run `score.py` again. Once there are about 60 labeled rows with at least 8 positives, the ML model trains and reports its own held-out precision@k.

## Limits

- OpenStreetMap coverage in Pakistan is partial. Many businesses have no website or phone tag.
- Website checks use simple text and HTML patterns, so they miss booking widgets loaded by JavaScript.
- The rule weights are my judgment, not learned. Treat them as a starting prior.
