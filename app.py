"""Lead Finder dashboard.  Run:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lead Finder", page_icon="🔎", layout="wide")

DATA = Path("data/scored.csv")
REPORT = Path("data/report.json")
HOT = 70

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1250px}
div[data-testid="stMetric"]{background:#fff;border:1px solid #E3E9E6;border-radius:10px;padding:14px 16px}
h1{letter-spacing:-.02em}
</style>""", unsafe_allow_html=True)

if not DATA.exists():
    st.title("Lead Finder")
    st.info("No scored data yet. Run `python make_sample.py` then `python score.py`, "
            "or collect real data with `collect.py`, `enrich.py` and `score.py`.")
    st.stop()


@st.cache_data
def load(path, mtime):
    return pd.read_csv(path, dtype={"phone": str, "email": str, "website": str, "address": str}).fillna({"phone": "", "email": "", "website": "", "reasons": "", "address": ""})


df = load(DATA, DATA.stat().st_mtime)
report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
synthetic = "synthetic" in df.columns and bool(df["synthetic"].fillna(False).astype(bool).any())

st.title("Lead Finder")
st.caption("Finds local businesses, checks how well they handle online customers, and ranks who to contact first.")
if synthetic:
    st.warning("This is **synthetic demo data** (made-up businesses and outcomes). "
               "Run the collector on a real city to get real leads.")

# ---- filters
with st.sidebar:
    st.header("Filters")
    cats = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    min_score = st.slider("Minimum score", 0, 100, 0)
    only_phone = st.checkbox("Has phone or email")
    no_booking = st.checkbox("No online booking")
    no_site = st.checkbox("No website")
    search = st.text_input("Search name or address")

f = df[df["category"].isin(cats) & (df["score"] >= min_score)]
if only_phone:
    f = f[(f["has_phone"] == 1) | (f["has_email"] == 1)]
if no_booking:
    f = f[f["has_booking"] == 0]
if no_site:
    f = f[f["has_website"] == 0]
if search:
    s = search.lower()
    f = f[f["name"].str.lower().str.contains(s) | f["address"].str.lower().str.contains(s)]

# ---- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Businesses found", len(f))
c2.metric("Hot leads (score 70+)", int((f["score"] >= HOT).sum()))
c3.metric("Reachable by phone or email", int(((f["has_phone"] == 1) | (f["has_email"] == 1)).sum()))
c4.metric("Average score", f"{f['score'].mean():.0f}" if len(f) else "-")

tab_leads, tab_map, tab_model = st.tabs(["Ranked leads", "Map", "How the score works"])

with tab_leads:
    show = f[["rank", "name", "category", "score", "reasons", "phone", "website"]].sort_values("rank")
    st.dataframe(
        show, hide_index=True, use_container_width=True, height=430,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "name": "Business", "category": "Type",
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "reasons": st.column_config.TextColumn("Why it is a lead", width="large"),
            "phone": "Phone",
            "website": st.column_config.LinkColumn("Website", display_text="Open"),
        })
    st.download_button("Download these leads (CSV)", show.to_csv(index=False).encode(), "leads.csv", "text/csv")

    st.subheader("Lead detail")
    if len(f):
        pick = st.selectbox("Choose a business", f.sort_values("rank")["name"] + "  (#" + f.sort_values("rank")["rank"].astype(str) + ")")
        r = f[f["rank"] == int(pick.split("(#")[1].rstrip(")"))].iloc[0]
        a, b = st.columns([1, 1])
        with a:
            st.markdown(f"**{r['name']}**  \n{r['category'].replace('_', ' ').title()} · {r['address'] or r['city']}")
            st.markdown(f"Score **{r['score']:.0f}** ({'Hot' if r['score'] >= HOT else 'Warm' if r['score'] >= 40 else 'Cold'})")
            st.markdown("**Gaps found**  \n" + "\n".join(f"- {x}" for x in r["reasons"].split("; ") if x))
            st.markdown(f"Phone: {r['phone'] or 'not listed'}  \nEmail: {r['email'] or 'not listed'}  \nWebsite: {r['website'] or 'none'}")
        with b:
            gaps = r["reasons"].split("; ")[0].lower() if r["reasons"] else "online enquiries"
            st.markdown("**Draft first message**")
            st.code(f"Assalam o Alaikum, I came across {r['name']} and noticed: {gaps}. "
                    "I help local businesses capture and follow up every enquiry automatically, "
                    "so no customer message is missed. Can I show you a 2 minute demo?", language=None)
    else:
        st.write("No businesses match these filters.")

with tab_map:
    m = f.dropna(subset=["lat", "lon"]).rename(columns={"lon": "longitude", "lat": "latitude"})
    if len(m):
        st.map(m[["latitude", "longitude"]], zoom=11)
    else:
        st.write("No location data for these businesses.")

with tab_model:
    st.markdown("**Rule score** adds points for gaps a business can fix (no website, no online booking, no WhatsApp link, "
                "not mobile friendly, outdated site) and lowers the score when there is no phone or email.")
    st.markdown("**ML score** is used when outcome labels exist (`converted` = 1 if the business became a client). "
                "Until then the rule score is used.")
    if "precision_at_k" in report:
        k = report["k"]
        st.markdown(f"**Results** ({report['evaluation']}, {report['n_labeled']} labeled businesses). "
                    f"Of the top {k} ranked, how many became clients?")
        p = report["precision_at_k"]
        chart = pd.DataFrame({"Share that converted": [p["random"], p["rule_score"], p["ml_model"]]},
                             index=["Random pick", "Rule score", "ML model"])
        st.bar_chart(chart, color="#0F6B5C", horizontal=True)
        st.caption(f"ROC AUC: rules {report['auc']['rule_score']}, ML {report['auc']['ml_model']}.")
        if synthetic:
            st.caption("These numbers come from simulated outcomes, so they only prove the pipeline works.")
        imp = pd.Series(report["feature_importance"]).head(8)
        st.markdown("**What the model relies on most**")
        st.bar_chart(imp, color="#0F6B5C", horizontal=True)
    else:
        st.info(report.get("note", "No model report found."))
