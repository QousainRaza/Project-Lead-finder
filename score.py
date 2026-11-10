"""Step 3: rank businesses by how likely they are to need your service.

Two scorers:
  1. Rule score   - works with no labels. Transparent reasons per business.
  2. ML score     - trains when a `converted` column (0/1) exists with enough labels.
                    Compared against the rule score and random on a held-out set.

Usage:
    python score.py --inp data/enriched.csv --out data/scored.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from enrich import FEATURE_COLS

HOT = 70


def rule_score(r) -> tuple:
    need, why = 0, []
    if not r["has_website"]:
        need += 35; why.append("No working website")
    else:
        if not r["has_booking"]:
            need += 20; why.append("No online booking")
        if not r["has_whatsapp"]:
            need += 8; why.append("No WhatsApp link")
        if not r["has_form"] and not r["has_chat"]:
            need += 10; why.append("No contact form or chat")
        if not r["mobile_friendly"]:
            need += 12; why.append("Not mobile friendly")
        if not r["has_https"]:
            need += 8; why.append("No HTTPS")
        if r["stale_years"] >= 3:
            need += 10; why.append("Website looks outdated")
    if r["social_count"] == 0:
        need += 5; why.append("No social links")
    reachable = bool(r["has_phone"] or r["has_email"])
    if not reachable:
        why.append("Hard to reach (no phone or email)")
    return min(need, 100) * (1.0 if reachable else 0.3), "; ".join(why) or "Already well set up"


def precision_at_k(y_true, scores, k):
    idx = np.argsort(-np.asarray(scores))[:k]
    return float(np.mean(np.asarray(y_true)[idx]))


MODEL_FEATURES = [c for c in FEATURE_COLS if c != "word_count"]  # word_count just mirrors has_website


def build_X(df: pd.DataFrame, cats) -> pd.DataFrame:
    X = df[MODEL_FEATURES].astype(float).copy()
    for c in cats:
        X[f"cat_{c}"] = (df["category"] == c).astype(float)
    return X


def train_and_evaluate(df: pd.DataFrame):
    labeled = df[df["converted"].isin([0, 1])]
    if len(labeled) < 60 or labeled["converted"].nunique() < 2 or labeled["converted"].sum() < 8:
        return None, None, None
    cats = sorted(df["category"].dropna().unique())
    X, y = build_X(labeled, cats), labeled["converted"].astype(int).values
    model = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.08, random_state=7)
    # out-of-fold predictions: every business is scored by a model that never saw it
    oof = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba")[:, 1]
    rule = labeled["rule_score"].values
    k = max(5, int(0.2 * len(y)))
    report = {
        "evaluation": "5-fold cross-validation, out-of-fold predictions",
        "n_labeled": int(len(y)), "k": k, "base_rate": round(float(y.mean()), 3),
        "precision_at_k": {"random": round(float(y.mean()), 3),
                           "rule_score": round(precision_at_k(y, rule, k), 3),
                           "ml_model": round(precision_at_k(y, oof, k), 3)},
        "auc": {"rule_score": round(float(roc_auc_score(y, rule)), 3),
                "ml_model": round(float(roc_auc_score(y, oof)), 3)},
    }
    model.fit(X, y)
    report["feature_importance"] = {c: round(float(v), 3) for c, v in
                                    sorted(zip(X.columns, model.feature_importances_), key=lambda t: -t[1])}
    return model, cats, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/enriched.csv")
    ap.add_argument("--out", default="data/scored.csv")
    ap.add_argument("--report", default="data/report.json")
    a = ap.parse_args()

    df = pd.read_csv(a.inp, dtype={"phone": str, "email": str, "website": str, "address": str})
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "converted" not in df:
        df["converted"] = np.nan

    scored = df.apply(rule_score, axis=1, result_type="expand")
    df["rule_score"], df["reasons"] = scored[0], scored[1]

    model, cats, report = train_and_evaluate(df)
    if model is not None:
        df["score"] = (model.predict_proba(build_X(df, cats))[:, 1] * 100).round(1)
        df["scorer"] = "ml_model"
    else:
        df["score"] = df["rule_score"].round(1)
        df["scorer"] = "rules"
        report = {"note": "No outcome labels yet, so rule scoring is used. Add a `converted` column (1 = became a client) to train the ML model."}

    df["rank"] = df["score"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("rank")
    df.to_csv(a.out, index=False)
    json.dump(report, open(a.report, "w"), indent=2)
    print(f"Scored {len(df)} businesses with {df['scorer'].iloc[0]} -> {a.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
