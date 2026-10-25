"""
Step 7: human-rated evaluation.  Run from backend/.

  python ml/human_eval.py export --n 150     -> data/human_eval/to_label.csv  (question + answer only)
                                              data/human_eval/items_key.csv (item_id + reference, NOT for raters)
      Raters see neither the reference answer nor the model output, so they judge the answer itself
      rather than its similarity to a reference. Two people fill in rater1 / rater2 independently with
      exactly one of Poor / Average / Good / Excellent.
  python ml/human_eval.py score              -> agreement table

Answers are taken from real app usage (the answers table) when there are enough, topped up with
adversarial DEV cases. Never use the hidden adversarial set here.
Agreement uses quadratic-weighted Cohen's kappa: confusing neighbouring categories costs less than
Poor <-> Excellent. Human-human kappa is the realistic ceiling for the model.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "human_eval"
ORDER = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}


def export(n):
    DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    if Path(config.DB_PATH).exists():
        with sqlite3.connect(config.DB_PATH) as c:
            rows = pd.read_sql("""SELECT q.question, q.reference, a.answer_text AS answer FROM answers a
                                  JOIN questions q ON q.id=a.question_id""", c).to_dict("records")
    adv = json.loads((ROOT / "data" / "adversarial" / "adversarial_dev.json").read_text())
    rows += [{"question": a["question"], "reference": a["reference"], "answer": a["answer"]} for a in adv]
    df = pd.DataFrame(rows).drop_duplicates("answer").sample(frac=1, random_state=7).head(n).reset_index(drop=True)
    df.insert(0, "item_id", range(1, len(df) + 1))
    df[["item_id", "reference"]].to_csv(DIR / "items_key.csv", index=False)
    sheet = df[["item_id", "question", "answer"]].copy()
    sheet["rater1"], sheet["rater2"] = "", ""
    sheet.to_csv(DIR / "to_label.csv", index=False)
    print(f"Wrote {len(df)} items to {DIR / 'to_label.csv'} (references kept in items_key.csv)")


def score():
    from ml import answer_classifier
    df = pd.read_csv(DIR / "to_label.csv", dtype={"rater1": str, "rater2": str}, keep_default_na=False)
    for col in ("rater1", "rater2"):
        df[col] = df[col].str.strip().str.capitalize()
    valid = df.rater1.isin(ORDER) & df.rater2.isin(ORDER)
    bad = df[~valid & ((df.rater1 != "") | (df.rater2 != ""))]
    if len(bad):
        print(f"WARNING: {len(bad)} rows have a label that is not Poor/Average/Good/Excellent "
              f"(item_ids {bad.item_id.tolist()[:10]}...). Fix them or they are excluded.")
    skipped = int((~valid).sum())
    df = df[valid].merge(pd.read_csv(DIR / "items_key.csv", keep_default_na=False), on="item_id", how="left")
    print(f"Scoring {len(df)} items with two valid ratings ({skipped} excluded).")
    df["model"] = [answer_classifier.predict(a, q, r if isinstance(r, str) and r else None)["label"]
                   for q, a, r in zip(df.question, df.answer, df.reference)]
    k = lambda a, b: cohen_kappa_score(df[a].map(ORDER), df[b].map(ORDER), weights="quadratic")
    res = {"items": len(df), "excluded": skipped, "human1_vs_human2": k("rater1", "rater2"),
           "model_vs_human1": k("model", "rater1"), "model_vs_human2": k("model", "rater2")}
    print(json.dumps({k_: round(v, 3) if isinstance(v, float) else v for k_, v in res.items()}, indent=2))
    df.to_csv(DIR / "scored.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "score"])
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    export(a.n) if a.cmd == "export" else score()
