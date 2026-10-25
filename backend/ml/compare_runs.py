"""
Compare two training-data conditions from the experiment log (read-only).
Run from backend/:  python ml/compare_runs.py --base RUN_ID --new RUN_ID [--sets FG]
Prints validation / adversarial-dev metrics side by side per model and feature set, and the adversarial-dev
acceptable rate per category (from ml/results/adversarial_dev_by_category_<run_id>.json).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent / "results"
FOCUS = ["correct_paraphrase", "restatement", "correct_concise", "correct_detailed", "partially_correct",
         "confident_wrong", "off_topic", "other_question", "keyword_stuffing", "long_nonsense", "dont_know"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--sets", default="FG")
    a = ap.parse_args()
    log = pd.read_csv(RES / "experiment_log.csv", dtype={"run_id": str})
    cols = ["val_accuracy", "val_macro_f1", "adv_dev_acceptable", "adv_dev_macro_f1"]
    b = log[(log.run_id == a.base) & log.features.isin(list(a.sets))].set_index(["features", "model"])
    n = log[(log.run_id == a.new) & log.features.isin(list(a.sets))].set_index(["features", "model"])
    print(f"BASE {a.base}: {b.training_data.iloc[0]} ({int(b.n_train.iloc[0])} rows)")
    print(f"NEW  {a.new}: {n.training_data.iloc[0]} ({int(n.n_train.iloc[0])} rows)\n")
    t = pd.concat({"base": b[cols], "new": n[cols]}, axis=1)
    for c in cols:
        t[("delta", c)] = (t[("new", c)] - t[("base", c)]).round(3)
    print(t.sort_index().to_string())
    cb = json.loads((RES / f"adversarial_dev_by_category_{a.base}.json").read_text())
    cn = json.loads((RES / f"adversarial_dev_by_category_{a.new}.json").read_text())
    rows = {}
    for key in sorted(cb):
        if key.split("/")[0] in a.sets and key in cn:
            for cat in FOCUS:
                rows[(key, cat)] = (cb[key].get(cat), cn[key].get(cat))
    per = pd.DataFrame(rows, index=["base", "new"]).T
    per["delta"] = per.new - per.base
    print("\nAdversarial dev acceptable rate by category (4 cases each):")
    print(per.unstack(0).swaplevel(axis=1).sort_index(axis=1).round(2).to_string())
