"""
Tune the Random Forest with grouped cross-validation. Run from backend/:
    python ml/tune_rf.py --set G

* Training data ONLY (train.csv). Folds are GroupKFold(5) grouped by question, so all 8 variants of a question
  are in the same fold - no question appears in both the fitting and the scoring part of a fold.
* Feature extractors are re-fitted inside every fold on that fold's training questions only.
* Selection criterion: mean CV macro-F1. Nothing else. Validation and adversarial-dev scores of the selected
  configuration are reported afterwards for information; they do not influence the choice.
* The hidden adversarial set is never loaded.
Every configuration is appended to ml/results/tuning_log.csv (fold scores, mean, std, params, data hash).
"""
import argparse
import datetime
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import LOG_DEFAULTS, SETS, Builder, adv_scores, append_log, cm_json, load_adv  # noqa: E402
from training_data import load_train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
# Sparse sets (with ~5,000 TF-IDF columns) get a smaller grid: each fit is ~1 min on one CPU.
GRID_SPARSE = {"n_estimators": [300], "max_depth": [None], "min_samples_leaf": [1, 2, 5],
               "max_features": ["sqrt", 0.03]}
GRID_DENSE = {"n_estimators": [300], "max_depth": [None, 25], "min_samples_leaf": [1, 2, 5],
              "max_features": ["sqrt", 0.5]}


def configs(fset):
    grid = GRID_SPARSE if SETS[fset]["tfidf"] else GRID_DENSE
    for vals in itertools.product(*grid.values()):
        yield dict(zip(grid, vals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="G")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--augment", action="store_true", help="append valid llm_train.csv rows in memory")
    ap.add_argument("--jorgpt", action="store_true", help="append JorGPT human-graded rows (jorgpt_import.csv) in memory")
    ap.add_argument("--only", default="", help="JSON list of configs to run instead of the grid (compute limits)")
    args = ap.parse_args()
    train, tinfo = load_train(args.augment, jorgpt=args.jorgpt)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gkf = GroupKFold(n_splits=args.folds)
    folds = []
    for tr_i, te_i in gkf.split(train, train.label, groups=train.question):
        tr, te = train.iloc[tr_i], train.iloc[te_i]
        assert not set(tr.question) & set(te.question)
        b = Builder(tr)                                   # fitted on this fold's training part only
        folds.append((b.X(tr, args.set), tr.label.values, b.X(te, args.set), te.label.values))
    print(f"{args.folds} grouped folds ready for set {args.set}")

    log, best = [], None
    todo = json.loads(args.only) if args.only else list(configs(args.set))
    for cfg in todo:
        t = time.time()
        scores = []
        for Xtr, ytr, Xte, yte in folds:
            m = RandomForestClassifier(random_state=42, n_jobs=-1, **cfg).fit(Xtr, ytr)
            scores.append(f1_score(yte, m.predict(Xte), average="macro"))
        row = {"run_id": run_id, "training_data": tinfo["training_data"], "train_sha256": tinfo["train_sha256"],
               "llm_sha256": tinfo["llm_sha256"], "n_llm": tinfo["n_llm"],
               "features": args.set, "params": json.dumps(cfg), "cv": f"GroupKFold({args.folds}) by question",
               "cv_macro_f1_mean": round(float(np.mean(scores)), 4), "cv_macro_f1_std": round(float(np.std(scores)), 4),
               "fold_scores": json.dumps([round(s, 4) for s in scores]), "seconds": round(time.time() - t, 1)}
        print(row["params"], row["cv_macro_f1_mean"], "+/-", row["cv_macro_f1_std"])
        append_log(OUT / "tuning_log.csv", [row], LOG_DEFAULTS)
        if best is None or row["cv_macro_f1_mean"] > best[1]:
            best = (cfg, row["cv_macro_f1_mean"])

    cfg = best[0]
    print("\nSELECTED by CV macro-F1:", cfg, best[1])
    # information only: refit on all training data, score validation + adversarial dev
    val = pd.read_csv(ROOT / "data" / "processed" / "validation.csv")
    dev = load_adv("dev")
    b = Builder(train)
    m = RandomForestClassifier(random_state=42, n_jobs=-1, **cfg).fit(b.X(train, args.set), train.label)
    pv, pdv = m.predict(b.X(val, args.set)), m.predict(b.X(dev, args.set))
    af1, aacc, cats = adv_scores(pdv, dev)
    summary = {"run_id": run_id, "features": args.set, "selected_params": cfg, "cv_macro_f1": best[1],
               "val_accuracy": round(accuracy_score(val.label, pv), 3),
               "val_macro_f1": round(f1_score(val.label, pv, average="macro"), 3),
               "val_confusion": cm_json(val.label, pv), "adv_dev_acceptable": round(aacc, 3),
               "adv_dev_macro_f1": round(af1, 3), "adv_dev_confusion": cm_json(dev.expected, pdv),
               "adv_dev_by_category": cats}
    (OUT / f"tuning_selected_{args.set}_{run_id}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
