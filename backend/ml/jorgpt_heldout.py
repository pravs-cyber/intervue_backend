"""
Held-out-question evaluation on real student answers (JorGPT). Run from backend/:  python ml/jorgpt_heldout.py

Question: can the model grade REAL human answers to questions it has never seen, and does adding JorGPT's other
questions to training help?

Split (pre-registered, question-level): every JorGPT question_id is ranked by md5(SEED + question_id); the 10 lowest
are held out for evaluation, the other 40 may be used for training. The split is written once to
data/sources/jorgpt/heldout_split.json (tracked) and re-used on every later run - it is never re-derived or tuned.
All answers to a question are on the same side.

Labels: teacher_grade -> M2 (0-2 Poor | 3-5 Average | 6-7 Good | 8-10 Excellent), from jorgpt_import.csv.
Conditions (evaluated on the SAME held-out answers, each with the question's ideal answer as reference):
  v3 candidate (frozen bundle, unchanged)                 ml/candidates/v3/answer_model.pkl
  base F / base G         RandomForest(150, seed 42) on train.csv only               (as in experiments.py)
  +JorGPT F / +JorGPT G   same, on train.csv + JorGPT rows of the 40 training questions
  reference rows: length-only RandomForest (+JorGPT data), and always-majority-class
Metrics: quadratic-weighted Cohen's kappa (primary), accuracy, macro-F1, confusion matrix, predicted distribution.
Appends to ml/results/jorgpt_heldout_log.csv. validation.csv / adversarial sets / v3 bundle are only read.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import Builder, append_log  # noqa: E402
from training_data import JORGPT_IMPORT, load_train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "sources" / "jorgpt" / "heldout_split.json"
LOG = Path(__file__).resolve().parent / "results" / "jorgpt_heldout_log.csv"
V3 = Path(__file__).resolve().parent / "candidates" / "v3" / "answer_model.pkl"
SEED = "intervue-jorgpt-heldout-v1"
N_EVAL = 10
LABELS = ["Poor", "Average", "Good", "Excellent"]
ORD = {c: i for i, c in enumerate(LABELS)}


def get_split(question_ids):
    if SPLIT.exists():                                   # pre-registered: never re-derive
        s = json.loads(SPLIT.read_text())
        assert set(s["eval"]) | set(s["train"]) == set(question_ids), "split file does not match the import"
        return s
    ranked = sorted(question_ids, key=lambda q: hashlib.md5((SEED + q).encode()).hexdigest())
    s = {"seed": SEED, "rule": f"sort by md5(seed + question_id); first {N_EVAL} = eval",
         "created": datetime.date.today().isoformat(), "eval": sorted(ranked[:N_EVAL]), "train": sorted(ranked[N_EVAL:])}
    SPLIT.write_text(json.dumps(s, indent=2))
    return s


def metrics(y, p):
    y, p = pd.Series(list(y)), pd.Series(list(p))
    return {"kappa_qw": round(cohen_kappa_score(y.map(ORD), p.map(ORD), weights="quadratic"), 3),
            "accuracy": round(accuracy_score(y, p), 3),
            "macro_f1": round(f1_score(y, p, labels=LABELS, average="macro", zero_division=0), 3),
            "confusion": json.dumps(confusion_matrix(y, p, labels=LABELS).tolist()),
            "pred_dist": json.dumps(p.value_counts().reindex(LABELS, fill_value=0).to_dict())}


def main():
    jor = pd.read_csv(JORGPT_IMPORT)
    split = get_split(sorted(jor.question_id.unique()))
    ev = jor[jor.question_id.isin(split["eval"])].reset_index(drop=True)
    print(f"held-out questions: {len(split['eval'])} ({len(ev)} answers); training questions: {len(split['train'])}")
    print("held-out labels:", ev.label.value_counts().reindex(LABELS).to_dict(),
          f"| course-specific questions held out: {int(ev.groupby('question_id').course_specific.first().sum())}")

    base, _ = load_train(jorgpt=False, verbose=False)
    aug, _ = load_train(jorgpt=True, verbose=False)
    aug = aug[~((aug.source == "jorgpt") & aug.question_id.isin(split["eval"]))].reset_index(drop=True)
    assert not set(aug.question) & set(ev.question), "held-out question leaked into training"
    assert (aug.source == "jorgpt").sum() == (jor.question_id.isin(split["train"])).sum()

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rows, preds = [], {}

    def log(condition, fset, train_rows, pred, note=""):
        r = {"run_id": run_id, "condition": condition, "features": fset, "n_train": train_rows,
             "n_eval": len(ev), "eval_questions": len(split["eval"]), "split_seed": SEED, **metrics(ev.label, pred),
             "note": note}
        rows.append(r)
        preds[f"{condition} {fset}".strip()] = list(pred)
        print(f"{condition:22s} {fset:2s} kappa {r['kappa_qw']:.3f}  acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  "
              f"pred {r['pred_dist']}")

    # frozen v3 bundle, exactly as it would grade these answers in the app
    from ml import answer_classifier as ac
    ac.BUNDLE_PATH = V3
    log("v3 candidate (frozen)", "F", 8832, [ac.predict(a, q, r)["label"] for q, a, r in
                                             zip(ev.question, ev.answer, ev.reference)], "300 trees, leaf 1")

    for cond, tr in (("base", base), ("+JorGPT", aug)):
        b = Builder(tr)
        for fset in ("F", "G"):
            m = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(b.X(tr, fset), tr.label)
            log(cond, fset, len(tr), m.predict(b.X(ev, fset)))
        if cond == "+JorGPT":
            lw = lambda d: np.log1p(d.answer.str.split().str.len().values).reshape(-1, 1)
            m = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=20).fit(lw(tr), tr.label)
            log("length only", "-", len(tr), m.predict(lw(ev)), "reference row: answer length is the only feature")
    log("majority class", "-", len(aug), [aug.label.mode()[0]] * len(ev), "reference row")
    append_log(LOG, rows)
    out = ev[["question_id", "entry_id", "teacher_grade", "label"]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_csv(LOG.parent / f"jorgpt_heldout_predictions_{run_id}.csv", index=False)
    robustness(out, ["v3 candidate (frozen) F", "base F", "base G"], ["+JorGPT F", "+JorGPT G"])
    print(f"\nlogged run {run_id} -> {LOG}")


def robustness(out, bases, news, n_boot=2000, seed=0):
    """Per-question kappa and a question-level bootstrap CI for kappa differences (resample whole questions)."""
    k = lambda d, c: cohen_kappa_score(d.label.map(ORD), d[c].map(ORD), weights="quadratic")
    print("\nper held-out question, quadratic kappa:")
    per = out.groupby("question_id").apply(lambda d: pd.Series({c: round(k(d, c), 2) for c in bases + news}),
                                           include_groups=False)
    print(per.to_string())
    rng, qs = np.random.default_rng(seed), out.question_id.unique()
    groups = {q: out[out.question_id == q] for q in qs}
    for b, n in [("base F", "+JorGPT F"), ("base G", "+JorGPT G"), ("v3 candidate (frozen) F", "+JorGPT F")]:
        diffs = []
        for _ in range(n_boot):
            d = pd.concat([groups[q] for q in rng.choice(qs, len(qs), replace=True)])
            diffs.append(k(d, n) - k(d, b))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        better = int((per[n] > per[b]).sum())
        print(f"{n} vs {b}: kappa diff {k(out, n) - k(out, b):+.3f}, 95% CI over questions [{lo:+.3f}, {hi:+.3f}], "
              f"better on {better}/{len(qs)} questions")


if __name__ == "__main__":
    main()
