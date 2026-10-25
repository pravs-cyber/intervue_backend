"""
Model comparison + ablation study.  Run from backend/:  python ml/experiments.py [--sets ABCDENFGM] [--models RandomForest,...]

Training data: train.csv only (default, "existing dataset") or, with --augment, train.csv + valid rows of
llm_train.csv appended in memory ("existing dataset + LLM augmentation"); see training_data.py. train.csv is
never modified. Evaluation always uses the same validation.csv (unseen questions) and the
adversarial DEV set. This script cannot read the hidden adversarial set - that is only done once, by
`evaluate_model.py --hidden`, after the final model is frozen.

Feature sets (ablation):
  A  answer TF-IDF only                        (the original v1 representation)
  B  hand-crafted: surface + relevance + reference
  C  A + B
  D  A + semantic (LSA similarity to question and reference)
  E  A + B + semantic                          (all features)
  N  surface + relevance + semantic, NO reference (what runtime uses when a question has no reference)
  -- v3, experimental pretrained-embedding features (GloVe, see build_embeddings.py) --
  F  E + embedding          G  B + semantic + embedding (dense only)          M  N + embedding (no reference)

Models (syllabus): Decision Tree, Naive Bayes, SVM, KNN, Neural Network (MLP), plus Random Forest as the
ensemble baseline. Scale-sensitive models get a StandardScaler inside their pipeline; trees do not.
Naive Bayes: MultinomialNB on A (non-negative counts), GaussianNB on B/N (dense, real-valued);
not run on the mixed sparse+dense sets C/D/E.
Every run is APPENDED to ml/results/experiment_log.csv with: run id, timestamp, train.csv hash, feature set and
groups, model class and parameters, validation accuracy / macro-F1 / confusion matrix, adversarial-dev
acceptable rate / macro-F1 / confusion matrix, fit time. The v2 baseline is frozen in ml/baselines/v2/.
"""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import FeatureExtractor, answer_tfidf
from training_data import load_train

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LABELS = ["Poor", "Average", "Good", "Excellent"]
SEED = 42

SETS = {
    "A": dict(tfidf=True, groups=()),
    "B": dict(tfidf=False, groups=("surface", "relevance", "reference")),
    "C": dict(tfidf=True, groups=("surface", "relevance", "reference")),
    "D": dict(tfidf=True, groups=("semantic_only",)),
    "E": dict(tfidf=True, groups=("surface", "relevance", "reference", "semantic")),
    "N": dict(tfidf=False, groups=("surface", "relevance", "semantic")),
    "F": dict(tfidf=True, groups=("surface", "relevance", "reference", "semantic", "embedding")),
    "G": dict(tfidf=False, groups=("surface", "relevance", "reference", "semantic", "embedding")),
    "M": dict(tfidf=False, groups=("surface", "relevance", "semantic", "embedding")),
}


def models(fset):
    sp = SETS[fset]["tfidf"]
    scale = lambda m: make_pipeline(StandardScaler(with_mean=not sp), m)
    out = {
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1),
        "SVM": scale(SVC(kernel="rbf", C=1.0, random_state=SEED)),
        "KNN": scale(KNeighborsClassifier(n_neighbors=15)),
        "NeuralNet(MLP)": scale(MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                              random_state=SEED)),
    }
    if fset == "A":
        out["NaiveBayes"] = MultinomialNB()
    elif not sp:
        out["NaiveBayes"] = GaussianNB()
    return out


class Builder:
    """Fits every representation on TRAIN only and builds X for any frame."""

    def __init__(self, train):
        self.fx = FeatureExtractor().fit(train.question, train.reference)
        self.tf = answer_tfidf().fit(train.answer)

    def X(self, df, fset):
        spec = SETS[fset]
        parts = []
        if spec["tfidf"]:
            parts.append(self.tf.transform(df.answer))
        groups = spec["groups"]
        if groups == ("semantic_only",):
            d, names = self.fx.transform(list(df.question), list(df.answer), list(df.reference),
                                         groups=("reference", "semantic"))
            d = d[:, [names.index("q_lsa"), names.index("ref_lsa")]]
            parts.append(sparse.csr_matrix(d))
        elif groups:
            refs = list(df.reference) if "reference" in groups else None
            d, _ = self.fx.transform(list(df.question), list(df.answer), refs, groups=groups)
            parts.append(sparse.csr_matrix(d) if spec["tfidf"] else d)
        return sparse.hstack(parts).tocsr() if spec["tfidf"] else parts[0]


def cm_json(y, p):
    return json.dumps(confusion_matrix(y, p, labels=LABELS).tolist())


def model_config(m):
    est = m.steps[-1][1] if hasattr(m, "steps") else m
    keep = ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "C", "kernel", "n_neighbors",
            "hidden_layer_sizes", "max_iter", "random_state", "class_weight")
    params = {k: v for k, v in est.get_params().items() if k in keep}
    return type(est).__name__, json.dumps(params, default=str), hasattr(m, "steps")


def data_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def append_log(path, rows, defaults=None):
    """Append rows, merging columns by NAME (older logs may lack newer columns; they get `defaults`)."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        for col, val in (defaults or {}).items():
            if col not in old.columns:
                old[col] = val
        new = pd.concat([old, new], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


# runs logged before augmentation support used train.csv only
LOG_DEFAULTS = {"training_data": "existing dataset", "llm_sha256": "", "n_llm": 0, "jorgpt_sha256": "", "n_jorgpt": 0}


def log_rows(rows):
    path = OUT / "experiment_log.csv"
    defaults = dict(LOG_DEFAULTS, n_original=pd.read_csv(path).n_train if path.exists() else 0)
    append_log(path, rows, defaults)


def load_adv(name):
    return pd.DataFrame(json.loads((ROOT / "data" / "adversarial" / f"adversarial_{name}.json").read_text()))


def adv_scores(pred, adv):
    ok = [p in acc for p, acc in zip(pred, adv.acceptable)]
    per_cat = pd.Series(ok, index=adv.category).groupby(level=0).mean().round(2).to_dict()
    return f1_score(adv.expected, pred, labels=LABELS, average="macro", zero_division=0), float(np.mean(ok)), per_cat


def v1_baseline(adv):
    """Score the ORIGINAL saved v1 model (answer TF-IDF -> RF, old data) for the 'before' row."""
    vec, clf = HERE / "v1_baseline" / "tfidf_vectorizer.pkl", HERE / "v1_baseline" / "answer_quality_model.pkl"
    if not (vec.exists() and clf.exists()):
        return None
    pred = joblib.load(clf).predict(joblib.load(vec).transform(adv.answer))
    return adv_scores(pred, adv), pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="ABCDENFGM")
    ap.add_argument("--augment", action="store_true", help="append valid llm_train.csv rows in memory")
    ap.add_argument("--jorgpt", action="store_true", help="append JorGPT human-graded rows (jorgpt_import.csv) in memory")
    ap.add_argument("--models", default="", help="comma-separated subset, e.g. RandomForest,SVM")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    train, tinfo = load_train(args.augment, jorgpt=args.jorgpt)
    val = pd.read_csv(ROOT / "data" / "processed" / "validation.csv")
    dev = load_adv("dev")
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_hash = tinfo["train_sha256"]

    b = v1_baseline(dev)
    if b and not (args.augment or args.jorgpt):
        (f1, acc, cats), pred = b
        print(f"v1 baseline on adversarial DEV: macro-F1 {f1:.3f}, acceptable {acc:.2f}\n  per category: {cats}")
        pd.DataFrame({"id": dev.id, "category": dev.category, "expected": dev.expected, "v1_pred": pred}) \
            .to_csv(OUT / "v1_adversarial_dev_predictions.csv", index=False)

    bld = Builder(train)
    rows, cats_log = [], {}
    for fset in args.sets:
        Xtr, Xva, Xdv = bld.X(train, fset), bld.X(val, fset), bld.X(dev, fset)
        for name, m in models(fset).items():
            if args.models and name not in args.models.split(","):
                continue
            t = time.time()
            m.fit(Xtr, train.label)
            pv, pd_ = m.predict(Xva), m.predict(Xdv)
            af1, aacc, cats = adv_scores(pd_, dev)
            cls, params, scaled = model_config(m)
            row = {"run_id": run_id, "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                   "training_data": tinfo["training_data"], "train_sha256": train_hash,
                   "llm_sha256": tinfo["llm_sha256"], "n_original": tinfo["n_original"], "n_llm": tinfo["n_llm"],
                   "jorgpt_sha256": tinfo["jorgpt_sha256"], "n_jorgpt": tinfo["n_jorgpt"],
                   "n_train": len(train), "model": name, "estimator": cls,
                   "params": params, "scaled": scaled, "features": fset,
                   "feature_groups": "+".join((["answer_tfidf"] if SETS[fset]["tfidf"] else []) + list(SETS[fset]["groups"])),
                   "val_accuracy": round(accuracy_score(val.label, pv), 3),
                   "val_macro_f1": round(f1_score(val.label, pv, average="macro"), 3),
                   "val_confusion": cm_json(val.label, pv),
                   "adv_dev_macro_f1": round(af1, 3), "adv_dev_acceptable": round(aacc, 3),
                   "adv_dev_confusion": cm_json(dev.expected, pd_),
                   "fit_seconds": round(time.time() - t, 1)}
            rows.append(row)
            log_rows([row])
            cats_log[f"{fset}/{name}"] = cats
            print(row)
    res = pd.DataFrame(rows)
    (OUT / f"adversarial_dev_by_category_{run_id}.json").write_text(json.dumps(cats_log, indent=2))
    cols = ["model", "features", "val_accuracy", "val_macro_f1", "adv_dev_acceptable", "adv_dev_macro_f1"]
    print("\n" + res[cols].sort_values("adv_dev_acceptable", ascending=False).to_string(index=False))
