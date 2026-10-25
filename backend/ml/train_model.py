"""
Train the runtime answer classifier and save it (default ml/answer_model.pkl).
Training data: train.csv only, or with --augment train.csv + valid llm_train.csv rows appended in memory.
Run from backend/:
    python ml/train_model.py                                   # v2 baseline config (E / N)
    python ml/train_model.py --ref-set F --noref-set M \
        --ref-params '{"min_samples_leaf": 2}' --noref-params '{"min_samples_leaf": 5}' \
        --out ml/answer_model.pkl                              # candidate (v3) config

Two models are saved in one bundle: one for questions WITH a reference answer, one WITHOUT.
The bundle records its feature groups, so ml/answer_classifier.py needs no code change per version.
Feature sets are defined in ml/experiments.py (SETS). The v2 baseline bundle is kept in ml/baselines/v2/.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so pickles reference ml.features
import json  # noqa: E402

from ml.experiments import SETS  # noqa: E402
from ml.features import FeatureExtractor, answer_tfidf  # noqa: E402
from ml.training_data import load_train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LABELS = ["Poor", "Average", "Good", "Excellent"]
V2_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2}


def make_model(name, params):
    if name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **{**V2_PARAMS, **params})
    raise SystemExit(f"Unknown model {name}")


def build(fx, tf, df, groups, use_tfidf, with_ref):
    d, _ = fx.transform(list(df.question), list(df.answer), list(df.reference) if with_ref else None, groups=groups)
    return sparse.hstack([tf.transform(df.answer), sparse.csr_matrix(d)]).tocsr() if use_tfidf else d


def register(args, bundle, tinfo, val_scores):
    """Append the exact command, configuration and data/artifact hashes to ml/results/model_registry.csv."""
    import datetime
    import hashlib
    import shlex
    sys.path.insert(0, str(HERE))
    from experiments import append_log
    art = Path(args.out)
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "artifact": str(art.resolve().relative_to(ROOT)) if art.resolve().is_relative_to(ROOT) else str(art),
           "artifact_sha256": hashlib.sha256(art.read_bytes()).hexdigest()[:16],
           "command": "python ml/train_model.py " + " ".join(shlex.quote(a) for a in sys.argv[1:]),
           "algorithm": args.model, "training_data": tinfo["training_data"], "n_train": tinfo["n_total"],
           "n_original": tinfo["n_original"], "n_llm": tinfo["n_llm"], "n_jorgpt": tinfo.get("n_jorgpt", 0),
           "train_sha256": tinfo["train_sha256"], "llm_sha256": tinfo["llm_sha256"],
           "jorgpt_sha256": tinfo.get("jorgpt_sha256", ""),
           "ref_spec": json.dumps({k: v for k, v in bundle["model_ref_spec"].items() if k != "params"} |
                                  {"params": {k: bundle["model_ref_spec"]["params"][k] for k in
                                              ("n_estimators", "min_samples_leaf", "max_features", "max_depth", "random_state")}},
                                  default=str),
           "noref_spec": json.dumps({k: v for k, v in bundle["model_noref_spec"].items() if k != "params"} |
                                    {"params": {k: bundle["model_noref_spec"]["params"][k] for k in
                                                ("n_estimators", "min_samples_leaf", "max_features", "max_depth", "random_state")}},
                                    default=str),
           **val_scores}
    append_log(HERE / "results" / "model_registry.csv", [row])
    print("Registered in ml/results/model_registry.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RandomForest")
    ap.add_argument("--ref-set", default="E")
    ap.add_argument("--noref-set", default="N")
    ap.add_argument("--ref-params", default="{}")
    ap.add_argument("--noref-params", default="{}")
    ap.add_argument("--out", default=str(HERE / "answer_model.pkl"))
    ap.add_argument("--augment", action="store_true", help="append valid llm_train.csv rows in memory")
    ap.add_argument("--jorgpt", action="store_true", help="append JorGPT human-graded rows (jorgpt_import.csv) in memory")
    args = ap.parse_args()
    train, tinfo = load_train(args.augment, jorgpt=args.jorgpt)
    val = pd.read_csv(ROOT / "data" / "processed" / "validation.csv")

    fx = FeatureExtractor().fit(train.question, train.reference)   # fit on TRAIN only
    tf = answer_tfidf().fit(train.answer)
    bundle = {"fx": fx, "tf": tf, "algorithm": args.model, "training_data": tinfo}
    val_scores = {}
    for key, fset, params, with_ref in (("model_ref", args.ref_set, args.ref_params, True),
                                        ("model_noref", args.noref_set, args.noref_params, False)):
        groups, use_tfidf = tuple(SETS[fset]["groups"]), SETS[fset]["tfidf"]
        assert ("reference" in groups) == with_ref, f"set {fset} does not match {key}"
        m = make_model(args.model, json.loads(params))
        m.fit(build(fx, tf, train, groups, use_tfidf, with_ref), train.label)
        pred = m.predict(build(fx, tf, val, groups, use_tfidf, with_ref))
        bundle[key + "_spec"] = {"set": fset, "groups": groups, "tfidf": use_tfidf, "params": m.get_params()}
        val_scores[f"{key}_val_accuracy"] = round(accuracy_score(val.label, pred), 3)
        val_scores[f"{key}_val_macro_f1"] = round(f1_score(val.label, pred, average="macro"), 3)
        print(f"{key}: validation accuracy {accuracy_score(val.label, pred):.3f}, "
              f"macro-F1 {f1_score(val.label, pred, average='macro'):.3f}")
        print(classification_report(val.label, pred, labels=LABELS, zero_division=0))
        bundle[key] = m
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print("Saved", args.out)
    register(args, bundle, tinfo, val_scores)

