"""
Evaluate the saved runtime model (ml/answer_model.pkl).
Run from backend/:   python ml/evaluate_model.py [--hidden]

Reports: synthetic validation (unseen questions), adversarial DEV set, and with --hidden the
hidden adversarial set (run once, at the end). Writes ml/metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import answer_classifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["Poor", "Average", "Good", "Excellent"]


def run(df, use_ref=True):
    return [answer_classifier.predict(a, q, r if use_ref else None)["label"]
            for q, a, r in zip(df.question, df.answer, df.reference)]


def show_cm(y, p):
    cm = confusion_matrix(y, p, labels=LABELS)
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string(), "\n")


def adversarial(name, use_ref):
    adv = pd.DataFrame(json.loads((ROOT / "data" / "adversarial" / f"adversarial_{name}.json").read_text()))
    adv["pred"] = run(adv, use_ref)
    adv["ok"] = [p in a for p, a in zip(adv.pred, adv.acceptable)]
    return {"macro_f1": round(f1_score(adv.expected, adv.pred, labels=LABELS, average="macro", zero_division=0), 3),
            "acceptable_rate": round(float(adv.ok.mean()), 3),
            "by_category": adv.groupby("category").ok.mean().round(2).to_dict()}, adv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--model-path", default="", help="score another bundle, e.g. ml/candidates/v3/answer_model.pkl")
    ap.add_argument("--metrics-out", default=str(Path(__file__).resolve().parent / "metrics.json"))
    args = ap.parse_args()
    if args.model_path:
        answer_classifier.BUNDLE_PATH = Path(args.model_path).resolve()
    val = pd.read_csv(ROOT / "data" / "processed" / "validation.csv")
    out = {}
    for use_ref in (True, False):
        tag = "with_reference" if use_ref else "no_reference"
        p = run(val, use_ref)
        out[tag] = {"validation_accuracy": round(accuracy_score(val.label, p), 3),
                    "validation_macro_f1": round(f1_score(val.label, p, average="macro"), 3)}
        print(f"== {tag} ==  validation acc {out[tag]['validation_accuracy']}, macro-F1 {out[tag]['validation_macro_f1']}")
        show_cm(val.label, p)
        for name in (["dev", "hidden"] if args.hidden else ["dev"]):
            res, adv = adversarial(name, use_ref)
            out[tag][f"adversarial_{name}"] = res
            print(f"adversarial {name}: {res}\n")
    out["model_path"] = str(answer_classifier.BUNDLE_PATH)
    Path(args.metrics_out).write_text(json.dumps(out, indent=2))
