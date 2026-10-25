"""
Runtime wrapper around the saved answer classifier (built by ml/train_model.py).

The bundle holds two models trained on the same data:
  * model_ref   - feature set E (answer TF-IDF + surface + relevance + reference + semantic)
  * model_noref - feature set N (surface + relevance + semantic, no reference)
The reference model is used when the question has a reference answer, otherwise the no-reference model.

Output is a PREDICTED ANSWER CATEGORY, not a verified quality judgement (see README §6).
"""
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from services.constants import CATEGORIES, CATEGORY_SCORE

_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = _DIR / "answer_model.pkl"
_bundle = None


def is_ready() -> bool:
    return BUNDLE_PATH.exists()


def _load():
    global _bundle
    if _bundle is None:
        if not is_ready():
            raise RuntimeError("Trained model not found. Run: python ml/prepare_dataset.py && python ml/train_model.py")
        _bundle = joblib.load(BUNDLE_PATH)
    return _bundle


# v2 bundles did not record their feature groups; these are their fixed specs.
_V2_SPECS = {"model_ref": {"groups": ("surface", "relevance", "reference", "semantic"), "tfidf": True},
             "model_noref": {"groups": ("surface", "relevance", "semantic"), "tfidf": False}}


def _features(b, question, answer, reference):
    key = "model_ref" if reference else "model_noref"
    spec = b.get(key + "_spec", _V2_SPECS[key])
    dense, _ = b["fx"].transform([question], [answer], [reference] if reference else None, groups=tuple(spec["groups"]))
    X = sparse.hstack([b["tf"].transform([answer]), sparse.csr_matrix(dense)]).tocsr() if spec["tfidf"] else dense
    return b[key], X, "with_reference" if reference else "no_reference"


def predict(answer: str, question: str = "", reference: str | None = None) -> dict:
    b = _load()
    model, X, used = _features(b, question, answer, reference)
    probs = dict(zip([str(c) for c in model.classes_], model.predict_proba(X)[0]))
    label = max(probs, key=probs.get)
    # expected score: probability-weighted, so progress shows up between the four category scores
    score = sum(probs.get(c, 0) * CATEGORY_SCORE[c] for c in CATEGORIES)
    return {
        "label": label,
        "score": round(float(score), 1),
        "probabilities": {c: round(float(probs.get(c, 0)), 3) for c in CATEGORIES},
        # share of trees voting for the label - NOT a calibrated probability
        "model_agreement": round(float(np.max(list(probs.values()))), 3),
        "model_variant": used,
    }
