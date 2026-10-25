"""
Read-only analysis: which mapping of JorGPT's 0-10 teacher grades onto Poor/Average/Good/Excellent fits our labels?
Run from backend/:  python ml/analyze_jorgpt_mapping.py        -> ml/results/jorgpt_mapping_report.txt
Nothing is imported or trained. Uses only teacher_grade (human); the LLM grades are ignored.

Because 43/50 question_text fields are corrupted, the main checks use reference-only features (answer vs
ideal_answer), computed with the feature extractor fitted on OUR training data:
  1. class sizes per mapping
  2. are the categories ordered? within-question Spearman of features with the grade, and per-class medians
  3. how separable are ADJACENT classes? within-question AUC of ref_emb / ref_coverage, averaged
  4. how do JorGPT categories compare with the same features on our own training categories?
  5. agreement of the v3 candidate with each mapping, on the 7 questions whose question text is usable
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.features import FeatureExtractor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "jorgpt" / "dataset_en.csv"
OUT = Path(__file__).resolve().parent / "results" / "jorgpt_mapping_report.txt"
CATS = ["Poor", "Average", "Good", "Excellent"]
ORD = {c: i for i, c in enumerate(CATS)}

# grade (0-10 integers) -> category; inclusive upper bounds
MAPPINGS = {
    "M1 academic (fail<5, 9-10 top)": {"Poor": 4, "Average": 6, "Good": 8, "Excellent": 10},
    "M2 scale quarters":              {"Poor": 2, "Average": 5, "Good": 7, "Excellent": 10},
    "M3 balanced":                    {"Poor": 3, "Average": 5, "Good": 7, "Excellent": 10},
    "M4 no Excellent (3-way)":        {"Poor": 3, "Average": 6, "Good": 10},
}
FEATS = ["ref_cosine", "ref_coverage", "ref_lsa", "ref_emb", "log_words"]


def apply_map(grades, m):
    out = pd.Series(index=grades.index, dtype=object)
    lo = -1
    for cat, hi in m.items():
        out[(grades > lo) & (grades <= hi)] = cat
        lo = hi
    return out


def clean_question(t):
    return len(str(t).split()) <= 45 and "**" not in str(t) and str(t)[:1].isupper()


def main():
    d = pd.read_csv(SRC)
    d = d[d.student_answer.notna() & d.teacher_grade.notna()].reset_index(drop=True)
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    fx = FeatureExtractor().fit(train.question, train.reference)          # fitted on OUR training data
    groups = ("surface", "reference", "semantic", "embedding")
    X, names = fx.transform([""] * len(d), list(d.student_answer), list(d.ideal_answer), groups=groups)
    F = pd.DataFrame(X, columns=names)[FEATS]
    Xt, _ = fx.transform([""] * len(train), list(train.answer), list(train.reference), groups=groups)
    Ft = pd.DataFrame(Xt, columns=names)[FEATS]
    L = [f"JorGPT rows used: {len(d)} (50 questions; teacher_grade only)\n"]

    L.append("1+2. Within-question Spearman rho between teacher grade and each feature (higher = feature tracks grade):")
    for f in FEATS:
        rhos = [spearmanr(g.teacher_grade, F.loc[g.index, f]).statistic for _, g in d.groupby("question_id")
                if g.teacher_grade.nunique() > 1]
        L.append(f"   {f:13s} median rho {np.nanmedian(rhos):+.2f}")

    L.append("\nOur own training data, median feature per category (for comparison):")
    L.append(pd.concat([Ft, train.label], axis=1).groupby("label")[FEATS].median().reindex(CATS).round(2).to_string())

    for name, m in MAPPINGS.items():
        lab = apply_map(d.teacher_grade, m)
        cats = [c for c in CATS if c in m]
        L.append(f"\n================ {name}: " + ", ".join(f"{c} <= {m[c]}" for c in cats))
        L.append("class sizes: " + ", ".join(f"{c} {int((lab == c).sum())}" for c in cats))
        L.append("median features per class:")
        L.append(pd.concat([F, lab.rename("cat")], axis=1).groupby("cat")[FEATS].median().reindex(cats).round(2).to_string())
        L.append("adjacent-class separability (within-question AUC, mean over questions; 0.5 = indistinguishable):")
        for a, b in zip(cats, cats[1:]):
            row = []
            for f in ["ref_emb", "ref_coverage", "log_words"]:
                aucs = []
                for _, g in pd.concat([F, lab.rename("cat"), d.question_id], axis=1).groupby("question_id"):
                    g = g[g.cat.isin([a, b])]
                    if g.cat.nunique() == 2 and len(g) >= 6:
                        aucs.append(roc_auc_score(g.cat == b, g[f]))
                row.append(f"{f} {np.mean(aucs):.2f} (n_q={len(aucs)})")
            L.append(f"   {a:8s} vs {b:9s}: " + "   ".join(row))

    # 5. v3 candidate agreement on clean-question rows only
    try:
        from ml import answer_classifier as ac
        ac.BUNDLE_PATH = Path(__file__).resolve().parent / "candidates" / "v3" / "answer_model.pkl"
        cq = d[d.question_text.map(clean_question)].reset_index(drop=True)
        pred = [ac.predict(a, q, r)["label"] for q, a, r in zip(cq.question_text, cq.student_answer, cq.ideal_answer)]
        L.append(f"\n5. v3 candidate vs each mapping on the {cq.question_id.nunique()} clean-question questions "
                 f"({len(cq)} answers). v3 predicted distribution: {pd.Series(pred).value_counts().to_dict()}")
        for name, m in MAPPINGS.items():
            lab = apply_map(cq.teacher_grade, m)
            k = cohen_kappa_score(lab.map(ORD), pd.Series(pred).map(ORD), weights="quadratic")
            L.append(f"   {name:34s} quadratic-weighted kappa {k:.2f}   exact agreement {(lab.values == np.array(pred)).mean():.2f}")
    except Exception as e:
        L.append(f"\n5. v3 agreement skipped: {e}")
    text = "\n".join(L)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
