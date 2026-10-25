"""
Import JorGPT (real student answers + human teacher grades) into our training-data schema. No training.
Run from backend/:   python ml/import_jorgpt.py

Inputs (data/external/jorgpt/, never modified):
    dataset_en.csv      answers, ideal answers, teacher grades (English)
    dataset_es.csv      original Spanish questions (used to verify the translation mapping)
and data/sources/jorgpt/questions_en.csv (TRACKED): the 50 reviewed Spanish -> English question translations
Outputs (data/external/jorgpt/, gitignored):
    jorgpt_import.csv           one row per graded answer, training schema + metadata
    jorgpt_import_summary.txt   counts by label / grade / question, every dropped row by reason

Decisions (fixed before any training):
    label   = teacher_grade ONLY, mapped 0-2 Poor | 3-5 Average | 6-7 Good | 8-10 Excellent  ("M2")
    ignored = deepseek / qwen / gemini / judge grades and raw JSON
    blank answers dropped; course-specific questions KEPT (flag is metadata, not a filter)
train.csv, validation.csv, the adversarial sets and all models are untouched.

Schema: question, reference, answer, label, kind  (same first five columns as train.csv)
        + source, question_id, entry_id, student_id, teacher_grade, course_specific, teacher_feedback
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "jorgpt"
PROCESSED = ROOT / "data" / "processed"
MAPPING = ROOT / "data" / "sources" / "jorgpt" / "questions_en.csv"

M2 = [(2, "Poor"), (5, "Average"), (7, "Good"), (10, "Excellent")]     # inclusive upper bounds
COURSE = re.compile(r"pok[eé]mon|swapi|\bP13\b", re.I)
SCHEMA = ["question", "reference", "answer", "label", "kind"]
META = ["source", "question_id", "entry_id", "student_id", "teacher_grade", "course_specific", "teacher_feedback"]


def grade_to_label(g):
    for hi, lab in M2:
        if g <= hi:
            return lab
    raise ValueError(g)


def _filled(s):
    return s.notna() & (s.astype(str).str.strip() != "")


def build(src: Path = SRC, processed: Path = PROCESSED, mapping: Path = None):
    en = pd.read_csv(src / "dataset_en.csv")
    es = pd.read_csv(src / "dataset_es.csv")
    qmap = pd.read_csv(mapping or MAPPING)
    checks = []

    # --- the two language files must describe the same rows
    m = en.merge(es[["entry_id", "question_id", "teacher_grade", "student_anom_id"]], on="entry_id",
                 suffixes=("", "_es"), how="left")
    assert len(en) == len(es) == len(m), "dataset_en / dataset_es row counts differ"
    for col in ("question_id", "teacher_grade", "student_anom_id"):
        assert (m[col] == m[col + "_es"]).all(), f"{col} differs between dataset_en and dataset_es"
    checks.append(f"dataset_en and dataset_es aligned on entry_id: {len(m)} rows, question_id/teacher_grade/student identical")

    # --- the translation mapping must cover every question and match the Spanish source exactly
    es_q = es.drop_duplicates("question_id").set_index("question_id").question_text
    assert set(qmap.question_id) == set(es_q.index) and len(qmap) == len(es_q) == qmap.question_id.nunique(), \
        "questions_en.csv does not cover exactly the dataset's question IDs"
    assert (qmap.set_index("question_id").question_es == es_q.reindex(qmap.question_id).values).all(), \
        "questions_en.csv Spanish text differs from dataset_es.csv"
    assert _filled(qmap.question_en).all(), "empty English question"
    checks.append(f"questions_en.csv: {len(qmap)} questions, Spanish text identical to dataset_es.csv")

    # --- build rows
    q = qmap.set_index("question_id")
    course = (q.question_es.fillna("") + " " + q.question_en.fillna("")).map(lambda t: bool(COURSE.search(t)))
    df = pd.DataFrame({
        "question": m.question_id.map(q.question_en), "reference": m.ideal_answer, "answer": m.student_answer,
        "teacher_grade": m.teacher_grade, "source": "jorgpt", "kind": "jorgpt_human",
        "question_id": m.question_id, "entry_id": m.entry_id, "student_id": m.student_anom_id,
        "course_specific": m.question_id.map(course), "teacher_feedback": m.teacher_feedback})

    # --- validate; every dropped row is counted by reason
    reasons = [
        ("blank answer", _filled(df.answer)),
        ("blank reference", _filled(df.reference)),
        ("missing teacher_grade", df.teacher_grade.notna()),
        ("teacher_grade not an integer 0-10", df.teacher_grade.between(0, 10) & (df.teacher_grade == df.teacher_grade.round())),
    ]
    keep, dropped = pd.Series(True, index=df.index), {}
    for why, ok in reasons:
        dropped[why] = int((keep & ~ok).sum())
        keep &= ok
    # same student, same question, same text: one copy if the grades agree; if they conflict, the label is
    # undecidable, so every copy is dropped (in this dataset: one pair graded 3 and 0)
    key = ["question_id", "student_id", "answer"]
    grp = df[keep].groupby(key, dropna=False).teacher_grade
    n_grades = df[key].merge(grp.nunique().rename("n").reset_index(), on=key, how="left").n.values
    conflict = keep & (n_grades > 1)
    dropped["identical resubmission with conflicting grades (all copies)"] = int(conflict.sum())
    keep &= ~conflict
    dup = keep & df.duplicated(key, keep="first")
    dropped["identical resubmission, same grade (extra copies)"] = int(dup.sum())
    keep &= ~dup
    other = keep & df.duplicated(["question_id", "student_id"], keep=False)
    if other.any():
        checks.append(f"note: {int(other.sum())} rows are different answers by the same student to the same question - kept")
    df = df[keep].copy()
    df["teacher_grade"] = df.teacher_grade.astype(int)
    df["label"] = df.teacher_grade.map(grade_to_label)

    # --- leakage checks against our own data (questions must not appear in train/validation/adversarial sets)
    ours = set()
    for f in ("train.csv", "validation.csv"):
        if (processed / f).exists():
            ours |= set(pd.read_csv(processed / f).question.str.strip().str.lower())
    for f in (ROOT / "data" / "adversarial").glob("adversarial_*.json"):
        ours |= set(pd.read_json(f).question.str.strip().str.lower())
    overlap = set(df.question.str.strip().str.lower()) & ours
    assert not overlap, f"JorGPT questions overlap our data: {list(overlap)[:3]}"
    checks.append("no JorGPT question appears in train.csv, validation.csv or the adversarial sets")
    return df[SCHEMA + META].reset_index(drop=True), dropped, checks


def summary(df, dropped, checks):
    by_q = df.groupby("question_id").agg(answers=("answer", "size"), mean_grade=("teacher_grade", "mean"),
                                         course_specific=("course_specific", "first"))
    words = df.answer.str.split().str.len()
    L = ["JorGPT import (M2 mapping, teacher_grade only)", *[f"check: {c}" for c in checks],
         f"\nrows imported: {len(df)}   questions: {df.question_id.nunique()}   students: {df.student_id.nunique()}",
         "rows dropped: " + (", ".join(f"{k} {v}" for k, v in dropped.items() if v) or "none"),
         "\nby label:\n" + df.label.value_counts().reindex(["Poor", "Average", "Good", "Excellent"]).to_string(),
         "\nby teacher_grade:\n" + df.teacher_grade.value_counts().sort_index().to_string(),
         "\nlabel by course_specific:\n" + pd.crosstab(df.course_specific, df.label)[["Poor", "Average", "Good", "Excellent"]].to_string(),
         f"\ncourse-specific questions: {int(by_q.course_specific.sum())} of {len(by_q)} "
         f"({int(df.course_specific.sum())} answers)",
         "\nanswers per question: " + by_q.answers.describe()[["min", "50%", "max"]].to_string().replace("\n", " | "),
         "answer words: " + words.describe()[["min", "50%", "mean", "max"]].round(1).to_string().replace("\n", " | "),
         "\nanswer words by label (median): " + df.assign(w=words).groupby("label").w.median()
         .reindex(["Poor", "Average", "Good", "Excellent"]).to_string().replace("\n", " | "),
         "\nper question (id, course_specific, answers, mean grade):"]
    L += [f"  {i}  {'C' if r.course_specific else '-'}  {r.answers:3d}  {r.mean_grade:.1f}" for i, r in by_q.iterrows()]
    return "\n".join(L)


def main():
    df, dropped, checks = build()
    out = SRC / "jorgpt_import.csv"
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    text = summary(df, dropped, checks)
    (SRC / "jorgpt_import_summary.txt").write_text(text)
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main())
