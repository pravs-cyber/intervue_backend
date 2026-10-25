"""
Single entry point for TRAINING data.  data/processed/train.csv is the authoritative dataset and is never written.

    load_train(augment=False) -> original train.csv only                       ("existing dataset")
    load_train(augment=True)  -> train.csv + valid rows of llm_train.csv,
                                 appended IN MEMORY                            ("existing dataset + LLM augmentation")
    load_train(jorgpt=True)   -> train.csv + valid rows of data/external/jorgpt/jorgpt_import.csv
                                 (real answers, human teacher grades)          ("existing dataset + JorGPT")
    Both flags together give "existing dataset + LLM augmentation + JorGPT".

LLM rows are kept only if: question, reference and answer are non-empty; label is Poor/Average/Good/Excellent;
kind is one of the four llm_* kinds and matches its label; the question exists in train.csv (so no validation
question can leak in); and (question, kind) is not a duplicate. Rejected rows are counted and reported, never
silently dropped. Every returned row carries `source` = "original" | "llm".
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TRAIN = PROCESSED / "train.csv"
LLM_TRAIN = PROCESSED / "llm_train.csv"
VALIDATION = PROCESSED / "validation.csv"
LABELS = {"Poor", "Average", "Good", "Excellent"}
LLM_KINDS = {"llm_paraphrase_correct": "Excellent", "llm_concise_correct": "Good",
             "llm_partial": "Average", "llm_confident_wrong": "Poor"}
BASELINE_NAME = "existing dataset"
AUGMENTED_NAME = "existing dataset + LLM augmentation"
JORGPT_IMPORT = ROOT / "data" / "external" / "jorgpt" / "jorgpt_import.csv"
ADVERSARIAL = ROOT / "data" / "adversarial"


def validate_external_rows(ext: pd.DataFrame, forbidden_questions: set):
    """External human-graded rows: questions must NOT be ours (train/validation/adversarial)."""
    norm = ext.question.astype(str).str.strip().str.lower()
    checks = [
        ("missing question/answer/reference", _filled(ext.question) & _filled(ext.answer) & _filled(ext.reference)),
        ("unknown label", ext.label.isin(LABELS)),
        ("question overlaps our train/validation/adversarial data", ~norm.isin(forbidden_questions)),
    ]
    ok, rejected = pd.Series(True, index=ext.index), {}
    for reason, passed in checks:
        rejected[reason] = int((ok & ~passed).sum())
        ok &= passed
    return ext[ok].copy(), {k: v for k, v in rejected.items() if v}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def _filled(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def validate_llm_rows(llm: pd.DataFrame, train_questions: set, validation_questions: set):
    """Return (accepted rows, {reason: count}) - each rejected row is counted under its first failing check."""
    checks = [
        ("missing question/answer/reference", _filled(llm.question) & _filled(llm.answer) & _filled(llm.reference)),
        ("unknown label", llm.label.isin(LABELS)),
        ("unknown kind or kind/label mismatch", llm.kind.map(LLM_KINDS).eq(llm.label)),
        ("validation question (leakage)", ~llm.question.isin(validation_questions)),
        ("question not in train.csv", llm.question.isin(train_questions)),
    ]
    ok, rejected = pd.Series(True, index=llm.index), {}
    for reason, passed in checks:
        rejected[reason] = int((ok & ~passed).sum())
        ok &= passed
    dup = ok & llm.duplicated(["question", "kind"], keep="first")
    rejected["duplicate (question, kind)"] = int(dup.sum())
    ok &= ~dup
    return llm[ok].copy(), {k: v for k, v in rejected.items() if v}


def load_train(augment: bool = False, verbose: bool = True, jorgpt: bool = False):
    """Returns (train DataFrame, info dict). The CSV files are only read."""
    train = pd.read_csv(TRAIN)
    train["source"] = "original"
    info = {"training_data": BASELINE_NAME, "train_sha256": sha(TRAIN), "n_original": len(train),
            "n_llm": 0, "llm_sha256": "", "llm_rejected": {}}
    if augment:
        if not LLM_TRAIN.exists():
            raise SystemExit(f"--augment given but {LLM_TRAIN} does not exist")
        llm = pd.read_csv(LLM_TRAIN)
        val_q = set(pd.read_csv(VALIDATION).question)
        acc, rej = validate_llm_rows(llm, set(train.question), val_q)
        acc["source"] = "llm"
        cols = list(train.columns)
        train = pd.concat([train, acc.reindex(columns=cols)], ignore_index=True)
        info.update(training_data=AUGMENTED_NAME, n_llm=len(acc), llm_sha256=sha(LLM_TRAIN), llm_rejected=rej)
        assert not set(train.question) & val_q, "validation question in training data"
    info.update(n_jorgpt=0, jorgpt_sha256="", jorgpt_rejected={})
    if jorgpt:
        if not JORGPT_IMPORT.exists():
            raise SystemExit(f"--jorgpt given but {JORGPT_IMPORT} does not exist - run ml/import_jorgpt.py")
        ext = pd.read_csv(JORGPT_IMPORT)
        ours = set(pd.read_csv(TRAIN).question) | set(pd.read_csv(VALIDATION).question)
        for f in ADVERSARIAL.glob("adversarial_*.json"):
            ours |= set(pd.read_json(f).question)
        acc, rej = validate_external_rows(ext, {q.strip().lower() for q in ours})
        acc["source"] = "jorgpt"
        cols = list(train.columns) + [c for c in ("question_id", "teacher_grade") if c not in train.columns]
        train = pd.concat([train, acc.reindex(columns=cols)], ignore_index=True)
        name = info["training_data"] + " + JorGPT"
        info.update(training_data=name, n_jorgpt=len(acc), jorgpt_sha256=sha(JORGPT_IMPORT), jorgpt_rejected=rej)
    info["n_total"] = len(train)
    if verbose:
        msg = f"Training data: {info['training_data']} - {info['n_original']} original rows"
        if augment:
            msg += f" + {info['n_llm']} LLM rows (of {info['n_llm'] + sum(info['llm_rejected'].values())}; " \
                   f"rejected: {info['llm_rejected'] or 'none'})"
        if jorgpt:
            msg += f" + {info['n_jorgpt']} JorGPT rows (rejected: {info['jorgpt_rejected'] or 'none'})"
        msg += f" = {info['n_total']} rows in memory"
        print(msg)
    return train, info
