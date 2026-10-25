import hashlib

import pandas as pd
import pytest

from ml import training_data as td


@pytest.fixture
def files(tmp_path, monkeypatch):
    train = pd.DataFrame([{"question": f"T{i}?", "reference": f"ref {i}", "answer": f"orig {i}", "label": "Good",
                           "kind": "truncated"} for i in range(5)])
    val = pd.DataFrame([{"question": "V0?", "reference": "vref", "answer": "a", "label": "Good", "kind": "truncated"}])
    paths = {"TRAIN": tmp_path / "train.csv", "VALIDATION": tmp_path / "validation.csv", "LLM_TRAIN": tmp_path / "llm_train.csv"}
    train.to_csv(paths["TRAIN"], index=False)
    val.to_csv(paths["VALIDATION"], index=False)
    for k, v in paths.items():
        monkeypatch.setattr(td, k, v)
    return paths


def llm_row(q="T0?", kind="llm_paraphrase_correct", label="Excellent", answer="An answer.", ref="ref 0"):
    return {"question": q, "reference": ref, "answer": answer, "label": label, "kind": kind}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_baseline_is_exactly_train_csv(files):
    df, info = td.load_train(augment=False, verbose=False)
    orig = pd.read_csv(files["TRAIN"])
    assert len(df) == len(orig) == info["n_original"] and info["n_llm"] == 0
    assert info["training_data"] == "existing dataset" and set(df.source) == {"original"}
    pd.testing.assert_frame_equal(df.drop(columns="source"), orig)


def test_augment_appends_valid_rows_in_memory_only(files):
    rows = [llm_row(f"T{i % 5}?", k, lab, ref=f"ref {i % 5}") for i, (k, lab) in
            enumerate(list(td.LLM_KINDS.items()) * 10)][:39]            # 39 rows, as in the current file
    pd.DataFrame(rows).to_csv(files["LLM_TRAIN"], index=False)
    before = {k: digest(p) for k, p in files.items()}
    df, info = td.load_train(augment=True, verbose=False)
    accepted = len(pd.DataFrame(rows).drop_duplicates(["question", "kind"]))
    assert info["training_data"] == "existing dataset + LLM augmentation"
    assert info["n_llm"] == accepted and len(df) == 5 + accepted
    assert (df.source == "llm").sum() == accepted
    assert {k: digest(p) for k, p in files.items()} == before          # nothing written


def test_invalid_llm_rows_are_rejected_and_reported(files):
    rows = [llm_row(),                                                  # valid
            llm_row(kind="llm_partial", label="Average", ref=""),       # missing reference
            llm_row(kind="llm_concise_correct", label="Excellent"),     # kind/label mismatch
            llm_row(q="V0?", ref="vref"),                               # validation question
            llm_row(q="NEW?"),                                          # not in train.csv
            llm_row(answer="different text"),                           # duplicate (question, kind)
            llm_row(kind="llm_confident_wrong", label="Terrible")]      # unknown label
    pd.DataFrame(rows).to_csv(files["LLM_TRAIN"], index=False)
    df, info = td.load_train(augment=True, verbose=False)
    assert info["n_llm"] == 1 and len(df) == 6
    assert info["llm_rejected"] == {"missing question/answer/reference": 1, "unknown label": 1,
                                    "unknown kind or kind/label mismatch": 1, "validation question (leakage)": 1,
                                    "question not in train.csv": 1, "duplicate (question, kind)": 1}
    assert "V0?" not in set(df.question)


def test_augment_without_file_fails_loudly(files):
    with pytest.raises(SystemExit):
        td.load_train(augment=True, verbose=False)


def test_jorgpt_source_appends_in_memory_and_rejects_overlap(files, tmp_path, monkeypatch):
    ext = tmp_path / "jorgpt_import.csv"
    pd.DataFrame([
        {"question": "What is Docker?", "reference": "r", "answer": "a container tool", "label": "Good", "kind": "jorgpt_human"},
        {"question": "T0?", "reference": "r", "answer": "copies our question", "label": "Good", "kind": "jorgpt_human"},
        {"question": "V0?", "reference": "r", "answer": "copies validation", "label": "Poor", "kind": "jorgpt_human"},
        {"question": "Why REST?", "reference": "r", "answer": "x", "label": "Superb", "kind": "jorgpt_human"},
    ]).to_csv(ext, index=False)
    monkeypatch.setattr(td, "JORGPT_IMPORT", ext)
    monkeypatch.setattr(td, "ADVERSARIAL", tmp_path / "no_adversarial_here")
    before = {k: digest(p) for k, p in files.items() if p.exists()} | {"ext": digest(ext)}
    df, info = td.load_train(jorgpt=True, verbose=False)
    assert info["training_data"] == "existing dataset + JorGPT" and info["n_jorgpt"] == 1
    assert info["jorgpt_rejected"] == {"question overlaps our train/validation/adversarial data": 2, "unknown label": 1}
    assert len(df) == 6 and (df.source == "jorgpt").sum() == 1
    assert ({k: digest(p) for k, p in files.items() if p.exists()} | {"ext": digest(ext)}) == before
