import pandas as pd
import pytest

from ml import import_jorgpt as ij

LLM_COLS = ["deepseek_raw_json", "deepseek_grade", "qwen_raw_json", "qwen_grade", "gemini_raw_json", "gemini_grade",
            "judge_status", "judge_raw_json", "judge_grade"]


def make(tmp_path, rows, questions):
    src, proc = tmp_path / "jorgpt", tmp_path / "processed"
    src.mkdir()
    proc.mkdir()
    base = []
    for i, (qid, ans, grade, student) in enumerate(rows):
        r = {"entry_id": i, "question_id": qid, "question_text": "garbled english text", "student_answer": ans,
             "ideal_answer": f"ideal for {qid}", "student_anom_id": student, "student_answer_length": 1,
             "teacher_grade": grade, "teacher_feedback": "fb", "teacher_corrected_at": "", "teacher_corrected": True}
        r.update({c: 9.9 if c.endswith("grade") else "x" for c in LLM_COLS})       # LLM grades must be ignored
        base.append(r)
    en = pd.DataFrame(base)
    es = en.copy()
    es["question_text"] = es.question_id.map({q: es_t for q, es_t, _ in questions})
    en.to_csv(src / "dataset_en.csv", index=False)
    es.to_csv(src / "dataset_es.csv", index=False)
    pd.DataFrame([{"question_id": q, "question_es": e, "question_en": t, "course_specific": False, "note": "",
                   "answers": 0} for q, e, t in questions]).to_csv(src / "questions_en.csv", index=False)
    pd.DataFrame({"question": ["Our own question?"]}).to_csv(proc / "train.csv", index=False)
    return src, proc


QS = [("Q1", "¿Qué es Docker?", "What is Docker?"),
      ("Q2", "Explica el sistema Pokémon.", "Explain the Pokémon system."),
      ("Q3", "Servicios pokemon-db y pokemon-backend.", "Services pokemon-db and pokemon-backend.")]


def test_mapping_boundaries():
    assert [ij.grade_to_label(g) for g in range(11)] == \
        ["Poor"] * 3 + ["Average"] * 3 + ["Good"] * 2 + ["Excellent"] * 3


def test_import_schema_labels_and_drops(tmp_path, monkeypatch):
    monkeypatch.setattr(ij, "ROOT", tmp_path)                       # no adversarial dir -> skipped
    rows = [("Q1", "Docker runs containers.", 9, "s1"), ("Q1", "", 5, "s2"),          # blank answer -> dropped
            ("Q1", "same text", 3, "s3"), ("Q1", "same text", 0, "s3"),             # conflicting copies -> both dropped
            ("Q2", "dup ok", 6, "s4"), ("Q2", "dup ok", 6, "s4"),                   # agreeing copies -> one kept
            ("Q2", "first try", 2, "s5"), ("Q2", "second try", 4, "s5"),            # different answers -> both kept
            ("Q3", "depends_on orders startup.", 7, "s6")]
    src, proc = make(tmp_path, rows, QS)
    df, dropped, checks = ij.build(src, proc, src / 'questions_en.csv')
    assert list(df.columns[:5]) == ij.SCHEMA and set(ij.META) <= set(df.columns)
    assert not any(c in df.columns for c in LLM_COLS)
    assert len(df) == 5
    assert dropped["blank answer"] == 1
    assert dropped["identical resubmission with conflicting grades (all copies)"] == 2
    assert dropped["identical resubmission, same grade (extra copies)"] == 1
    assert dict(zip(df.answer, df.label)) == {"Docker runs containers.": "Excellent", "dup ok": "Good",
                                             "first try": "Poor", "second try": "Average",
                                             "depends_on orders startup.": "Good"}
    assert (df.question[df.question_id == "Q1"] == "What is Docker?").all()      # English question from mapping
    assert set(df.kind) == {"jorgpt_human"} and set(df.source) == {"jorgpt"}
    flags = df.groupby("question_id").course_specific.first().to_dict()
    assert flags == {"Q1": False, "Q2": True, "Q3": True}                     # lower-case pokemon-db caught


def test_translation_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ij, "ROOT", tmp_path)
    src, proc = make(tmp_path, [("Q1", "a", 5, "s1")], QS[:1])
    q = pd.read_csv(src / "questions_en.csv")
    q.loc[0, "question_es"] = "texto distinto"
    q.to_csv(src / "questions_en.csv", index=False)
    with pytest.raises(AssertionError, match="Spanish text differs"):
        ij.build(src, proc, src / 'questions_en.csv')


def test_overlap_with_our_questions_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ij, "ROOT", tmp_path)
    src, proc = make(tmp_path, [("Q1", "a", 5, "s1")], QS[:1])
    pd.DataFrame({"question": ["What is Docker?"]}).to_csv(proc / "train.csv", index=False)
    with pytest.raises(AssertionError, match="overlap"):
        ij.build(src, proc, src / 'questions_en.csv')
