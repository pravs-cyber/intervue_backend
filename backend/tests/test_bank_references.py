import pandas as pd
import pytest

from ml import fill_bank_references as fb

GOOD = ("An array stores elements in contiguous memory, so any index is reached in constant time, but inserting "
        "or deleting in the middle means shifting elements. A linked list stores nodes that point to the next node, "
        "so insertion and deletion at a known position are cheap, but reaching the k-th element needs a walk from "
        "the head, which takes linear time. Arrays are cache friendly and have no pointer overhead; linked lists grow "
        "without reallocation. Choose an array for indexed access and a linked list for frequent insertions.")


@pytest.fixture
def csv(tmp_path, monkeypatch):
    path = tmp_path / "bank_references.csv"
    monkeypatch.setattr(fb, "CSV", path)
    monkeypatch.setattr(fb, "TECHNICAL", [("DSA", 0, "What is the difference between an array and a linked list?"),
                                          ("OS", 1, "What is a deadlock?")])
    monkeypatch.setattr(fb.config, "AI_API_KEY", "test")
    return path


def test_flags():
    q = "What is the difference between an array and a linked list?"
    assert fb.flags_for(q, GOOD) == ""
    assert "empty" in fb.flags_for(q, "")
    assert "markdown" in fb.flags_for(q, GOOD + "\n- bullet")
    assert "refusal" in fb.flags_for(q, GOOD + " As an AI I cannot be sure.")
    assert "key terms" in fb.flags_for(q, "Sorting algorithms such as quicksort divide data. " * 10)
    assert "length" in fb.flags_for(q, "Arrays are contiguous; linked lists use nodes.")


def test_generate_writes_pending_rows_only_to_csv(csv):
    calls = []
    fake = lambda system, user, max_tokens: calls.append(max_tokens) or GOOD
    done, failed = fb.generate(gen=fake, delay=0, log=lambda *_: None)
    df = pd.read_csv(csv, keep_default_na=False)
    assert done == 2 and failed == 0 and set(df.status) == {"pending"} and (df.reference != "").all()
    assert calls == [800, 800]
    assert df["flags"].iloc[0] == "" and "key terms" in df["flags"].iloc[1]          # deadlock answer is about arrays


def test_reviewed_rows_are_never_regenerated(csv):
    fb.generate(gen=lambda s, u, max_tokens: GOOD, delay=0, log=lambda *_: None)
    df = pd.read_csv(csv, keep_default_na=False, dtype=str)
    df.loc[0, ["status", "reference"]] = ["approved", "edited by reviewer " + GOOD]
    df.loc[1, "status"] = "rejected"
    df.to_csv(csv, index=False)
    calls = []
    fb.generate(gen=lambda s, u, max_tokens: calls.append(1) or "new", delay=0, log=lambda *_: None)
    after = pd.read_csv(csv, keep_default_na=False, dtype=str)
    assert calls == [] and after.reference.iloc[0].startswith("edited by reviewer")


def test_apply_writes_only_approved(csv, tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(fb, "TECHNICAL", [("DSA", 0, "What is the difference between an array and a linked list?")])
    fb.generate(gen=lambda s, u, max_tokens: GOOD, delay=0, log=lambda *_: None)
    from database.db import get_conn, init_db
    init_db()                                                                    # pending: nothing applied
    with get_conn() as c:
        ref = c.execute("SELECT reference FROM questions WHERE question LIKE 'What is the difference between an array%'").fetchone()[0]
    assert ref is None
    df = pd.read_csv(csv, keep_default_na=False, dtype=str)
    df.loc[0, "status"] = "approved"
    df.to_csv(csv, index=False)
    with get_conn() as c:
        assert fb.apply(c) == 1
        ref = c.execute("SELECT reference FROM questions WHERE question LIKE 'What is the difference between an array%'").fetchone()[0]
        hr = c.execute("SELECT COUNT(*) FROM questions WHERE type='hr' AND reference IS NOT NULL").fetchone()[0]
    assert ref == GOOD and hr == 0


def test_unknown_status_is_rejected(csv):
    fb.generate(gen=lambda s, u, max_tokens: GOOD, delay=0, log=lambda *_: None)
    df = pd.read_csv(csv, keep_default_na=False, dtype=str)
    df.loc[0, "status"] = "ok"
    df.to_csv(csv, index=False)
    with pytest.raises(AssertionError, match="unknown status"):
        fb.load()


def _fixture_rows(csv, monkeypatch):
    """4 rows: pending+flagged, pending+clean, approved+flagged, rejected+flagged."""
    qs = [("DSA", 0, f"Question {i} about arrays and linked lists?") for i in range(4)]
    monkeypatch.setattr(fb, "TECHNICAL", qs)
    fb.generate(gen=lambda s, u, max_tokens: GOOD, delay=0, log=lambda *_: None)
    df = pd.read_csv(csv, keep_default_na=False, dtype=str)
    df["reference"] = ["old flagged A", GOOD, "old flagged C", "old flagged D"]
    df["flags"] = ["length 3 words (asked 80-120)", "", "length 3 words", "length 3 words"]
    df["status"] = ["pending", "pending", "approved", "rejected"]
    df.to_csv(csv, index=False)
    return df


def test_default_generate_does_not_touch_flagged_rows(csv, monkeypatch):
    _fixture_rows(csv, monkeypatch)
    calls = []
    fb.generate(gen=lambda s, u, max_tokens: calls.append(u) or GOOD, delay=0, log=lambda *_: None)
    assert calls == []


def test_regenerate_flagged_only_touches_pending_flagged_rows(csv, monkeypatch):
    before = _fixture_rows(csv, monkeypatch)
    calls = []
    done, _ = fb.generate(gen=lambda s, u, max_tokens: calls.append(u) or GOOD, delay=0, log=lambda *_: None,
                          regenerate_flagged=True)
    after = pd.read_csv(csv, keep_default_na=False, dtype=str)
    assert done == 1 and len(calls) == 1 and "Question 0" in calls[0]
    assert after.reference.iloc[0] == GOOD and after.previous_reference.iloc[0] == "old flagged A"
    assert after["flags"].iloc[0] == ""                                          # flags recomputed
    for i in (1, 2, 3):                                                          # clean / approved / rejected untouched
        assert after.reference.iloc[i] == before.reference.iloc[i] and after.status.iloc[i] == before.status.iloc[i]
        assert after.previous_reference.iloc[i] == ""


def test_regenerate_flagged_dry_run_writes_nothing(csv, monkeypatch):
    _fixture_rows(csv, monkeypatch)
    raw = csv.read_bytes()
    lines = []
    fb.generate(gen=lambda s, u, max_tokens: 1 / 0, delay=0, log=lines.append, regenerate_flagged=True, dry_run=True)
    assert csv.read_bytes() == raw
    assert sum("would regenerate" in l for l in lines) == 1 and "Question 0" in lines[0]


def test_old_csv_without_previous_reference_column_loads(csv, monkeypatch):
    _fixture_rows(csv, monkeypatch)
    pd.read_csv(csv, dtype=str, keep_default_na=False).drop(columns="previous_reference").to_csv(csv, index=False)
    assert "previous_reference" in fb.load().columns
