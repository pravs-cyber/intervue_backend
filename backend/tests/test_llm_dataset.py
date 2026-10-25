import httpx
import pandas as pd

from ml import generate_llm_dataset as gen

KINDS = {"llm_paraphrase_correct", "llm_concise_correct", "llm_partial", "llm_confident_wrong"}


def make_root(tmp_path):
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    rows = [{"question": f"Q{i}?", "reference": f"Reference answer {i}.", "answer": "x", "label": "Excellent",
             "kind": "reference"} for i in range(3)]
    pd.DataFrame(rows).to_csv(d / "train.csv", index=False)
    return tmp_path, d / "llm_train.csv"


def fake_chat(system, user, max_tokens=300):
    return f"answer to {user} ({system[-20:]})"


def check(df):
    answered = df[df.answer.notna() & (df.answer.astype(str).str.strip() != "")]
    assert len(answered) and answered.reference.notna().all()
    assert (answered.reference.astype(str).str.strip() != "").all()
    for q, ref in zip(answered.question, answered.reference):
        assert ref == "Reference answer " + q[1:-1] + "."          # the reference of THAT question
    assert set(df.kind) <= KINDS
    assert (df.label == df.kind.map(gen.KINDS)).all()
    assert not df.duplicated(["question", "kind"]).any()


def test_every_generated_row_has_its_reference(tmp_path):
    root, out = make_root(tmp_path)
    gen.generate("train", root=root, chat=fake_chat, save_every=1, delay=0)
    df = pd.read_csv(out)
    assert len(df) == 12
    check(df)


def test_rerun_adds_no_duplicates_and_keeps_rows(tmp_path):
    root, out = make_root(tmp_path)
    gen.generate("train", limit=2, root=root, chat=fake_chat, delay=0)
    first = pd.read_csv(out)
    gen.generate("train", root=root, chat=fake_chat, delay=0)          # resumes: only Q2 is new
    df = pd.read_csv(out)
    assert len(df) == 12
    pd.testing.assert_frame_equal(df.head(len(first)), first)   # existing valid rows untouched
    check(df)


def test_repair_fills_empty_confident_wrong_references(tmp_path):
    root, out = make_root(tmp_path)
    gen.generate("train", root=root, chat=fake_chat, delay=0)
    df = pd.read_csv(out)
    df.loc[df.kind == "llm_confident_wrong", "reference"] = ""
    df = pd.concat([df, df.iloc[[0]]])                          # a duplicate row too
    df.to_csv(out, index=False)
    refs = gen.reference_lookup(pd.read_csv(root / "data" / "processed" / "train.csv"))
    fixed = gen.repair(pd.read_csv(out), refs)
    assert len(fixed) == 12
    check(fixed)


def test_empty_answers_are_regenerated_not_kept(tmp_path):
    root, out = make_root(tmp_path)
    gen.generate("train", root=root, chat=fake_chat, delay=0)
    df = pd.read_csv(out)
    df.loc[df.kind == "llm_partial", "answer"] = ""
    df.to_csv(out, index=False)
    gen.generate("train", root=root, chat=fake_chat, delay=0)
    df = pd.read_csv(out)
    assert len(df) == 12 and (df.answer.astype(str).str.strip() != "").all()
    check(df)


# ---------------------------------------------------------------- rate limiting (HTTP 429)
def http_429(retry_after=None):
    req = httpx.Request("POST", "https://api.example/chat/completions")
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, headers=headers, request=req))


class FlakyChat:
    """Raises 429 for the first `fail` calls, then answers."""
    def __init__(self, fail, retry_after=None):
        self.fail, self.retry_after, self.calls = fail, retry_after, 0

    def __call__(self, system, user, max_tokens=300):
        self.calls += 1
        if self.calls <= self.fail:
            raise http_429(self.retry_after)
        return fake_chat(system, user)


def test_retry_after_header_is_respected():
    waits = []
    chat = FlakyChat(fail=2, retry_after=7)
    out = gen.call_with_retry(chat, "s", "u", sleep=waits.append, log=lambda *_: None)
    assert out.startswith("answer to") and waits == [7.0, 7.0]


def test_exponential_backoff_without_header():
    waits = []
    gen.call_with_retry(FlakyChat(fail=3), "s", "u", sleep=waits.append, log=lambda *_: None)
    assert waits == [5, 15, 30]


def test_exhausted_retries_raise_rate_limited():
    import pytest
    with pytest.raises(gen.RateLimited):
        gen.call_with_retry(FlakyChat(fail=99), "s", "u", sleep=lambda _: None, log=lambda *_: None)


def test_huge_retry_after_gives_up_without_sleeping():
    import pytest
    waits = []
    with pytest.raises(gen.RateLimited):
        gen.call_with_retry(FlakyChat(fail=1, retry_after=3600), "s", "u", sleep=waits.append, log=lambda *_: None)
    assert waits == []


def test_non_429_errors_are_not_retried():
    import pytest
    def boom(system, user, max_tokens=300):
        raise ValueError("bad json")
    with pytest.raises(ValueError):
        gen.call_with_retry(boom, "s", "u", sleep=lambda _: None, log=lambda *_: None)


def test_rate_limited_question_is_skipped_then_resumed(tmp_path):
    root, out = make_root(tmp_path)
    calls = {"n": 0}

    def limited_on_q1(system, user, max_tokens=300):
        if "Q1?" in user:
            raise http_429()
        return fake_chat(system, user)

    _, st = gen.generate("train", root=root, chat=limited_on_q1, delay=0, sleep=lambda _: None,
                         log=lambda *_: None)
    df = pd.read_csv(out)
    assert st["generated"] == 8 and st["skipped"] == 4      # Q1's four pairs skipped, nothing counted
    assert set(df.question) == {"Q0?", "Q2?"} and len(df) == 8
    check(df)
    _, st = gen.generate("train", root=root, chat=fake_chat, delay=0, log=lambda *_: None)   # resume
    assert st["existing"] == 8 and st["generated"] == 4
    df = pd.read_csv(out)
    assert len(df) == 12
    check(df)


def test_delay_only_after_successful_calls(tmp_path):
    root, _ = make_root(tmp_path)
    waits = []
    gen.generate("train", limit=1, root=root, chat=FlakyChat(fail=1), delay=2, sleep=waits.append,
                 log=lambda *_: None)
    assert waits == [5, 2, 2, 2, 2]                          # one backoff, then a delay after each of 4 successes


def test_limit_counts_questions_and_dry_run_counts_missing(tmp_path, capsys):
    root, out = make_root(tmp_path)
    _, st = gen.generate("train", limit=2, root=root, chat=fake_chat, delay=0, log=lambda *_: None)
    assert st["target"] == 8 and st["generated"] == 8
    _, qs, _, done, _ = gen.plan("train", None, root)
    assert len(qs) == 3 and len(gen.missing_pairs(qs, done)) == 4
    _, qs, _, done, _ = gen.plan("train", 2, root)
    assert gen.missing_pairs(qs, done) == []


def test_three_rate_limited_questions_in_a_row_stop_the_run(tmp_path):
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    pd.DataFrame([{"question": f"Q{i}?", "reference": f"Reference answer {i}.", "answer": "x",
                   "label": "Excellent", "kind": "reference"} for i in range(6)]).to_csv(d / "train.csv", index=False)
    chat = FlakyChat(fail=10 ** 6)
    _, st = gen.generate("train", root=tmp_path, chat=chat, delay=0, sleep=lambda _: None, log=lambda *_: None)
    assert st["generated"] == 0 and st["skipped"] == 24
    assert chat.calls == 3 * (len(gen.BACKOFF) + 1)          # stopped after Q2, never called for Q3-Q5


# ---------------------------------------------------------------- empty completions
class FakeResponse:
    def __init__(self, content, finish_reason="length"):
        self._body = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_chat_rejects_empty_content(monkeypatch):
    import pytest
    from services import llm_service
    for content in ["", "   \n", None]:
        monkeypatch.setattr(llm_service.httpx, "post", lambda *a, c=content, **k: FakeResponse(c))
        with pytest.raises(llm_service.EmptyCompletion):
            llm_service._chat("s", "u", max_tokens=800)
    monkeypatch.setattr(llm_service.httpx, "post", lambda *a, **k: FakeResponse("  A real answer. ", "stop"))
    assert llm_service._chat("s", "u") == "A real answer."


def test_empty_completion_is_skipped_not_saved_and_not_retried(tmp_path):
    from services import llm_service
    root, out = make_root(tmp_path)
    calls = {"n": 0}

    def empty_for_partial(system, user, max_tokens=800):
        calls["n"] += 1
        if "blur the most important detail" in system:          # the 'partial' (Average) prompt
            raise llm_service.EmptyCompletion("empty message.content")
        return fake_chat(system, user)

    waits = []
    _, st = gen.generate("train", root=root, chat=empty_for_partial, delay=0, sleep=waits.append, log=lambda *_: None)
    df = pd.read_csv(out)
    assert st["generated"] == 9 and st["skipped"] == 3 and calls["n"] == 12   # one call per pair, no 429 retries
    assert waits == [] and "llm_partial" not in set(df.kind)
    check(df)
    _, st = gen.generate("train", root=root, chat=fake_chat, delay=0, log=lambda *_: None)   # resume fills them
    assert st["generated"] == 3 and len(pd.read_csv(out)) == 12


def test_max_tokens_reaches_the_api_call(tmp_path):
    root, _ = make_root(tmp_path)
    seen = set()

    def spy(system, user, max_tokens=None):
        seen.add(max_tokens)
        return fake_chat(system, user)

    gen.generate("train", limit=1, root=root, chat=spy, delay=0, log=lambda *_: None)
    assert seen == {800}
    gen.generate("train", root=root, chat=spy, delay=0, log=lambda *_: None, max_tokens=1200)
    assert seen == {800, 1200}
