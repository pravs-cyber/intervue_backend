from services import interview_service as svc
from services.constants import LEVELS


def start(client, auth, **kw):
    body = {"type": "technical", "topics": ["DSA", "SQL"], "num_questions": 4, **kw}
    r = client.post("/api/interviews", json=body, headers=auth)
    assert r.status_code == 200, r.text
    return r.json()


def answer(client, auth, iv_id, text):
    r = client.post(f"/api/interviews/{iv_id}/answer", json={"answer": text}, headers=auth)
    assert r.status_code == 200, r.text
    return r.json()


def test_first_question_is_medium_first_topic(client, auth):
    iv = start(client, auth)
    q = iv["current_question"]
    assert q["skill"] == "DSA" and q["difficulty"] == "Medium"
    assert "_reference" not in q and "reference" not in q          # never leaked to the browser


def test_poor_stays_on_skill_and_gets_easier(client, auth):
    iv = start(client, auth)
    res = answer(client, auth, iv["id"], "poor answer")
    assert res["next_question"]["skill"] == "DSA"
    assert res["next_question"]["difficulty"] == "Easy"


def test_good_rotates_topic_and_gets_harder(client, auth):
    iv = start(client, auth)
    res = answer(client, auth, iv["id"], "good answer")
    assert res["next_question"]["skill"] == "SQL"
    assert res["next_question"]["difficulty"] == "Hard"


def test_rotation_after_poor_continues_in_order():
    topics = ["DSA", "SQL", "OS"]
    assert svc._next_topic(topics, "DSA") == "SQL"
    assert svc._next_topic(topics, "OS") == "DSA"
    assert svc._next_topic(topics, "Java") == "DSA"


def test_interview_completes_with_report(client, auth):
    iv = start(client, auth)
    labels = ["excellent", "good", "average", "poor"]
    for i, l in enumerate(labels):
        res = answer(client, auth, iv["id"], f"{l} answer")
    assert res["completed"] is True
    state = client.get(f"/api/interviews/{iv['id']}", headers=auth).json()
    rep = state["report"]
    assert rep["questions_answered"] == 4
    assert rep["good_excellent"] == 2 and rep["average"] == 1 and rep["poor"] == 1
    assert rep["overall_score"] == round((90 + 75 + 50 + 25) / 4)
    assert client.post(f"/api/interviews/{iv['id']}/answer", json={"answer": "x"}, headers=auth).status_code == 400


def test_skill_scores_and_history_deletion(client, auth):
    iv = start(client, auth)
    answer(client, auth, iv["id"], "excellent")
    skills = {s["skill"]: s for s in client.get("/api/skills", headers=auth).json()["skills"]}
    assert skills["DSA"]["score"] == 90 and skills["DSA"]["attempts"] == 1
    assert client.delete("/api/history", headers=auth).json()["deleted"] == 1
    skills = {s["skill"]: s for s in client.get("/api/skills", headers=auth).json()["skills"]}
    assert skills["DSA"]["score"] is None
    assert client.get(f"/api/interviews/{iv['id']}", headers=auth).status_code == 404


def test_levels_constant_matches_bank():
    assert LEVELS == ["Easy", "Medium", "Hard"]
