"""
Core interview logic: adaptive question selection, scoring, skill updates, report.
Everything here is plain rule-based Python (no RL, no extra ML).
"""
import json
import random
from statistics import mean

from ml import answer_classifier
from services import llm_service, resume_service
from services.constants import (ALL_SKILLS, DIFFICULTY_DELTA, LEVELS, SKILL_NAMES, TECH_SKILLS,  # noqa: F401
                                skill_status)


def _clamp(i: int) -> int:
    return max(0, min(2, i))


def _insert_question(conn, text, skill, difficulty_idx, qtype, source):
    # LLM-written questions also get an LLM-written reference answer so the reference-aware model can be used.
    # Note: this means the LLM indirectly shapes the features (not the decision) - see README.
    reference = llm_service.generate_reference(text, skill) if source == "llm" else None
    cur = conn.execute(
        "INSERT INTO questions (question, skill, difficulty, type, source, reference) VALUES (?,?,?,?,?,?)",
        (text, skill, LEVELS[difficulty_idx], qtype, source, reference),
    )
    return cur.lastrowid


def question_dict(conn, qid):
    r = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
    return {"id": r["id"], "question": r["question"], "skill": r["skill"],
            "skill_name": SKILL_NAMES.get(r["skill"], r["skill"]), "difficulty": r["difficulty"], "type": r["type"],
            "_reference": r["reference"]}


def public_question(q):
    """Never send the reference answer to the browser."""
    return {k: v for k, v in q.items() if not k.startswith("_")} if q else q


def _asked_ids(conn, iv_id):
    return {r["question_id"] for r in conn.execute("SELECT question_id FROM answers WHERE interview_id=?", (iv_id,))}


def _bank_pick(conn, qtype, skills, diff, exclude_ids):
    """Pick an unasked bank question: right skill, closest difficulty."""
    for skill in skills:
        for d in sorted(range(3), key=lambda x: abs(x - diff)):
            rows = [r for r in conn.execute(
                "SELECT id FROM questions WHERE source='bank' AND type=? AND skill=? AND difficulty=?",
                (qtype, skill, LEVELS[d])) if r["id"] not in exclude_ids]
            if rows:
                return random.choice(rows)["id"]
    return None


# ------------------------------------------------------------------ start
def create_interview(conn, user_id, itype, topics, total, resume_text):
    topics = [t for t in topics if t in TECH_SKILLS]
    if itype == "technical" and not topics:
        topics = list(TECH_SKILLS)
    if itype == "hr":
        topics = ["Communication"]

    queue = []
    if itype == "resume":
        parsed = resume_service.parse_resume(resume_text)
        items = llm_service.generate_resume_questions(parsed, total) or resume_service.fallback_questions(parsed, total)
        if not items:
            raise ValueError("Could not build questions from this resume. Paste more resume text.")
        for it in items:
            skill = it.get("skill") if it.get("skill") in ALL_SKILLS else "Communication"
            queue.append(_insert_question(conn, it["question"], skill, 1, "resume", "llm" if llm_service.is_configured() else "template"))
        total = len(queue)

    cur = conn.execute(
        "INSERT INTO interviews (user_id, type, total_questions, topics, question_queue, resume_text) VALUES (?,?,?,?,?,?)",
        (user_id, itype, total, json.dumps(topics), json.dumps(queue), resume_text))
    iv_id = cur.lastrowid
    iv = conn.execute("SELECT * FROM interviews WHERE id=?", (iv_id,)).fetchone()
    qid = _next_question(conn, iv, None, None, None)
    conn.execute("UPDATE interviews SET current_question_id=? WHERE id=?", (qid, iv_id))
    return iv_id


def _next_topic(topics, current):
    """Rotate to the topic after `current` (Poor answers stay on the same topic, see below)."""
    if current not in topics:
        return topics[0]
    return topics[(topics.index(current) + 1) % len(topics)]


# ------------------------------------------------------- adaptive selection
def _next_question(conn, iv, last_q, last_label, last_answer):
    """Returns the next question id (or None if nothing is left)."""
    asked = _asked_ids(conn, iv["id"])
    if last_q is not None:
        asked.add(last_q["id"])
    itype = iv["type"]

    # resume interviews: walk the pre-built queue, adding an LLM follow-up after an Excellent answer
    if itype == "resume":
        if last_label == "Excellent" and last_q and llm_service.is_configured():
            fu = llm_service.generate_followup(last_q["question"], last_answer, last_q["skill"])
            if fu:
                return _insert_question(conn, fu, last_q["skill"], 2, "resume", "llm")
        for qid in json.loads(iv["question_queue"]):
            if qid not in asked:
                return qid
        return None

    topics = json.loads(iv["topics"])
    if last_q is None:                                  # first question: medium, first topic
        skill, diff = topics[0], 1
    else:
        last_idx = LEVELS.index(last_q["difficulty"])
        diff = _clamp(last_idx + DIFFICULTY_DELTA[last_label])
        # Poor -> stay on the same skill (easier); otherwise rotate to the next topic
        skill = last_q["skill"] if last_label == "Poor" else _next_topic(topics, last_q["skill"])

    # Excellent -> LLM follow-up (harder, same skill) if available
    if last_label == "Excellent" and itype == "technical" and llm_service.is_configured():
        fu = llm_service.generate_followup(last_q["question"], last_answer, last_q["skill"])
        if fu:
            return _insert_question(conn, fu, last_q["skill"], diff, "technical", "llm")

    if itype == "technical" and llm_service.is_configured():
        texts = [r["question"] for r in conn.execute(
            "SELECT q.question FROM answers a JOIN questions q ON q.id=a.question_id WHERE a.interview_id=?", (iv["id"],))]
        q_text = llm_service.generate_question(SKILL_NAMES.get(skill, skill), LEVELS[diff], texts)
        if q_text:
            return _insert_question(conn, q_text, skill, diff, "technical", "llm")

    qtype = "hr" if itype == "hr" else "technical"
    others = [skill] + [t for t in topics if t != skill]
    return _bank_pick(conn, qtype, others, diff, asked)


# ----------------------------------------------------------------- answer
def evaluate(question: str, answer: str, skill: str, reference: str | None = None) -> dict:
    """ML prediction (the classifier decides) + LLM/template feedback (explains)."""
    pred = answer_classifier.predict(answer, question, reference)
    fb = llm_service.generate_feedback(question, answer, pred["label"], SKILL_NAMES.get(skill, skill), reference)
    return {"ml_prediction": pred["label"], "score": pred["score"], "probabilities": pred["probabilities"],
            "model_agreement": pred["model_agreement"], "model_variant": pred["model_variant"], "feedback": fb}


def _update_skill(conn, user_id, skill, score):
    row = conn.execute("SELECT score, attempts FROM skill_scores WHERE user_id=? AND skill=?", (user_id, skill)).fetchone()
    if row is None:
        conn.execute("INSERT INTO skill_scores (user_id, skill, score, attempts) VALUES (?,?,?,1)", (user_id, skill, score))
    else:
        new = (row["score"] * row["attempts"] + score) / (row["attempts"] + 1)
        conn.execute("UPDATE skill_scores SET score=?, attempts=? WHERE user_id=? AND skill=?",
                     (new, row["attempts"] + 1, user_id, skill))


def submit_answer(conn, iv, user_id, answer_text):
    q = question_dict(conn, iv["current_question_id"])
    res = evaluate(q["question"], answer_text, q["skill"], q["_reference"])
    conn.execute(
        "INSERT INTO answers (interview_id, question_id, answer_text, ml_prediction, score, feedback) VALUES (?,?,?,?,?,?)",
        (iv["id"], q["id"], answer_text, res["ml_prediction"], res["score"], json.dumps(res["feedback"])))
    _update_skill(conn, user_id, q["skill"], res["score"])

    answered = conn.execute("SELECT COUNT(*) FROM answers WHERE interview_id=?", (iv["id"],)).fetchone()[0]
    next_q = None
    if answered < iv["total_questions"]:
        nid = _next_question(conn, iv, q, res["ml_prediction"], answer_text)
        if nid:
            next_q = question_dict(conn, nid)

    if next_q is None:                                  # finished
        overall = conn.execute("SELECT AVG(score) FROM answers WHERE interview_id=?", (iv["id"],)).fetchone()[0]
        conn.execute("UPDATE interviews SET status='completed', score=?, current_question_id=NULL WHERE id=?", (overall, iv["id"]))
    else:
        conn.execute("UPDATE interviews SET current_question_id=? WHERE id=?", (next_q["id"], iv["id"]))

    return {"question": public_question(q), "ml_prediction": res["ml_prediction"], "score": res["score"],
            "probabilities": res["probabilities"], "model_agreement": res["model_agreement"],
            "feedback": res["feedback"], "next_question": public_question(next_q), "completed": next_q is None, "answered": answered,
            "total": iv["total_questions"] if next_q else answered}


# ----------------------------------------------------------------- report
def build_report(conn, iv_id):
    rows = conn.execute(
        "SELECT a.ml_prediction, a.score, q.skill FROM answers a JOIN questions q ON q.id=a.question_id WHERE a.interview_id=?",
        (iv_id,)).fetchall()
    if not rows:
        return None
    by_skill = {}
    for r in rows:
        by_skill.setdefault(r["skill"], []).append(r["score"])
    skill_avgs = {s: round(mean(v)) for s, v in by_skill.items()}
    tech = [r["score"] for r in rows if r["skill"] != "Communication"]
    comm = [r["score"] for r in rows if r["skill"] == "Communication"]
    labels = [r["ml_prediction"] for r in rows]
    gaps = sorted(skill_avgs.items(), key=lambda kv: kv[1])[:3]
    return {
        "overall_score": round(mean(r["score"] for r in rows)),
        "technical_score": round(mean(tech)) if tech else None,
        "communication_score": round(mean(comm)) if comm else None,
        "skills": [{"skill": s, "name": SKILL_NAMES.get(s, s), "score": v, "status": skill_status(v)} for s, v in skill_avgs.items()],
        "strong_areas": [SKILL_NAMES.get(s, s) for s, v in skill_avgs.items() if v > 75],
        "needs_improvement": [SKILL_NAMES.get(s, s) for s, v in skill_avgs.items() if v < 60],
        "skill_gaps": [{"skill": s, "name": SKILL_NAMES.get(s, s), "score": v, "status": skill_status(v)} for s, v in gaps],
        "questions_answered": len(rows),
        "good_excellent": sum(l in ("Good", "Excellent") for l in labels),
        "average": labels.count("Average"),
        "poor": labels.count("Poor"),
    }


def interview_state(conn, iv):
    answers = conn.execute(
        """SELECT a.id, a.answer_text, a.ml_prediction, a.score, a.feedback, q.question, q.skill, q.difficulty
           FROM answers a JOIN questions q ON q.id=a.question_id WHERE a.interview_id=? ORDER BY a.id""", (iv["id"],)).fetchall()
    return {
        "id": iv["id"], "type": iv["type"], "status": iv["status"], "score": iv["score"], "created_at": iv["created_at"],
        "total_questions": iv["total_questions"], "answered": len(answers),
        "current_question": public_question(question_dict(conn, iv["current_question_id"])) if iv["current_question_id"] else None,
        "answers": [{"question": a["question"], "skill": SKILL_NAMES.get(a["skill"], a["skill"]), "difficulty": a["difficulty"],
                     "answer": a["answer_text"], "ml_prediction": a["ml_prediction"], "score": a["score"],
                     "feedback": json.loads(a["feedback"] or "{}")} for a in answers],
        "report": build_report(conn, iv["id"]) if iv["status"] == "completed" else None,
    }


# ----------------------------------------------------------------- deletion
def recompute_skills(conn, user_id):
    """Rebuild skill_scores from the answers that still exist (after a deletion)."""
    conn.execute("DELETE FROM skill_scores WHERE user_id=?", (user_id,))
    conn.execute("""INSERT INTO skill_scores (user_id, skill, score, attempts)
                    SELECT i.user_id, q.skill, AVG(a.score), COUNT(*) FROM answers a
                    JOIN interviews i ON i.id=a.interview_id JOIN questions q ON q.id=a.question_id
                    WHERE i.user_id=? GROUP BY q.skill""", (user_id,))


def delete_interviews(conn, user_id, iv_id=None):
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM interviews WHERE user_id=?" + (" AND id=?" if iv_id else ""),
        (user_id, iv_id) if iv_id else (user_id,))]
    for i in ids:
        conn.execute("DELETE FROM answers WHERE interview_id=?", (i,))
        conn.execute("DELETE FROM interviews WHERE id=?", (i,))
    recompute_skills(conn, user_id)
    return len(ids)
