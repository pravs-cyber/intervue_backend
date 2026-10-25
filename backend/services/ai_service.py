"""
LLM access (OpenAI-compatible chat-completions API), configured ONLY via .env:
    AI_API_KEY, AI_MODEL, (optional) AI_BASE_URL

The LLM is used for: question generation, follow-ups, detailed feedback.
It never decides the answer-quality label - that always comes from the ML model.
If no key is configured (or a call fails) we fall back to simple rule-based text so
the app still runs in development. Every response says which source produced it.
"""
import json
import os
import re
from typing import Optional

import httpx

from services.constants import DIFFICULTY, SKILL_NAMES

AI_API_KEY = lambda: os.getenv("AI_API_KEY", "").strip()   # read lazily so .env is respected
AI_MODEL = lambda: os.getenv("AI_MODEL", "gpt-4o-mini")
AI_BASE_URL = lambda: os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def enabled() -> bool:
    return bool(AI_API_KEY())


def _chat(system: str, user: str, max_tokens: int = 500) -> str:
    resp = httpx.post(
        f"{AI_BASE_URL()}/chat/completions",
        headers={"Authorization": f"Bearer {AI_API_KEY()}"},
        json={"model": AI_MODEL(), "max_tokens": max_tokens, "temperature": 0.4,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------- questions ----------
def generate_question(itype: str, skill: str, difficulty: int, asked: list[str], resume_summary: str = "") -> Optional[str]:
    if not enabled():
        return None
    kind = {"technical": f"a {DIFFICULTY[difficulty]} technical interview question about {SKILL_NAMES[skill]}",
            "hr": f"a {DIFFICULTY[difficulty]} HR/behavioural interview question",
            "resume": f"a {DIFFICULTY[difficulty]} interview question based on this candidate's resume:\n{resume_summary}"}[itype]
    try:
        text = _chat("You are a campus-placement interviewer. Reply with ONLY the question text, one question, no numbering.",
                     f"Ask {kind}.\nDo NOT repeat any of these already-asked questions:\n- " + "\n- ".join(asked[-8:]),
                     max_tokens=120)
        return text.strip('"') or None
    except Exception as e:
        print("[ai] question generation failed:", e)
        return None


def generate_followup(question: str, answer: str, skill: str) -> Optional[str]:
    if not enabled():
        return None
    try:
        text = _chat("You are an interviewer. Reply with ONLY one short, harder follow-up question.",
                     f"Topic: {SKILL_NAMES[skill]}\nQuestion: {question}\nCandidate answer: {answer}\n"
                     "The candidate answered very well. Ask one deeper follow-up question.", max_tokens=100)
        return text.strip('"') or None
    except Exception as e:
        print("[ai] follow-up failed:", e)
        return None


# ---------- feedback ----------
def _fallback_feedback(question: str, answer: str, skill: str, label: str) -> dict:
    words = len(answer.split())
    has_example = bool(re.search(r"for example|e\.g\.|such as|for instance|like a ", answer, re.I))
    well, improve = [], []
    if label in ("Excellent", "Good"):
        well.append("You identified the main concept correctly and stayed relevant to the question.")
    elif label == "Average":
        well.append("You touched on the core idea of the question.")
    else:
        well.append("You made an attempt - that is a starting point to build on.")
    if words >= 60:
        well.append("Your answer had good depth and structure.")
    if has_example:
        well.append("You supported your explanation with an example.")
    if words < 60:
        improve.append("Explain in more depth - aim for 4-6 sentences covering what, why and how.")
    if not has_example:
        improve.append("Add a practical example to make the answer concrete.")
    if label != "Excellent":
        improve.append("Mention trade-offs or compare with a related concept.")
    summary = {"Excellent": "Excellent answer - complete and well explained.",
               "Good": "Good answer - correct, with room to add more depth.",
               "Average": "Average answer - the basics are there but it needs more detail.",
               "Poor": "This answer needs work - revise the fundamentals of this topic and try again."}[label]
    return {"summary": summary, "went_well": well[:3], "improve": improve[:3],
            "recommended_topic": f"{SKILL_NAMES[skill]} fundamentals", "source": "fallback"}


def generate_feedback(question: str, answer: str, skill: str, label: str) -> dict:
    if not enabled():
        return _fallback_feedback(question, answer, skill, label)
    try:
        raw = _chat(
            "You are a supportive interview coach. Respond with ONLY valid JSON: "
            '{"summary": "1-2 sentences", "went_well": ["..",".."], "improve": ["..",".."], "recommended_topic": ".."}. '
            "Keep each bullet under 25 words. Be specific to the candidate's answer.",
            f"Skill: {SKILL_NAMES[skill]}\nQuestion: {question}\nCandidate answer: {answer}\n"
            f"An ML model rated this answer: {label}. Your feedback must be consistent with that rating.",
            max_tokens=400)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return {"summary": str(data.get("summary", "")), "went_well": list(data.get("went_well", []))[:4],
                "improve": list(data.get("improve", []))[:4],
                "recommended_topic": str(data.get("recommended_topic", SKILL_NAMES[skill])), "source": "llm"}
    except Exception as e:
        print("[ai] feedback failed, using fallback:", e)
        return _fallback_feedback(question, answer, skill, label)
