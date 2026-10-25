from statistics import mean

from fastapi import APIRouter, Depends

from database.db import get_conn
from services.constants import ALL_SKILLS, skill_name, skill_status
from services.auth_service import get_current_user

router = APIRouter(prefix="/api", tags=["dashboard"])


def _skills(conn, user_id):
    rows = {r["skill"]: r for r in conn.execute("SELECT * FROM skill_scores WHERE user_id=?", (user_id,))}
    out = []
    for s in ALL_SKILLS:
        r = rows.get(s)
        out.append({"skill": s, "name": skill_name(s), "score": round(r["score"]) if r else None,
                    "attempts": r["attempts"] if r else 0, "status": skill_status(r["score"]) if r else "Not attempted"})
    return out


def _history(conn, user_id):
    rows = conn.execute(
        """SELECT i.id, i.type, i.score, i.status, i.created_at, i.total_questions,
                  (SELECT COUNT(*) FROM answers a WHERE a.interview_id=i.id) AS answered
           FROM interviews i WHERE i.user_id=? ORDER BY i.id DESC""", (user_id,)).fetchall()
    return [{**dict(r), "score": round(r["score"]) if r["score"] is not None else None} for r in rows]


@router.get("/skills")
def skills(user=Depends(get_current_user)):
    with get_conn() as conn:
        items = _skills(conn, user["id"])
    attempted = sorted([s for s in items if s["score"] is not None], key=lambda s: s["score"])
    return {"skills": items, "gaps": attempted[:3]}


@router.get("/history")
def history(user=Depends(get_current_user)):
    with get_conn() as conn:
        return {"interviews": _history(conn, user["id"])}


@router.get("/dashboard")
def dashboard(user=Depends(get_current_user)):
    with get_conn() as conn:
        items = _skills(conn, user["id"])
        hist = _history(conn, user["id"])
    done = [h for h in hist if h["status"] == "completed"]
    attempted = [s for s in items if s["score"] is not None]
    return {
        "total_interviews": len(done),
        "average_score": round(mean(h["score"] for h in done)) if done else None,
        "strongest_skill": max(attempted, key=lambda s: s["score"])["name"] if attempted else None,
        "weakest_skill": min(attempted, key=lambda s: s["score"])["name"] if attempted else None,
        "skills": items,
        "score_history": [{"id": h["id"], "score": h["score"], "created_at": h["created_at"]} for h in reversed(done)][-10:],
        "recent": hist[:5],
    }
