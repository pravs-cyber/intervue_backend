from fastapi import APIRouter, Depends

from database.db import get_db
from services.auth import current_user
from services.constants import SKILLS, SKILL_NAMES, skill_status
from services.interview_service import build_report

router = APIRouter(prefix="/api")


def _skills(db, user_id):
    rows = {r["skill"]: r for r in db.execute("SELECT * FROM skill_scores WHERE user_id=?", (user_id,))}
    out = []
    for s in SKILLS:
        r = rows.get(s)
        out.append({"skill": s, "name": SKILL_NAMES[s], "score": round(r["score"]) if r else None,
                    "attempts": r["attempts"] if r else 0, "status": skill_status(r["score"]) if r else None})
    return out


def _gaps(skills):
    practiced = sorted([s for s in skills if s["score"] is not None], key=lambda s: s["score"])
    return practiced[:3]


def _interview_rows(db, user_id, limit=None):
    q = "SELECT * FROM interviews WHERE user_id=? AND status='completed' ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
    out = []
    for r in db.execute(q, (user_id,)).fetchall():
        rep = build_report(db, r["id"])
        out.append({"id": r["id"], "type": r["type"], "score": round(r["score"]), "created_at": r["created_at"],
                    "counts": rep["counts"]})
    return out


@router.get("/skills")
def skills(user=Depends(current_user), db=Depends(get_db)):
    sk = _skills(db, user["id"])
    return {"skills": sk, "gaps": _gaps(sk)}


@router.get("/history")
def history(user=Depends(current_user), db=Depends(get_db)):
    return {"interviews": _interview_rows(db, user["id"])}


@router.get("/dashboard")
def dashboard(user=Depends(current_user), db=Depends(get_db)):
    sk = _skills(db, user["id"])
    practiced = [s for s in sk if s["score"] is not None]
    rows = _interview_rows(db, user["id"])
    return {
        "total_interviews": len(rows),
        "average_score": round(sum(r["score"] for r in rows) / len(rows)) if rows else None,
        "strongest": max(practiced, key=lambda s: s["score"]) if practiced else None,
        "weakest": min(practiced, key=lambda s: s["score"]) if practiced else None,
        "skills": sk, "gaps": _gaps(sk),
        "score_history": [{"id": r["id"], "score": r["score"], "created_at": r["created_at"]} for r in reversed(rows[:10])],
        "recent": rows[:5],
    }
