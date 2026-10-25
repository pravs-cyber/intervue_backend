from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from database.db import get_conn
from ml import answer_classifier
from models.schemas import AnswerIn, EvaluateIn, InterviewIn
from services import interview_service as svc
from services import resume_service
from services.auth_service import get_current_user
from services.rate_limit import limited_user

import config

ALLOWED_RESUME = {".pdf": {"application/pdf"}, ".txt": {"text/plain"}}

router = APIRouter(prefix="/api", tags=["interviews"])


def _own_interview(conn, iv_id, user_id):
    iv = conn.execute("SELECT * FROM interviews WHERE id=? AND user_id=?", (iv_id, user_id)).fetchone()
    if not iv:
        raise HTTPException(404, "Interview not found")
    return iv


@router.post("/resume")
async def upload_resume(file: UploadFile = File(...), user=Depends(limited_user)):
    name = (file.filename or "").lower()
    ext = next((e for e in ALLOWED_RESUME if name.endswith(e)), None)
    if not ext or (file.content_type and file.content_type not in ALLOWED_RESUME[ext] | {"application/octet-stream"}):
        raise HTTPException(415, "Only .pdf and .txt resumes are accepted")
    data = await file.read(config.MAX_RESUME_BYTES + 1)
    if len(data) > config.MAX_RESUME_BYTES:
        raise HTTPException(413, "Resume file is larger than 2 MB")
    text = resume_service.extract_text(file.filename or "", data)
    if len(text.strip()) < 30:
        raise HTTPException(422, "Could not read text from this file. Try a text-based PDF or .txt file.")
    return {"text": text[:8000], "parsed": resume_service.parse_resume(text)}


@router.post("/interviews")
def start_interview(body: InterviewIn, user=Depends(limited_user)):
    if not answer_classifier.is_ready():
        raise HTTPException(503, "ML model not trained yet. Run ml/prepare_dataset.py and ml/train_model.py")
    if body.type == "resume" and len(body.resume_text.strip()) < 30:
        raise HTTPException(422, "Please upload or paste your resume first")
    with get_conn() as conn:
        try:
            iv_id = svc.create_interview(conn, user["id"], body.type, body.topics, body.num_questions, body.resume_text)
        except ValueError as e:
            raise HTTPException(422, str(e))
        iv = conn.execute("SELECT * FROM interviews WHERE id=?", (iv_id,)).fetchone()
        if not iv["current_question_id"]:
            raise HTTPException(500, "No questions available")
        return svc.interview_state(conn, iv)


@router.get("/interviews/{iv_id}")
def get_interview(iv_id: int, user=Depends(get_current_user)):
    with get_conn() as conn:
        return svc.interview_state(conn, _own_interview(conn, iv_id, user["id"]))


@router.post("/interviews/{iv_id}/answer")
def answer(iv_id: int, body: AnswerIn, user=Depends(limited_user)):
    with get_conn() as conn:
        iv = _own_interview(conn, iv_id, user["id"])
        if iv["status"] == "completed" or not iv["current_question_id"]:
            raise HTTPException(400, "Interview already completed")
        if not body.answer.strip():
            raise HTTPException(422, "Answer cannot be empty")
        return svc.submit_answer(conn, iv, user["id"], body.answer.strip())


@router.post("/evaluate-answer")
def evaluate_answer(body: EvaluateIn, user=Depends(limited_user)):
    """Stateless: run ML + feedback on one answer without saving anything. Auth + rate limit (may call a paid LLM)."""
    if not answer_classifier.is_ready():
        raise HTTPException(503, "ML model not trained yet")
    return svc.evaluate(body.question, body.answer, body.skill or "General", body.reference)


@router.delete("/interviews/{iv_id}")
def delete_interview(iv_id: int, user=Depends(get_current_user)):
    with get_conn() as conn:
        _own_interview(conn, iv_id, user["id"])
        svc.delete_interviews(conn, user["id"], iv_id)
    return {"deleted": 1}


@router.delete("/history")
def delete_history(user=Depends(get_current_user)):
    """Delete all of the user's interviews and answers; skill scores are rebuilt (to empty)."""
    with get_conn() as conn:
        return {"deleted": svc.delete_interviews(conn, user["id"])}
