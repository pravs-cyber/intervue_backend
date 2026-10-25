import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from database.db import get_conn
from models.schemas import LoginIn, ProfileIn, RegisterIn
from services.auth_service import create_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api", tags=["auth"])


@router.post("/auth/register")
def register(body: RegisterIn):
    with get_conn() as conn:
        try:
            cur = conn.execute("INSERT INTO users (name, email, password_hash, target_role) VALUES (?,?,?,?)",
                               (body.name.strip(), body.email.lower(), hash_password(body.password), body.target_role))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Email already registered")
        uid = cur.lastrowid
    return {"token": create_token(uid), "user": {"id": uid, "name": body.name, "email": body.email.lower(), "target_role": body.target_role}}


@router.post("/auth/login")
def login(body: LoginIn):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (body.email.lower(),)).fetchone()
    if not row or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "Incorrect email or password")
    return {"token": create_token(row["id"]),
            "user": {"id": row["id"], "name": row["name"], "email": row["email"], "target_role": row["target_role"]}}


@router.get("/me")
def me(user=Depends(get_current_user)):
    return user


@router.put("/me")
def update_me(body: ProfileIn, user=Depends(get_current_user)):
    with get_conn() as conn:
        conn.execute("UPDATE users SET name=?, target_role=? WHERE id=?", (body.name.strip(), body.target_role, user["id"]))
    return {**user, "name": body.name.strip(), "target_role": body.target_role}
