import hashlib
import hmac
import os
import time

import jwt
from fastapi import Header, HTTPException

import config
from database.db import get_conn


def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()
    return f"{salt}${h}"


def verify_password(password: str, stored: str) -> bool:
    salt, h = stored.split("$")
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()
    return hmac.compare_digest(h, check)


def create_token(user_id: int) -> str:
    return jwt.encode({"sub": str(user_id), "exp": int(time.time()) + 7 * 24 * 3600}, config.SECRET_KEY, algorithm="HS256")


def get_current_user(authorization: str = Header(default="")):
    """FastAPI dependency: reads 'Authorization: Bearer <token>'."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    try:
        payload = jwt.decode(authorization[7:], config.SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
    with get_conn() as conn:
        row = conn.execute("SELECT id, name, email, target_role FROM users WHERE id=?", (int(payload["sub"]),)).fetchone()
    if not row:
        raise HTTPException(401, "User not found")
    return dict(row)
