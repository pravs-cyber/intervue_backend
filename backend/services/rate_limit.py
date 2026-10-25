"""Tiny in-memory per-user rate limiter (sliding 60 s window). Fine for one process; use Redis if you scale out."""
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException

import config
from services.auth_service import get_current_user

_hits: dict[int, deque] = defaultdict(deque)


def check(user_id: int, limit: int | None = None):
    limit = limit or config.RATE_LIMIT_PER_MINUTE
    now, q = time.monotonic(), _hits[user_id]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests - please wait a minute and try again.")
    q.append(now)


def limited_user(user=Depends(get_current_user)):
    """Dependency: authenticated user + rate limit."""
    check(user["id"])
    return user
