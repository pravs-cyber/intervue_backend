import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

APP_ENV = os.getenv("APP_ENV", "development")          # development | production
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# Comma-separated list, e.g. "https://intervue.vercel.app,https://intervue-git-main-you.vercel.app"
FRONTEND_ORIGINS = [o.strip() for o in os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",") if o.strip()]
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "intervue.db"))

_DEFAULT_SECRET = "dev-only-insecure-secret-key-change-me-0123456789"  # >=32 bytes for HS256
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY or SECRET_KEY == "change-me-to-a-long-random-string":
    if APP_ENV != "development":
        sys.exit("SECRET_KEY must be set to a long random value when APP_ENV is not 'development'.")
    SECRET_KEY = _DEFAULT_SECRET
    print("WARNING: using the development SECRET_KEY. Set SECRET_KEY in .env before deploying.", file=sys.stderr)

# limits
MAX_ANSWER_CHARS = 4000
MAX_RESUME_BYTES = 2 * 1024 * 1024
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))   # LLM-backed endpoints, per user
