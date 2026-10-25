from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from database.db import init_db
from ml import answer_classifier
from routes import auth, dashboard, interviews
from services import llm_service

app = FastAPI(title="Intervue API")
app.add_middleware(CORSMiddleware, allow_origins=config.FRONTEND_ORIGINS + ["http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])
init_db()
for r in (auth.router, interviews.router, dashboard.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    return {"status": "ok", "ml_model_ready": answer_classifier.is_ready(), "llm_configured": llm_service.is_configured()}
