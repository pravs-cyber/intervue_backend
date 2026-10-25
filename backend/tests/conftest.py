import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App on a fresh temp DB, no LLM, and a fake classifier whose label is driven by the answer text."""
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "AI_API_KEY", "")
    monkeypatch.setattr(config, "SECRET_KEY", "test-only-secret-key-not-for-production-0123456789")
    from ml import answer_classifier
    from services.constants import CATEGORIES, CATEGORY_SCORE

    def fake_predict(answer, question="", reference=None):
        label = next((c for c in CATEGORIES if c.lower() in answer.lower()), "Average")
        return {"label": label, "score": CATEGORY_SCORE[label], "probabilities": {c: float(c == label) for c in CATEGORIES},
                "model_agreement": 1.0, "model_variant": "fake"}

    monkeypatch.setattr(answer_classifier, "is_ready", lambda: True)
    monkeypatch.setattr(answer_classifier, "predict", fake_predict)
    from services import rate_limit
    rate_limit._hits.clear()
    from database.db import init_db
    init_db()
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)


@pytest.fixture
def auth(client):
    r = client.post("/api/auth/register", json={"name": "T", "email": "t@example.com", "password": "password123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
