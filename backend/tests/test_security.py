def test_evaluate_requires_auth(client):
    r = client.post("/api/evaluate-answer", json={"question": "q", "answer": "a"})
    assert r.status_code == 401


def test_answer_length_limit(client, auth):
    r = client.post("/api/evaluate-answer", json={"question": "q", "answer": "x" * 5000}, headers=auth)
    assert r.status_code == 422


def test_rate_limit(client, auth, monkeypatch):
    import config
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 3)
    codes = [client.post("/api/evaluate-answer", json={"question": "q", "answer": "a"}, headers=auth).status_code
             for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_resume_type_and_size(client, auth):
    r = client.post("/api/resume", files={"file": ("cv.exe", b"MZ...", "application/octet-stream")}, headers=auth)
    assert r.status_code == 415
    big = b"a" * (2 * 1024 * 1024 + 10)
    r = client.post("/api/resume", files={"file": ("cv.txt", big, "text/plain")}, headers=auth)
    assert r.status_code == 413


def test_other_users_interview_is_hidden(client, auth):
    iv = client.post("/api/interviews", json={"type": "hr", "num_questions": 3}, headers=auth).json()
    other = client.post("/api/auth/register", json={"name": "O", "email": "o@example.com", "password": "password123"}).json()
    h = {"Authorization": f"Bearer {other['token']}"}
    assert client.get(f"/api/interviews/{iv['id']}", headers=h).status_code == 404
    assert client.delete(f"/api/interviews/{iv['id']}", headers=h).status_code == 404
