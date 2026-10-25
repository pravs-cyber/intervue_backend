# Deploying the Intervue backend to Render

## 0. Files in this package
    backend/                 your backend code (cleaned: no __pycache__, no unused ml/ artifacts)
    backend/requirements.txt  NEW - slim, runtime-only deps (see the comment inside it)
    backend/main.py          MODIFIED - CORS now reads a list of allowed origins
    backend/config.py        MODIFIED - FRONTEND_ORIGIN is parsed as a comma-separated list
    data/sources/            your approved bank_references.csv (read at startup) - must sit
                              next to backend/, NOT inside it (same layout as your real repo)
    render.yaml               a Render Blueprint (optional - see step 2)
    .gitignore                a corrected root .gitignore (see step 1 - important)

## 1. Fix your git repo BEFORE pushing (do this once)
Your current root .gitignore has `backend/ml/*.pkl`, which blanket-excludes every .pkl file under
backend/ml/ - including `answer_model.pkl`, the file the running app actually loads. If you push
as-is, Render will clone a repo with no trained model and the API will report `ml_model_ready: false`.

Replace your repo's root `.gitignore` with the one in this package, then track the model file with
Git LFS (GitHub's normal file-size limit is 100 MB; answer_model.pkl is ~137 MB):

    git lfs install
    git lfs track "backend/ml/answer_model.pkl"
    git add .gitattributes backend/ml/answer_model.pkl .gitignore
    git commit -m "Track the trained model with Git LFS; fix .gitignore"
    git push

Render supports Git LFS automatically - no extra Render config needed once the repo has it set up.

**Alternative, if you'd rather not use LFS:** upload `answer_model.pkl` as an asset on a GitHub
Release in your own repo (Releases -> Draft a new release -> attach the file), keep it out of git
entirely, and change the Render build command to fetch it first:

    pip install -r requirements.txt && curl -L -o ml/answer_model.pkl "<your release asset URL>"

(update `buildCommand` in render.yaml, or the dashboard field, to the line above).

## 2. Create the service
**Option A - Blueprint (uses render.yaml):** Render dashboard -> New -> Blueprint -> select your repo.
**Option B - manual:** New -> Web Service -> select your repo, then set:
  - Root Directory: `backend`
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Either way, Render still clones your WHOLE repo (Root Directory only changes where the build/start
commands run from), so `data/sources/` next to `backend/` is picked up correctly.

## 3. Set environment variables (Render dashboard -> your service -> Environment)
  APP_ENV=production
  SECRET_KEY=<a long random string - the app refuses to start without one outside development>
  AI_API_KEY=<your Groq or other OpenAI-compatible key>
  AI_BASE_URL=https://api.groq.com/openai/v1        (or whatever provider you use)
  AI_MODEL=openai/gpt-oss-20b                        (or your chosen model)
  FRONTEND_ORIGIN=https://<your-app>.vercel.app      (set this AFTER deploying the frontend; comma-
                                                       separate multiple URLs if you have more than one)

## 4. Know the SQLite caveat
Render's free tier has no persistent disk: `intervue.db` is recreated empty every time the service
redeploys, restarts, or spins down from inactivity (15 minutes). Fine for a demo; if you need data
to survive, use a paid instance with a persistent disk, or move to Render's free Postgres instead.

## 5. Verify
Once deployed, open `https://<your-service>.onrender.com/api/health` - you should see
`{"status":"ok","ml_model_ready":true,...}`. If `ml_model_ready` is false, the model file didn't
make it into the deploy - re-check step 1.
