"""
FastAPI application — session‑based AskTheRepo API.

Endpoints:
    POST /start          – begin a new repo analysis session
    GET  /status/{sid}   – poll session readiness
    POST /chat/{sid}     – ask the agent a question
    POST /end/{sid}      – destroy a session (idempotent)
    GET  /health         – liveness probe
"""

import glob
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import requests as http_requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from session import RepoSession

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))

raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
parsed_origins = [
    o.strip().rstrip("/")
    for o in raw_origins.split(",")
    if o.strip()
]
allow_all = "*" in parsed_origins or not parsed_origins or parsed_origins == [""]

if allow_all:
    cors_origins = ["*"]
    cors_credentials = False
else:
    cors_origins = []
    for o in parsed_origins:
        cors_origins.append(o)
        cors_origins.append(o + "/")
    cors_credentials = True

MAX_SESSIONS = 10

# ---------------------------------------------------------------------------
# Session store (in‑process memory — single‑instance only)
# ---------------------------------------------------------------------------
sessions: dict[str, RepoSession] = {}
sessions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Janitor — daemon thread that reaps idle sessions
# ---------------------------------------------------------------------------
_janitor_stop = threading.Event()


def _janitor_loop():
    while not _janitor_stop.wait(timeout=60):
        now = time.time()
        to_destroy: list[str] = []
        with sessions_lock:
            for sid, sess in sessions.items():
                if now - sess.last_active > SESSION_TTL_SECONDS:
                    to_destroy.append(sid)
            for sid in to_destroy:
                sess = sessions.pop(sid, None)
                if sess:
                    sess.destroy()


# ---------------------------------------------------------------------------
# Startup / shutdown cleanup
# ---------------------------------------------------------------------------
def _cleanup_leftover_tempdirs():
    """Remove repo_* folders left behind by a previous crash."""
    pattern = os.path.join(tempfile.gettempdir(), "repo_*")
    for d in glob.glob(pattern):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _cleanup_leftover_tempdirs()
    janitor = threading.Thread(target=_janitor_loop, daemon=True)
    janitor.start()
    yield
    _janitor_stop.set()
    with sessions_lock:
        for sess in sessions.values():
            sess.destroy()
        sessions.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AskTheRepo", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
    )


# Allow CORS (supports * for public APIs or specific origins + Vercel deployment domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app" if not allow_all else None,
    allow_credentials=cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------
_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+"
)


def _validate_github_url(url: str) -> str | None:
    """Return an error message if the URL is not a valid public GitHub repo."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return "URL must use https."
    if parsed.hostname != "github.com":
        return "Only github.com repositories are supported."
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(path_parts) < 2:
        return "URL must include owner and repository name."
    if not _GITHUB_URL_RE.match(url):
        return "Invalid GitHub URL format."
    return None


def _extract_owner_repo(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    owner, repo = parts[0], parts[1]
    # Strip .git suffix
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def _check_github_repo(owner: str, repo: str) -> str | None:
    """
    Check the GitHub API for repo existence and size.
    Returns an error message on failure, or None on success.
    If the GitHub API itself is unreachable, returns None (fall through to clone).
    """
    try:
        resp = http_requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            timeout=10,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        if resp.status_code == 404:
            return f"Repository {owner}/{repo} not found or is private."
        if resp.status_code == 403:
            # Rate limited — fall through
            return None
        if resp.status_code == 200:
            data = resp.json()
            size_kb = data.get("size", 0)
            if size_kb > 50_000:
                return (
                    f"Repository is too large ({size_kb // 1000} MB). "
                    "Maximum supported size is 50 MB."
                )
        return None
    except Exception:
        return None  # API unreachable — let clone decide


# ---------------------------------------------------------------------------
# Background task: build session
# ---------------------------------------------------------------------------
def _build_session(session: RepoSession, owner: str, repo: str):
    """Run in BackgroundTasks — clone, index, build agent."""
    # Pre‑check via GitHub API
    err = _check_github_repo(owner, repo)
    if err:
        session.status = "error"
        session.error = err
        return
    session.build()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    for p in [
        os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"),
        os.path.join(os.path.dirname(__file__), "frontend", "index.html"),
    ]:
        if os.path.exists(p):
            return FileResponse(p)
    return {
        "status": "ok",
        "service": "AskTheRepo API",
        "message": "Backend is running. Connect your frontend or visit /health."
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/start")
@limiter.limit("5/hour")
async def start_session(body: StartRequest, request: Request,
                        background_tasks: BackgroundTasks):
    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"detail": "GROQ_API_KEY is missing in backend environment variables. Please set GROQ_API_KEY in your deployment environment."},
        )

    # Validate URL
    err = _validate_github_url(body.repo_url)
    if err:
        return JSONResponse(status_code=400, content={"detail": err})

    # Capacity check
    with sessions_lock:
        if len(sessions) >= MAX_SESSIONS:
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is at capacity. Try again later."},
            )

    owner, repo = _extract_owner_repo(body.repo_url)

    session = RepoSession(
        repo_url=body.repo_url,
        groq_api_key=GROQ_API_KEY,
        groq_model=GROQ_MODEL,
    )

    with sessions_lock:
        sessions[session.id] = session

    background_tasks.add_task(_build_session, session, owner, repo)

    return {"session_id": session.id}


@app.get("/status/{sid}")
async def session_status(sid: str):
    with sessions_lock:
        session = sessions.get(sid)
    if not session:
        return JSONResponse(status_code=404, content={"detail": "Session not found."})
    session.last_active = time.time()
    return {"status": session.status, "error": session.error}


@app.post("/chat/{sid}")
async def chat(sid: str, body: ChatRequest):
    with sessions_lock:
        session = sessions.get(sid)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"detail": "Session expired or not found. Please start a new session."},
        )
    if session.status != "ready":
        return JSONResponse(
            status_code=409,
            content={"detail": f"Session is not ready yet (status: {session.status})."},
        )
    if len(body.question) > 1000:
        return JSONResponse(
            status_code=400,
            content={"detail": "Question too long (max 1000 characters)."},
        )

    try:
        answer = session.ask(body.question)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Agent error: {str(exc)[:200]}"},
        )
    return {"answer": answer}


@app.post("/end/{sid}")
async def end_session(sid: str):
    with sessions_lock:
        session = sessions.pop(sid, None)
    if session:
        session.destroy()
    return {"status": "ok"}
