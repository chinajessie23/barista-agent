"""
FastAPI backend for Barista Agent.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import chat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_allowed_origins() -> list[str]:
    """Build CORS origins list."""
    origins = [
        "http://localhost:3000",  # Local Next.js dev
    ]
    # Add production frontend URL if configured
    frontend_url = os.getenv("FRONTEND_URL")
    if frontend_url:
        origins.append(frontend_url)
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Barista Agent API starting up")
    yield
    logger.info("Barista Agent API shutting down")


app = FastAPI(
    title="Barista Agent API",
    description="A coffee shop order-taking agent powered by LangGraph and Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration - use regex for Vercel preview deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=r"https://[\w-]+\.vercel\.app",  # Matches *.vercel.app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    finished: bool


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "barista-agent"}


@app.get("/health")
def health():
    """Health check for Railway."""
    return {"status": "healthy"}


# Note: Using sync endpoints since LangGraph's invoke() is synchronous.
# FastAPI runs these in a thread pool automatically.
# For high concurrency, consider async def + graph.ainvoke().


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Chat with the barista agent.

    - Send a message and optional session_id
    - Returns the agent's response and session_id for continuity
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        response, finished = chat(request.message, session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            finished=finished,
        )
    except Exception as e:
        logger.exception("Chat error for session %s", session_id)
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@app.post("/start", response_model=ChatResponse)
def start_conversation():
    """
    Start a new conversation with the barista.

    Returns the initial greeting.
    """
    session_id = str(uuid.uuid4())

    try:
        response, finished = chat("", session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            finished=finished,
        )
    except Exception as e:
        logger.exception("Failed to start conversation")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")
