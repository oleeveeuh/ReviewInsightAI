#!/usr/bin/env python3
"""
FastAPI service for ReviewInsight AI.

Endpoints:
- POST /analyze       - Analyze a review (plan-and-execute agent; requires
                        an OpenAI API key at request time)
- GET  /memory/stats  - Aggregates from the agent's local analysis log
- GET  /tools         - List registered agent tools
- GET  /health        - Health check (works without an API key)

The /analyze endpoint calls the OpenAI API. The service itself starts and
reports health without any API key; LLM-dependent endpoints return a clear
error when the key (or the openai package) is missing.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make project root importable regardless of how the server is launched
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.memory import MemoryAwareAgent

logger = logging.getLogger("reviewinsight.api")
logging.basicConfig(level=logging.INFO)

MAX_REVIEW_CHARS = 10_000

# =============================================================================
# Pydantic models
# =============================================================================


class ReviewInput(BaseModel):
    """Request body for review analysis."""
    text: str = Field(..., description="Review text to analyze", min_length=1,
                      max_length=MAX_REVIEW_CHARS)
    review_id: Optional[str] = Field(None, description="Optional unique review identifier")


class AnalysisResponse(BaseModel):
    """Response from review analysis."""
    review_text: str
    plan: Dict[str, Any]
    final_analysis: Dict[str, any]
    memory_stats: Optional[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]


# =============================================================================
# FastAPI app
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent (no API key needed for startup)."""
    global agent
    # Retrieval is optional; the tool degrades gracefully when unset.
    agent = MemoryAwareAgent(vector_db=None)
    logger.info("Agent initialized and ready")
    yield
    agent = None


app = FastAPI(
    title="ReviewInsight Agent API",
    description="Prototype API for LLM-assisted employee review analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS: configuration-driven, never wildcard-with-credentials ------------
# ALLOWED_ORIGINS is a comma-separated list, e.g. "http://localhost:8501".
# Unset/empty means no cross-origin browser access is allowed.

def parse_allowed_origins(raw: str) -> List[str]:
    """Parse the ALLOWED_ORIGINS env value into an origin list.

    Empty/whitespace input -> [] (no cross-origin browser access). Wildcards
    are rejected explicitly: '*' with credentials is never a safe default.
    """
    origins = [o.strip() for o in (raw or "").split(",") if o.strip()]
    return [o for o in origins if o != "*"]


_allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=bool(_allowed_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

agent: Optional[MemoryAwareAgent] = None


# =============================================================================
# Endpoints
# =============================================================================


@app.post("/analyze", response_model=AnalysisResponse,
          summary="Analyze Employee Review",
          description=("Analyze a review with the plan-and-execute agent. "
                       "Requires OPENAI_API_KEY (and the openai package) at "
                       "request time; other endpoints work without it."))
async def analyze_review(review: ReviewInput):
    """Analyze a review using the plan-and-execute agent."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        return agent.analyze(review_text=review.text, review_id=review.review_id)
    except (ImportError, RuntimeError) as e:
        # Missing openai package or API key: actionable message, no internals.
        logger.warning("LLM unavailable for /analyze: %s", type(e).__name__)
        raise HTTPException(
            status_code=503,
            detail=("LLM features are unavailable: install the optional LLM "
                    "dependencies and set OPENAI_API_KEY to use /analyze."),
        ) from e
    except Exception:
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. See server logs for details.",
        ) from None


@app.get("/memory/stats", summary="Get Memory Statistics",
         description="Aggregates from the agent's local analysis log (JSONL)")
async def get_memory_stats():
    """Aggregates from the agent's local analysis log."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        return agent.memory.get_summary_stats()
    except Exception:
        logger.exception("Failed to read agent memory")
        raise HTTPException(
            status_code=500,
            detail="Failed to compute memory statistics. See server logs.",
        ) from None


@app.get("/tools", summary="List Available Tools",
         description="List the tools the agent can plan with")
async def list_tools():
    """List all available tools."""
    if agent is None:
        raise HTTPException(503, detail="Agent not initialized")
    return {
        "tools": agent.tool_registry.list_tools(),
        "count": len(agent.tool_registry.tools),
    }


@app.get("/health", summary="Health Check",
         description="Service health (no API key required)")
async def health_check():
    """Health check endpoint (works without any API key)."""
    return {
        "status": "healthy" if agent else "initializing",
        "agent": "ready" if agent else "not ready",
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
        "api_version": "1.0.0",
    }


@app.get("/", summary="API Information")
async def root():
    """API root with documentation links."""
    return {
        "name": "ReviewInsight Agent API",
        "version": "1.0.0",
        "description": "Prototype API for LLM-assisted review analysis",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


# =============================================================================
# Run App
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("REVIEWINSIGHT AI - API SERVICE (prototype)")
    print("=" * 60)
    print("\nStarting FastAPI server on 127.0.0.1:8000")
    print("Documentation: http://127.0.0.1:8000/docs\n")

    # Dev server binds to loopback by default; use a real ASGI server and a
    # specific host if you ever expose this beyond localhost.
    uvicorn.run(app, host="127.0.0.1", port=8000)
