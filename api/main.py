#!/usr/bin/env python3
"""
FastAPI service for ReviewInsight AI.

Provides REST API endpoints:
- POST /analyze - Analyze a review (agentic)
- GET /memory/stats - Get memory statistics
- GET /tools - List available tools
- GET /health - Health check

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agent.memory import MemoryAwareAgent

# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class ReviewInput(BaseModel):
    """Request body for review analysis"""
    text: str = Field(..., description="Review text to analyze", min_length=1)
    review_id: Optional[str] = Field(None, description="Optional unique review identifier")

    class Config:
        schema_extra = {
            "example": {
                "text": "Great benefits but mandatory overtime is exhausting",
                "review_id": "test_123"
            }
        }


class AnalysisResponse(BaseModel):
    """Response from review analysis"""
    review_text: str
    plan: Dict[str, Any]
    final_analysis: Dict[str, Any]
    memory_stats: Optional[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "review_text": "Great benefits but...",
                "plan": {"reasoning": "...", "steps": []},
                "final_analysis": {
                    "sentiment": 2,
                    "themes": ["overtime", "pay_benefits"],
                    "retention_risk": "medium"
                },
                "memory_stats": {"total_analyses": 42},
                "tool_outputs": []
            }
        }


# =============================================================================
# FastAPI App
# =============================================================================

# Initialize FastAPI
app = FastAPI(
    title="ReviewInsight Agent API",
    description="Agentic employee review analysis system with tool orchestration and memory",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent (load once at startup)
agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    # vector_db would be loaded here in full implementation
    agent = MemoryAwareAgent(vector_db=None)
    print("✅ Agent initialized and ready")


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/analyze", response_model=AnalysisResponse,
          summary="Analyze Employee Review",
          description="""
Analyze an employee review using the agentic system.

The agent will:
1. Plan which tools to use (retrieval, sentiment analysis, anomaly detection)
2. Execute tools in intelligent sequence
3. Synthesize results with memory context
4. Return structured analysis with retention risk assessment
          """)
async def analyze_review(review: ReviewInput):
    """
    Analyze a review using the agentic system

    Args:
        review: ReviewInput with text and optional ID

    Returns:
        Complete analysis with tool outputs and memory stats
    """

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = agent.analyze(
            review_text=review.text,
            review_id=review.review_id
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/memory/stats",
         summary="Get Memory Statistics",
         description="Retrieve historical analysis statistics from agent memory")
async def get_memory_stats():
    """
    Get agent memory statistics

    Returns:
        Statistics including theme distribution, anomaly rate, etc.
    """

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        stats = agent.memory.get_summary_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@app.get("/tools",
         summary="List Available Tools",
         description="Get list of tools the agent can use")
async def list_tools():
    """
    List all available tools with descriptions

    Returns:
        List of tool specifications
    """

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    return {
        "tools": agent.tool_registry.list_tools(),
        "count": len(agent.tool_registry.tools)
    }


@app.get("/health",
         summary="Health Check",
         description="Check if API is running and agent is ready")
async def health_check():
    """
    Health check endpoint

    Returns:
        Status information
    """

    return {
        "status": "healthy" if agent else "initializing",
        "agent": "ready" if agent else "not ready",
        "api_version": "1.0.0"
    }


@app.get("/",
         summary="API Information",
         description="Get basic API information and links to documentation")
async def root():
    """API root with links to documentation"""
    return {
        "name": "ReviewInsight Agent API",
        "version": "1.0.0",
        "description": "Agentic employee review analysis",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


# =============================================================================
# Run App
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("REVIEWINSIGHT AI - API SERVICE")
    print("=" * 60)
    print()
    print("Starting FastAPI server...")
    print(f"Version: 1.0.0")
    print(f"Documentation: http://localhost:8000/docs")
    print(f"ReDoc: http://localhost:8000/redoc")
    print()
    print("Ready for requests")

    # Run with uvicorn for development
    # In production: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
