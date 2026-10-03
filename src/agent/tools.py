#!/usr/bin/env python3
"""
Tool abstraction layer for ReviewInsight AI agent.

Provides clean interface for LLM to call domain-specific functions.
Designed for agentic workflows where LLM decides which tools to use.

Usage:
    from src.agent.tools import ToolRegistry, retrieve_similar_reviews, analyze_sentiment
"""

from typing import Dict, List, Any, Optional, ClassVar
from pydantic import BaseModel, Field


# =============================================================================
# Base Tool Class
# =============================================================================

class Tool(BaseModel):
    """Base class for all tools."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config for Tool."""
        arbitrary_types_allowed = True


# =============================================================================
# Tool Implementations
# =============================================================================

class RetrieveSimilarTool(Tool):
    """Finds reviews similar to input using semantic search.

    For context: When an employee mentions "overnight shift struggles," the agent
    can retrieve similar reviews discussing night shifts to provide context.
    """

    name: str = Field(default="retrieve_similar_reviews")
    description: str = Field(
        default="Finds reviews similar to input using semantic search. "
                "Use for context when analyzing specific situations."
    )
    parameters: Dict[str, Any] = Field(default={
        "query_text": {
            "description": "Text query to find similar reviews",
            "type": "string",
            "required": True
        },
        "k": {
            "description": "Number of similar reviews to return (default 5)",
            "type": "integer",
            "default": 5
        }
    })

    def execute(self, query_text: str, k: int = 5, **kwargs) -> Dict:
        """Execute the tool - find similar reviews using semantic search."""

        # Use the in-memory vector store with real embeddings
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.agent.embeddings import get_vector_store

        store = get_vector_store()

        # Search for similar reviews
        results = store.search(query_text, k=k)

        return {
            "tool": self.name,
            "results": results,
            "count": len(results)
        }


class AnalyzeSentimentTool(Tool):
    """Analyzes sentiment 1-5 scale with context from similar reviews.

    For context: Provides richer analysis by considering what similar employees said.
    Should be called AFTER retrieving context reviews.
    """

    name: str = Field(default="analyze_sentiment")
    description: str = Field(
        default="Analyzes sentiment 1-5 scale. Use AFTER retrieving context reviews "
                "for more nuanced analysis."
    )
    parameters: Dict[str, Any] = Field(default={
        "review_text": {
            "description": "Review text to analyze",
            "type": "string",
            "required": True
        },
        "context_reviews": {
            "description": "Array of similar review texts for context (optional)",
            "type": "array",
            "required": False,
            "items": {
                "review_id": "string",
                "text": "string"
            }
        }
    })

    def execute(self, review_text: str, context_reviews: Optional[List[Dict]] = None, **kwargs) -> Dict:
        """Execute the tool - analyze sentiment."""

        # Import OpenAI (will be used in real execution)
        try:
            from openai import OpenAI
        except ImportError:
            return {"error": "OpenAI library not available. Set OPENAI_API_KEY."}

        # Get API key
        import os
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return {"error": "OPENAI_API_KEY not found in environment."}

        # Build messages with clear format specification
        system_prompt = """You are a sentiment analysis expert for employee reviews.

Analyze the review and return JSON with this exact structure:
{
  "sentiment": INTEGER,  // 1-5 scale where 1=very negative, 2=negative, 3=neutral, 4=positive, 5=very positive
  "themes": ["theme1", "theme2"],  // List of 2-5 main themes discussed
  "retention_risk": "low" or "medium" or "high"  // Risk of employee leaving based on review
}

Common themes to look for:
- overtime, workload, work_life_balance
- pay_benefits, compensation, salary
- management, leadership, communication
- safety, workplace_conditions
- career_growth, advancement, training
- culture, team_environment"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Review to analyze:\n{review_text[:3000]}"}
        ]

        # Add context if provided
        if context_reviews:
            context_str = "\n\nContext from similar reviews:\n"
            for i, cr in enumerate(context_reviews[:3], 1):
                context_str += f"{i}. {cr.get('text', cr)[:150]}...\n"
            messages[1]["content"] += context_str

        # Call OpenAI
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # Parse and normalize response
            import json
            result = json.loads(response.choices[0].message.content)

            # Normalize sentiment to integer 1-5
            sentiment = result.get("sentiment", 3)
            if isinstance(sentiment, dict):
                # Extract from nested structure
                if 'score' in sentiment:
                    sentiment = sentiment['score']
                elif 'overall' in sentiment:
                    overall = sentiment['overall'].lower()
                    sentiment_map = {
                        'very negative': 1, 'extremely negative': 1, 'negative': 2,
                        'neutral': 3, 'mixed': 3,
                        'positive': 4, 'very positive': 5, 'extremely positive': 5
                    }
                    sentiment = sentiment_map.get(overall, 3)
                else:
                    # Try to find any numeric value
                    for v in sentiment.values():
                        if isinstance(v, (int, float)):
                            sentiment = v
                            break
                    else:
                        sentiment = 3

            # Ensure sentiment is int 1-5
            try:
                sentiment = int(max(1, min(5, float(sentiment))))
            except (ValueError, TypeError):
                sentiment = 3

            # Normalize themes to list
            themes = result.get("themes", [])
            if not isinstance(themes, list):
                themes = []

            # Normalize retention_risk
            retention_risk = result.get("retention_risk", "medium")
            if retention_risk not in ["low", "medium", "high"]:
                retention_risk = "medium"

            return {
                "tool": self.name,
                "sentiment": sentiment,
                "themes": themes,
                "retention_risk": retention_risk,
                "context_used": len(context_reviews) if context_reviews else 0
            }

        except Exception as e:
            return {
                "tool": self.name,
                "error": str(e),
                "sentiment": 3,  # Default fallback
                "themes": [],
                "retention_risk": "medium"
            }


class DetectAnomalyTool(Tool):
    """Checks if themes are unusual compared to historical patterns.

    For quality control: Detect novel complaints that deviate from typical patterns.
    """

    name: str = Field(default="detect_anomaly")
    description: str = Field(
        default="Checks if themes are unusual vs historical data. "
                "Use for quality control on novel complaints."
    )
    parameters: Dict[str, Any] = Field(default={
        "themes": {
            "description": "Array of theme strings from detected review",
            "type": "array",
            "required": True
        },
        "sentiment": {
            "description": "Detected sentiment score (1-5)",
            "type": "integer",
            "required": True
        }
    })

    # Historical baseline frequencies (would be calculated from real data)
    THEME_BASELINE: ClassVar[Dict[str, float]] = {
        "overtime": 0.15,
        "pay_benefits": 0.35,
        "management": 0.28,
        "safety": 0.12,
        "career_growth": 0.08,
        "workload": 0.22,
        "work_life_balance": 0.18,
        "training": 0.06,
        "culture": 0.14,
        "other": 0.05
    }

    ANOMALY_THRESHOLD: ClassVar[float] = 0.05  # Theme below 5% is rare/anomaly

    def execute(self, themes: List[str], sentiment: int, **kwargs) -> Dict:
        """Execute the tool - detect anomalies."""

        rare_themes = []
        common_reasons = []

        for theme in themes:
            # Normalize theme
            theme_norm = theme.lower().strip()

            # Check frequency
            frequency = self.THEME_BASELINE.get(theme_norm, 0.01)  # Default 1%

            if frequency < self.ANOMALY_THRESHOLD:
                rare_themes.append(theme)
                common_reasons.append(f"'{theme}' is rare ({frequency:.1%} < {self.ANOMALY_THRESHOLD*100}%)")

        # Check for critical combinations
        is_anomalous = False
        critical_reasons = []

        if sentiment <= 2 and "safety" in [t.lower() for t in themes]:
            is_anomalous = True
            critical_reasons.append(f"Negative sentiment ({sentiment}) with safety concerns is unusual")

        if "pay_benefits" in themes and "management" in themes and sentiment <= 2:
            is_anomalous = True
            critical_reasons.append("Pay/benefits complaints with negative sentiment AND management issues")

        return {
            "tool": self.name,
            "is_anomalous": is_anomalous,
            "rare_themes": rare_themes if rare_themes else None,
            "critical_combinations": critical_reasons if critical_reasons else None,
            "anomaly_reason": "; ".join(common_reasons) if common_reasons else "No anomalies detected"
        }


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self, vector_db: Optional[Any] = None):
        """Initialize tool registry.

        Args:
            vector_db: Optional vector database for semantic search (not used yet)
        """
        self.tools: Dict[str, Tool] = {}
        self.vector_db = vector_db

        # Register core tools
        self.register(RetrieveSimilarTool())
        self.register(AnalyzeSentimentTool())
        self.register(DetectAnomalyTool())

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self.tools[tool.name] = tool
        print(f"  Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """List all available tools for LLM."""

        return [
            {
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for name, tool in self.tools.items()
        ]

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Parameters for the tool

        Returns:
            Tool execution result or error dict
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return {
                "tool": tool_name,
                "error": str(e),
                "trace": str(e)
            }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create registry
    registry = ToolRegistry()

    # List tools for LLM
    tools_info = registry.list_tools()

    print("=" * 60)
    print("AVAILABLE TOOLS")
    print("=" * 60)

    for tool_info in tools_info:
        print(f"\n{tool_info['name']}")
        print(f"  Description: {tool_info['description']}")

        print("  Parameters:")
        for param, details in tool_info['parameters'].items():
            required = " (required)" if details.get("required", False) else ""
            default_val = f" = {details.get('default', 'none')}" if 'default' in details else ""
            print(f"    {param}{required}{default_val}")
            if details.get("description"):
                print(f"      {details['description']}")

    print("\n" + "=" * 60)
    print("To use in agent code:")
    print("  registry = ToolRegistry()")
    print("  tool = registry.get_tool('tool_name')")
    print("  result = tool.execute(**parameters)")
    print("=" * 60)
