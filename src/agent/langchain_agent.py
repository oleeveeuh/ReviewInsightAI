#!/usr/bin/env python3
"""
LangChain-based agent for ReviewInsight AI.

This provides an alternative implementation using the LangChain framework,
allowing for comparison with the custom ReAct-style agent.

Usage:
    from src.agent.langchain_agent import LangChainAgent

    agent = LangChainAgent()
    result = agent.analyze("Employee review text...")
"""

import json
from typing import Dict, Any, List
from pathlib import Path

# LangChain imports
try:
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import Tool, StructuredTool  # noqa: F401
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not installed. Install with: pip install langchain langchain-openai")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.vector_store import VectorStore
from agent.tools import AnalyzeSentimentTool, DetectAnomalyTool


class LangChainAgent:
    """
    LangChain-based agent for review analysis using ReAct pattern.

    This implementation uses LangChain's built-in agent framework
    to compare against the custom ReviewAgent implementation.
    """

    def __init__(self, vector_db_path: str = 'data/database/reviews.duckdb',
                 model: str = 'gpt-4o-mini'):
        """
        Initialize LangChain agent.

        Args:
            vector_db_path: Path to DuckDB database for vector search
            model: OpenAI model to use
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed")

        # Initialize components
        self.vector_store = VectorStore(vector_db_path)
        self.llm = ChatOpenAI(model=model, temperature=0.3)

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create tools
        self.tools = self._create_tools()

        # Create agent
        self.agent_executor = self._create_agent()

        print(f"✅ LangChain agent initialized with {len(self.tools)} tools")

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools from custom tool implementations."""

        # Vector search tool
        def vector_search(query: str, k: int = 5) -> str:
            """Find semantically similar reviews for context."""
            results = self.vector_store.search(query, k=k)

            output = f"Found {len(results)} similar reviews:\n"
            for i, r in enumerate(results[:k], 1):
                output += f"{i}. [{r['source']}] {r['similarity']:.2f}: {r['text'][:100]}...\n"

            return output

        # Sentiment analysis tool
        def analyze_sentiment(review_text: str) -> str:
            """Analyze sentiment, themes, and retention risk."""
            # Reuse existing tool logic
            tool = AnalyzeSentimentTool()
            result = tool.execute(review_text=review_text)

            return json.dumps({
                'sentiment': result.get('sentiment'),
                'themes': result.get('themes', []),
                'retention_risk': result.get('retention_risk')
            })

        # Anomaly detection tool
        def detect_anomaly(themes_json: str, sentiment: int) -> str:
            """Check if the analysis indicates anomalous patterns."""
            tool = DetectAnomalyTool()

            # Parse themes from JSON string
            try:
                themes = json.loads(themes_json)
            except Exception:
                themes = []

            result = tool.execute(themes=themes, sentiment=sentiment)

            return json.dumps({
                'is_anomalous': result.get('is_anomalous'),
                'anomaly_reason': result.get('anomaly_reason')
            })

        # Wrap as LangChain tools
        tools = [
            Tool(
                name="VectorSearch",
                func=vector_search,
                description="Search for semantically similar employee reviews to provide context. Input: query string. Output: similar reviews with similarity scores."
            ),
            Tool(
                name="AnalyzeSentiment",
                func=analyze_sentiment,
                description="Analyze employee sentiment, extract themes, and assess retention risk. Input: review text. Output: JSON with sentiment (1-5), themes, and risk level."
            ),
            Tool(
                name="DetectAnomaly",
                func=detect_anomaly,
                description="Detect unusual patterns in themes and sentiment. Input: themes as JSON array and sentiment score. Output: anomaly status and reasoning."
            )
        ]

        return tools

    def _create_agent(self) -> AgentExecutor:
        """Create LangChain ReAct agent executor."""

        # ReAct prompt template
        template = """You are an intelligent HR analyst that analyzes employee reviews.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

        return executor

    def analyze(self, review_text: str) -> Dict[str, Any]:
        """
        Analyze a review using LangChain agent.

        Args:
            review_text: Employee review to analyze

        Returns:
            Analysis results with sentiment, themes, risk, and reasoning
        """

        # Construct input question
        question = f"""Analyze this employee review and provide:
1. Sentiment score (1-5)
2. Key themes identified
3. Retention risk assessment (low/medium/high)

Review: {review_text}"""

        # Run agent
        try:
            result = self.agent_executor.invoke({"input": question})

            # Extract final answer
            final_answer = result.get("output", "")

            # Parse structured data from final answer
            # (This is a simplified version - you might want more robust parsing)
            analysis = {
                'sentiment': None,
                'themes': [],
                'retention_risk': None,
                'reasoning': final_answer,
                'agent_type': 'langchain'
            }

            # Try to extract JSON from the answer
            import re
            json_match = re.search(r'\{[^}]+\}', final_answer)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    analysis.update({
                        'sentiment': parsed.get('sentiment'),
                        'themes': parsed.get('themes', []),
                        'retention_risk': parsed.get('retention_risk')
                    })
                except Exception:
                    pass

            # Add intermediate steps for transparency
            analysis['intermediate_steps'] = result.get("intermediate_steps", [])

            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'agent_type': 'langchain',
                'review_text': review_text
            }

    def batch_analyze(self, reviews: List[str]) -> List[Dict]:
        """
        Analyze multiple reviews in batch.

        Args:
            reviews: List of review texts

        Returns:
            List of analysis results
        """
        results = []
        for i, review in enumerate(reviews):
            print(f"\nProcessing review {i+1}/{len(reviews)}...")
            result = self.analyze(review)
            results.append(result)

        return results


# Convenience function
def analyze_with_langchain(review_text: str,
                           vector_db_path: str = 'data/database/reviews.duckdb') -> Dict:
    """
    Analyze a review using LangChain agent.

    Args:
        review_text: Employee review to analyze
        vector_db_path: Path to vector database

    Returns:
        Analysis results
    """
    agent = LangChainAgent(vector_db_path=vector_db_path)
    return agent.analyze(review_text)


# Demo / testing
if __name__ == '__main__':
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not installed.")
        print("Install with: pip install langchain langchain-openai")
        exit(1)

    # Test agent
    test_review = """
    The benefits are decent and the pay is okay, but the mandatory overtime
    is exhausting. Management doesn't communicate schedule changes until the
    last minute. During peak season it's even worse - 60 hour weeks are common.
    """

    print("="*70)
    print("LANGCHAIN AGENT TEST")
    print("="*70)

    agent = LangChainAgent()
    result = agent.analyze(test_review)

    print("\nFinal Analysis:")
    print(json.dumps(result, indent=2))
