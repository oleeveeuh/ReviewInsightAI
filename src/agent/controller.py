#!/usr/bin/env python3
"""
Agent controller for ReviewInsight AI - ReAct Pattern Implementation.

Agentic system that uses LLM to:
1. Plan which tools to use (Reasoning)
2. Execute tools in sequence (Acting)
3. Synthesize results (Output)

This enables more sophisticated analysis than simple one-shot labeling.

Usage:
    from src.agent.controller import ReviewAgent, analyze_review_agentic

    agent = ReviewAgent()
    result = agent.analyze("Employee complains about mandatory overtime...")
"""

from typing import Dict, Any, List, Optional
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agent.tools import ToolRegistry

load_dotenv()


def _get_client():
    """Get or create OpenAI client (lazy initialization)."""
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


AGENT_SYSTEM_PROMPT = """You are an intelligent agent that analyzes employee reviews.

You have access to these tools:
{tools_description}

Your task: Analyze the given review by deciding which tools to use and in what order.

RECOMMENDED WORKFLOW:
1. First: retrieve_similar_reviews (get context)
2. Then: analyze_sentiment (with context from step 1)
3. Optionally: detect_anomaly (if themes/sentiment warrant it)

However, YOU decide based on the review content. Don't use tools you don't need.

Respond with a JSON plan:
{{
  "reasoning": "Brief explanation of your strategy",
  "steps": [
    {{"tool": "tool_name", "params": {{"param": "value"}}}},
    {{"tool": "tool_name", "params": {{"param": "value"}}}}
  ]
}}

Be strategic and efficient."""


class ReviewAgent:
    """Agentic review analyzer with tool orchestration"""

    def __init__(self, vector_db=None, model="gpt-4o-mini"):
        """
        Initialize agent

        Args:
            vector_db: Optional vector database for retrieval
            model: OpenAI model for planning
        """
        self.tool_registry = ToolRegistry(vector_db=vector_db)
        self.model = model

    def plan(self, review_text: str) -> Dict[str, Any]:
        """
        Let LLM create analysis plan

        Args:
            review_text: Review to analyze

        Returns:
            Plan dict with reasoning and steps
        """

        # Build tools description
        tools_list = self.tool_registry.list_tools()
        tools_desc = json.dumps(tools_list, indent=2)

        # Create system prompt with tools
        system = AGENT_SYSTEM_PROMPT.format(tools_description=tools_desc)

        # User message
        user = f"""Analyze this employee review:

"{review_text}"

Create a plan for which tools to use."""

        try:
            response = _get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.3,
                max_tokens=400
            )

            content = response.choices[0].message.content

            # Parse JSON plan
            # Handle markdown wrappers
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            plan = json.loads(content.strip())

            return plan

        except json.JSONDecodeError:
            # Fallback to default plan if parsing fails
            print("  ⚠️ Plan parsing failed, using default plan")
            return {
                "reasoning": "Using standard pipeline (plan parsing failed)",
                "steps": [
                    {"tool": "retrieve_similar_reviews", "params": {"query_text": review_text, "k": 5}},
                    {"tool": "analyze_sentiment", "params": {"review_text": review_text}}
                ]
            }
        except Exception as e:
            print(f"  ⚠️ Planning error: {e}")
            return {
                "reasoning": f"Error during planning: {e}",
                "steps": []
            }

    def execute_plan(self, review_text: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the planned tool sequence

        Args:
            review_text: Original review
            plan: Plan from plan() method

        Returns:
            Results dict with tool outputs and final analysis
        """

        results = {
            "review_text": review_text,
            "plan": plan,
            "tool_outputs": [],
            "final_analysis": {}
        }

        # Track outputs for cross-step references
        step_outputs = {}

        for i, step in enumerate(plan.get('steps', [])):
            tool_name = step.get('tool')
            params = step.get('params', {})

            # Handle "from_previous_step" references
            if params:
                for key, value in list(params.items()):
                    if value == "from_previous_step" and step_outputs:
                        # Use results from last step
                        last_output = list(step_outputs.values())[-1]
                        if 'results' in last_output:
                            params[key] = last_output['results']

            # Execute tool
            print(f"  Step {i+1}/{len(plan.get('steps', []))}: {tool_name}")

            try:
                output = self.tool_registry.execute(tool_name, **params)

                results['tool_outputs'].append({
                    "step": i + 1,
                    "tool": tool_name,
                    "output": output
                })

                step_outputs[tool_name] = output

            except Exception as e:
                print(f"    ⚠️ Tool execution error: {e}")
                results['tool_outputs'].append({
                    "step": i + 1,
                    "tool": tool_name,
                    "error": str(e)
                })

        # Synthesize final analysis from tool outputs
        sentiment_output = step_outputs.get('analyze_sentiment', {})
        anomaly_output = step_outputs.get('detect_anomaly', {})

        results['final_analysis'] = {
            "sentiment": sentiment_output.get('sentiment'),
            "themes": sentiment_output.get('themes', []),
            "retention_risk": sentiment_output.get('retention_risk'),
            "is_anomalous": anomaly_output.get('is_anomalous', False),
            "anomaly_reason": anomaly_output.get('reason') if anomaly_output.get('reason') else anomaly_output.get('anomaly_reason'),
            "reasoning": plan.get('reasoning')
        }

        # Handle anomalous data in detect_anomaly output
        if anomaly_output.get('critical_combinations'):
            results['final_analysis']['anomaly_details'] = {
                'critical_combinations': anomaly_output['critical_combinations'],
                'rare_themes': anomaly_output.get('rare_themes')
            }

        return results

    def analyze(self, review_text: str) -> Dict[str, Any]:
        """
        Full agentic analysis: plan → execute → synthesize

        Args:
            review_text: Review to analyze

        Returns:
            Complete analysis results
        """

        print(f"\n{'='*60}")
        print("AGENTIC ANALYSIS")
        print(f"{'='*60}")
        print(f"Review: {review_text[:100]}{'...' if len(review_text) > 100 else ''}")

        # Step 1: Plan
        print(f"\n🧠 Planning...")
        plan = self.plan(review_text)
        print(f"Reasoning: {plan.get('reasoning', 'N/A')}")
        print(f"Steps: {len(plan.get('steps', []))}")

        # Step 2: Execute
        print(f"\n⚙️  Executing plan...")
        results = self.execute_plan(review_text, plan)

        # Step 3: Done
        print(f"\n✅ Analysis complete")
        print(f"{'='*60}\n")

        return results


# Alias for backwards compatibility
execute_tool = ToolRegistry.execute


def analyze_review_agentic(review_text: str, vector_db=None) -> Dict[str, Any]:
    """
    Analyze a review using the agent

    Args:
        review_text: Review to analyze
        vector_db: Optional vector database

    Returns:
        Analysis results
    """
    agent = ReviewAgent(vector_db=vector_db)
    return agent.analyze(review_text)


# Example usage
if __name__ == '__main__':
    # Test the agent
    test_review = """
    The benefits are decent and the pay is okay, but the mandatory overtime
    every single week is killing my work-life balance. Management doesn't
    communicate schedule changes until the last minute. During peak season
    it's even worse - 60 hour weeks are common.
    """

    print("Testing ReviewAgent:")
    result = analyze_review_agentic(test_review)

    print("\nFinal Analysis:")
    print(json.dumps(result['final_analysis'], indent=2))
