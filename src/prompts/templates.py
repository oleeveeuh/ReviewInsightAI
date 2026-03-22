#!/usr/bin/env python3
"""
Prompt engineering framework for LLM-based sentiment analysis.
Defines multiple prompt versions for A/B testing.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    version: str
    name: str
    system_prompt: str
    user_template: str
    few_shot_examples: Optional[List[Dict]] = None
    description: str = ""

    def render(self, review_text: str, k_shot: int = 0) -> Tuple[str, str]:
        """
        Render prompt with optional few-shot examples.

        Returns:
            (system_prompt, user_message)
        """
        # Build user message
        user_msg = self.user_template.format(review_text=review_text)

        # Add few-shot examples if requested
        if k_shot > 0 and self.few_shot_examples:
            examples_text = "\n\nHere are some example analyses:\n\n"
            for i, example in enumerate(self.few_shot_examples[:k_shot]):
                examples_text += f"Example {i+1}:\n"
                examples_text += f"Review: {example['review']}\n"
                examples_text += f"Analysis: {example['analysis']}\n\n"

            user_msg = examples_text + "\nNow analyze this review:\n\n" + user_msg

        return self.system_prompt, user_msg

    def render_full(self, review_text: str, k_shot: int = 0) -> str:
        """Render as a single combined prompt (for APIs that don't separate system/user)."""
        system, user = self.render(review_text, k_shot)
        return f"{system}\n\n{user}"


# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "review": "Great benefits and pay is decent, but the mandatory overtime is exhausting. Work-life balance is non-existent during peak season.",
        "analysis": '{"sentiment": 2, "themes": ["overtime", "pay_benefits", "work_life_balance"], "retention_risk": "medium", "reasoning": "Positive about pay/benefits but complaints about overtime and work-life balance indicate negative sentiment with medium retention risk."}'
    },
    {
        "review": "Management doesn't care about employees. Constant pressure to go faster. Unsafe working conditions, no breaks.",
        "analysis": '{"sentiment": 1, "themes": ["management", "workload", "safety"], "retention_risk": "high", "reasoning": "Severe complaints about management, safety, and workload with very negative tone indicate high retention risk."}'
    },
    {
        "review": "Pretty standard warehouse job. Pay is okay, not great. Gets boring but it pays the bills. Coworkers are nice.",
        "analysis": '{"sentiment": 3, "themes": ["pay_benefits", "workload", "culture"], "retention_risk": "low", "reasoning": "Neutral assessment with balanced pros/cons indicates average satisfaction and low retention risk."}'
    },
    {
        "review": "Love the flexible schedule! I can pick my own days. The work is easy but standing for 10 hours hurts my feet.",
        "analysis": '{"sentiment": 4, "themes": ["work_life_balance", "workload", "other"], "retention_risk": "low", "reasoning": "Mostly positive about flexibility and work despite minor physical discomfort indicates good satisfaction."}'
    },
    {
        "review": "Worst place I've ever worked. Toxic environment, managers yell at you, and you're just a number. Avoid!",
        "analysis": '{"sentiment": 1, "themes": ["management", "culture", "other"], "retention_risk": "high", "reasoning": "Extremely negative language about management and culture with warning to others indicates very high retention risk."}'
    },
    {
        "review": "Good pay and benefits start immediately. Opportunities for advancement if you work hard. Safety is taken seriously.",
        "analysis": '{"sentiment": 5, "themes": ["pay_benefits", "career_growth", "safety"], "retention_risk": "low", "reasoning": "Consistently positive across pay, growth, and safety indicates excellent satisfaction and very low retention risk."}'
    }
]


# ============================================================================
# PROMPT VERSIONS
# ============================================================================

V1_ZERO_SHOT = PromptTemplate(
    version="v1.0",
    name="Zero-shot Basic",
    description="Minimal prompt, no context or examples",
    system_prompt="You are an employee sentiment analyst.",
    user_template="""Analyze this employee review and extract:
1. Sentiment (1-5 scale)
2. Top 3 themes
3. Retention risk (low/medium/high)

Review: {review_text}

Return as JSON only:
{{
  "sentiment": <1-5>,
  "themes": ["theme1", "theme2", "theme3"],
  "retention_risk": "low|medium|high"
}}""",
    few_shot_examples=FEW_SHOT_EXAMPLES
)


V2_ROLE_ENHANCED = PromptTemplate(
    version="v2.0",
    name="Role Context Enhanced",
    description="Adds HR analyst role and warehouse domain knowledge",
    system_prompt="""You are an HR analyst specializing in employee retention at Amazon fulfillment centers and similar warehouse operations.

You understand industry-specific terminology:
- VTO: Voluntary Time Off
- MET: Mandatory Extra Time (overtime)
- Peak Season: High-volume periods (November-December, Prime Day)
- FC: Fulfillment Center
- Rate/Quota: Items processed per hour

When analyzing sentiment, consider:
- Overall tone, not just individual complaints
- Severity of issues raised
- Whether complaints are deal-breakers or minor annoyances""",
    user_template="""Analyze this employee review:

Review: {review_text}

Extract:
1. Sentiment (1=Very Negative, 2=Negative, 3=Neutral, 4=Positive, 5=Very Positive)
2. Main themes (select up to 3: overtime, pay_benefits, management, safety, career_growth, workload, work_life_balance, training, culture, other)
3. Retention risk (low/medium/high based on severity of complaints)

Return JSON only, no other text:
{{
  "sentiment": <1-5>,
  "themes": ["theme1", "theme2", "theme3"],
  "retention_risk": "low|medium|high"
}}""",
    few_shot_examples=FEW_SHOT_EXAMPLES
)


V3_FEW_SHOT = PromptTemplate(
    version="v3.0",
    name="3-Shot Learning",
    description="Adds few-shot examples to guide the model",
    system_prompt=V2_ROLE_ENHANCED.system_prompt,
    user_template=V2_ROLE_ENHANCED.user_template,
    few_shot_examples=FEW_SHOT_EXAMPLES
)


V4_CHAIN_OF_THOUGHT = PromptTemplate(
    version="v4.0",
    name="Chain of Thought",
    description="Asks model to think step-by-step and include reasoning in JSON output",
    system_prompt=V2_ROLE_ENHANCED.system_prompt,
    user_template="""Analyze this review and provide your analysis as JSON only.

Review: {review_text}

In your response, include a brief reasoning that explains your analysis.

Return ONLY valid JSON:
{{
  "sentiment": <1-5>,
  "themes": ["theme1", "theme2"],
  "retention_risk": "low|medium|high",
  "reasoning": "<brief explanation of your analysis>"
}}""",
    few_shot_examples=FEW_SHOT_EXAMPLES
)


V5_STRUCTURED = PromptTemplate(
    version="v5.0",
    name="Structured Output",
    description="Explicit JSON schema, minimal fluff",
    system_prompt=V2_ROLE_ENHANCED.system_prompt,
    user_template="""Review: {review_text}

Return ONLY valid JSON, no other text:
{{
  "sentiment": <number 1-5>,
  "themes": [<up to 3 from: overtime, pay_benefits, management, safety, career_growth, workload, work_life_balance, training, culture, other>],
  "retention_risk": "<low|medium|high>"
}}""",
    few_shot_examples=FEW_SHOT_EXAMPLES
)


V6_DETAILED = PromptTemplate(
    version="v6.0",
    name="Detailed Analysis",
    description="Most comprehensive prompt with multiple outputs",
    system_prompt="""You are an expert HR analyst specializing in employee sentiment and retention at warehouse/fulfillment centers.

Your analysis considers:
- **Sentiment**: The overall emotional tone, beyond simple keyword counting
- **Themes**: Key topics discussed (workplace-specific categories)
- **Retention Risk**: Likelihood the employee will leave based on dissatisfaction severity
- **Key Insights**: Brief summary of main takeaways

Warehouse context you understand:
- Physical demands (standing all shift, lifting, heat/cold)
- Operational pressures (rates, quotas, mandatory overtime)
- Management structures (shift leads, area managers, HR)
- Seasonal variations (peak vs. non-peak periods)""",
    user_template="""Review: {review_text}

Provide a comprehensive analysis:

1. **Sentiment Score** (1-5):
   - 1 = Very Negative (hated it, actively warning others)
   - 2 = Negative (mostly complaints, disappointed)
   - 3 = Neutral (balanced pros/cons, factual)
   - 4 = Positive (mostly praise, satisfied)
   - 5 = Very Positive (loves it, highly recommends)

2. **Themes Present** (select ALL that apply):
   - overtime, pay_benefits, management, safety, career_growth, workload, work_life_balance, training, culture, other

3. **Retention Risk** (low/medium/high):
   - Low: Satisfied or neutral, likely to stay
   - Medium: Has complaints but manageable
   - High: Very dissatisfied, likely to quit

4. **Key Insight**: One sentence summary

Return as JSON:
{{
  "sentiment": <1-5>,
  "themes": ["theme1", "theme2", ...],
  "retention_risk": "low|medium|high",
  "key_insight": "<one sentence>"
}}""",
    few_shot_examples=FEW_SHOT_EXAMPLES
)


# ============================================================================
# EXPORTS
# ============================================================================

ALL_PROMPTS = {
    'v1': V1_ZERO_SHOT,
    'v2': V2_ROLE_ENHANCED,
    'v3': V3_FEW_SHOT,
    'v4': V4_CHAIN_OF_THOUGHT,
    'v5': V5_STRUCTURED,
    'v6': V6_DETAILED,
}


def get_prompt(version: str = 'v3') -> PromptTemplate:
    """Get a prompt template by version."""
    if version not in ALL_PROMPTS:
        available = ', '.join(ALL_PROMPTS.keys())
        raise ValueError(f"Unknown version: {version}. Available: {available}")
    return ALL_PROMPTS[version]


def compare_prompt_versions() -> str:
    """Print comparison of all prompt versions."""
    output = ["=" * 70]
    output.append("PROMPT VERSION COMPARISON")
    output.append("=" * 70)

    for version_id, prompt in ALL_PROMPTS.items():
        output.append(f"\n{version_id.upper()} - {prompt.name}")
        output.append(f"Description: {prompt.description}")
        output.append(f"Few-shot examples: {len(prompt.few_shot_examples) if prompt.few_shot_examples else 0}")

    output.append("\n" + "=" * 70)
    return "\n".join(output)


if __name__ == '__main__':
    # Print comparison
    print(compare_prompt_versions())

    # Example usage
    print("\n" + "=" * 70)
    print("EXAMPLE RENDERING (V3 with 2-shot)")
    print("=" * 70)

    sample_review = "The pay is good but management is terrible. Standing for 10 hours hurts."
    system, user = V3_FEW_SHOT.render(sample_review, k_shot=2)

    print("\n--- SYSTEM PROMPT ---")
    print(system[:200] + "...")
    print("\n--- USER MESSAGE ---")
    print(user[:500] + "...")
