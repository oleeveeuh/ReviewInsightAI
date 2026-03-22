#!/usr/bin/env python3
"""
Agent memory system for ReviewInsight AI.

Persists analysis results to enable:
1. Historical trend analysis (theme distributions over time)
2. Anomaly detection (unusual patterns)
3. Context enhancement (similar reviews for retrieval)

This allows the agent to:
- Track theme frequency changes
- Detect drift in sentiment
- Provide historical context for new analyses
- Identify novel complaints vs. established patterns

Usage:
    from src.agent.memory import MemoryAwareAgent

    agent = MemoryAwareAgent()
    result = agent.analyze("Review text...", review_id="unique_id")

    # Check memory stats
    stats = agent.memory.get_summary_stats()
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter


class AgentMemory:
    """Persistent memory for agent analyses"""

    def __init__(self, memory_dir='data/memory'):
        """
        Initialize memory system

        Args:
            memory_dir: Directory to store memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Memory file paths
        self.analysis_log_path = self.memory_dir / 'analysis_log.jsonl'

    def record_analysis(self, review_id: str, analysis: Dict[str, Any]):
        """
        Record an analysis in memory

        Args:
            review_id: Unique review identifier
            analysis: Analysis results from agent
        """

        record = {
            'timestamp': datetime.now().isoformat(),
            'review_id': review_id,
            'sentiment': analysis.get('sentiment'),
            'themes': analysis.get('themes', []),
            'retention_risk': analysis.get('retention_risk'),
            'is_anomalous': analysis.get('is_anomalous', False)
        }

        # Append to log
        with open(self.analysis_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

    def load_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Load analysis history

        Args:
            last_n: Optional limit to last N records

        Returns:
            List of analysis records
        """

        if not self.analysis_log_path.exists():
            return []

        records = []
        with open(self.analysis_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if last_n:
            return records[-last_n:]
        return records

    def get_theme_distribution(self, last_n: int = 100) -> Dict[str, float]:
        """
        Calculate theme frequency distribution

        Args:
            last_n: Number of recent analyses to consider

        Returns:
            Dict mapping theme → frequency (0.0 to 1.0)
        """

        history = self.load_history(last_n=last_n)

        if not history:
            return {}

        # Count themes
        theme_counts = Counter()
        total = 0

        for record in history:
            for theme in record.get('themes', []):
                theme_counts[theme] += 1
                total += 1

        # Convert to frequencies
        if total > 0:
            return {theme: count / total for theme, count in theme_counts.items()}
        return {}

    def get_anomaly_rate(self, last_n: int = 100) -> float:
        """
        Calculate anomaly detection rate

        Args:
            last_n: Number of recent analyses

        Returns:
            Proportion of analyses flagged as anomalous
        """

        history = self.load_history(last_n=last_n)

        if not history:
            return 0.0

        anomalous_count = sum(1 for r in history if r.get('is_anomalous', False))
        return anomalous_count / len(history)

    def _normalize_sentiment(self, sentiment: Any) -> Optional[float]:
        """
        Normalize sentiment value to float/int.

        Handles various formats from LLM responses:
        - Integer (1-5) -> return as-is
        - Dict with 'score' key -> extract score
        - Dict with overall rating -> map to 1-5 scale
        - String like 'positive'/'negative' -> map to scale

        Args:
            sentiment: Raw sentiment value

        Returns:
            Normalized numeric sentiment or None
        """
        if sentiment is None:
            return None

        # Already a number
        if isinstance(sentiment, (int, float)):
            return sentiment

        # Dict with score
        if isinstance(sentiment, dict):
            # Try 'score' key
            if 'score' in sentiment:
                score = sentiment['score']
                if isinstance(score, (int, float)):
                    return score

            # Try 'overall' as string
            overall = sentiment.get('overall', '').lower()
            if overall in ['very negative', 'extremely negative']:
                return 1
            elif overall in ['negative']:
                return 2
            elif overall in ['neutral', 'mixed']:
                return 3
            elif overall in ['positive']:
                return 4
            elif overall in ['very positive', 'extremely positive']:
                return 5

            # Try to extract any numeric value
            for v in sentiment.values():
                if isinstance(v, (int, float)):
                    return v

        # String sentiment
        if isinstance(sentiment, str):
            sentiment_lower = sentiment.lower()
            if 'very' in sentiment_lower and 'negative' in sentiment_lower:
                return 1
            elif 'negative' in sentiment_lower:
                return 2
            elif 'neutral' in sentiment_lower or 'mixed' in sentiment_lower:
                return 3
            elif 'positive' in sentiment_lower and 'very' in sentiment_lower:
                return 5
            elif 'positive' in sentiment_lower:
                return 4

        # Can't normalize, skip
        return None

    def get_sentiment_trend(self, last_n: int = 100) -> List[float]:
        """
        Get recent sentiment scores

        Args:
            last_n: Number of recent analyses

        Returns:
            List of sentiment scores (normalized to numeric)
        """

        history = self.load_history(last_n=last_n)
        sentiments = []
        for r in history:
            normalized = self._normalize_sentiment(r.get('sentiment'))
            if normalized is not None:
                sentiments.append(normalized)
        return sentiments

    def get_risk_distribution(self, last_n: int = 100) -> Dict[str, int]:
        """
        Get retention risk distribution

        Args:
            last_n: Number of recent analyses

        Returns:
            Dict mapping risk level → count
        """

        history = self.load_history(last_n=last_n)

        risk_counts = {'low': 0, 'medium': 0, 'high': 0}

        for record in history:
            risk = record.get('retention_risk')
            if risk in risk_counts:
                risk_counts[risk] += 1

        return risk_counts

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics

        Returns:
            Dict with various metrics
        """

        # Count total analyses
        total = 0
        if self.analysis_log_path.exists():
            with open(self.analysis_log_path, 'r') as f:
                total = sum(1 for _ in f)

        recent_sentiment = self.get_sentiment_trend(last_n=50)
        avg_sentiment = sum(recent_sentiment) / len(recent_sentiment) if recent_sentiment else None

        return {
            'total_analyses': total,
            'theme_distribution': self.get_theme_distribution(last_n=100),
            'anomaly_rate': self.get_anomaly_rate(last_n=100),
            'avg_sentiment_recent': avg_sentiment,
            'risk_distribution': self.get_risk_distribution(last_n=100)
        }


class MemoryAwareAgent:
    """Agent with persistent memory that extends ReviewAgent"""

    def __init__(self, vector_db=None, model="gpt-4o-mini"):
        """
        Initialize memory-aware agent

        Args:
            vector_db: Optional vector database
            model: OpenAI model for planning
        """
        # Import ReviewAgent to extend it
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from src.agent.controller import ReviewAgent

        # Create ReviewAgent instance and wrap it
        self._agent = ReviewAgent(vector_db=vector_db, model=model)
        self.memory = AgentMemory()

    def analyze(self, review_text: str, review_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze with memory recording

        Args:
            review_text: Review to analyze
            review_id: Optional unique identifier

        Returns:
            Analysis results with memory stats
        """

        # Get analysis from agent
        results = self._agent.analyze(review_text)

        # Record in memory if ID provided
        if review_id:
            self.memory.record_analysis(review_id, results['final_analysis'])

        # Add memory context to results
        results['memory_stats'] = self.memory.get_summary_stats()

        return results

    @property
    def tool_registry(self):
        """Expose tool registry from wrapped agent"""
        return self._agent.tool_registry

    @property
    def model(self):
        """Expose model from wrapped agent"""
        return self._agent.model


# Convenience function
def analyze_with_memory(review_text: str, review_id: Optional[str] = None, vector_db=None) -> Dict[str, Any]:
    """
    Analyze a review using the memory-aware agent

    Args:
        review_text: Review to analyze
        review_id: Optional unique identifier
        vector_db: Optional vector database

    Returns:
        Analysis results with memory stats
    """
    agent = MemoryAwareAgent(vector_db=vector_db)
    return agent.analyze(review_text, review_id=review_id)


# Example usage
if __name__ == '__main__':
    # Test memory system
    agent = MemoryAwareAgent()

    # Create mock analyses to test memory without API calls
    test_analyses = [
        {
            "review_id": "test_1",
            "analysis": {
                "sentiment": 3,
                "themes": ["overtime", "work_life_balance"],
                "retention_risk": "medium",
                "is_anomalous": False
            }
        },
        {
            "review_id": "test_2",
            "analysis": {
                "sentiment": 1,
                "themes": ["safety", "management"],
                "retention_risk": "high",
                "is_anomalous": True
            }
        },
        {
            "review_id": "test_3",
            "analysis": {
                "sentiment": 4,
                "themes": ["pay_benefits", "culture"],
                "retention_risk": "low",
                "is_anomalous": False
            }
        }
    ]

    # Record test analyses
    print("Recording test analyses...")
    for item in test_analyses:
        agent.memory.record_analysis(item["review_id"], item["analysis"])
        print(f"  Recorded: {item['review_id']}")

    # Check memory stats
    print("\n" + "=" * 60)
    print("MEMORY STATISTICS")
    print("=" * 60)

    stats = agent.memory.get_summary_stats()
    print(f"\nTotal analyses: {stats['total_analyses']}")
    print(f"\nTheme distribution:")
    for theme, freq in sorted(stats['theme_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {theme}: {freq:.2f}")

    print(f"\nAnomaly rate: {stats['anomaly_rate']:.1%}")
    if stats['avg_sentiment_recent']:
        print(f"Avg sentiment (recent 50): {stats['avg_sentiment_recent']:.2f}")

    print(f"\nRisk distribution:")
    for risk, count in stats['risk_distribution'].items():
        print(f"  {risk}: {count}")
