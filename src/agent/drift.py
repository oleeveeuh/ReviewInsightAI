#!/usr/bin/env python3
"""
Drift detection system for ReviewInsight AI.

Tracks how employee concerns change over time by comparing current theme
distributions to historical baselines using KL Divergence.

Mathematical details:
- KL divergence: D_KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
- Measures how much current distribution diverges from baseline
- Higher values = more drift

Usage:
    from src.agent.drift import DriftDetector, DetectDriftTool

    detector = DriftDetector(baseline_distribution={
        'overtime': 0.34,
        'pay_benefits': 0.28,
        'management': 0.25
    })

    result = detector.detect_drift(current_distribution)
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import numpy as np
    from scipy.stats import entropy
    HAS_SCIPY = True
except ImportError:
    # Fallback without scipy
    HAS_SCIPY = False
    import math

    def entropy(pk, qk=None):
        """Fallback entropy calculation without scipy."""
        if qk is None:
            return -sum([p * math.log(p) for p in pk if p > 0])
        else:
            return sum([pk[i] * math.log(pk[i] / qk[i]) for i in range(len(pk)) if pk[i] > 0 and qk[i] > 0])


from agent.tools import Tool


class DriftDetector:
    """Detect distribution drift in themes using KL divergence"""

    def __init__(self, baseline_distribution: Dict[str, float]):
        """
        Initialize with baseline theme distribution

        Args:
            baseline_distribution: Historical theme frequencies
        """
        self.baseline = baseline_distribution
        self.all_themes = sorted(baseline_distribution.keys())

    def detect_drift(self, current_distribution: Dict[str, float],
                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect if current distribution has drifted from baseline

        Args:
            current_distribution: Current theme frequencies
            threshold: KL divergence threshold for alert

        Returns:
            Dict with drift analysis
        """

        # Align distributions (add missing themes with small epsilon)
        epsilon = 0.001
        baseline_vec = []
        current_vec = []

        # Get union of all themes
        all_themes_union = set(self.all_themes) | set(current_distribution.keys())
        all_themes_sorted = sorted(all_themes_union)

        for theme in all_themes_sorted:
            baseline_vec.append(self.baseline.get(theme, epsilon))
            current_vec.append(current_distribution.get(theme, epsilon))

        # Normalize to probabilities
        if HAS_SCIPY:
            baseline_vec = np.array(baseline_vec, dtype=float)
            current_vec = np.array(current_vec, dtype=float)
        else:
            baseline_vec = [float(x) for x in baseline_vec]
            current_vec = [float(x) for x in current_vec]

        baseline_sum = sum(baseline_vec)
        current_sum = sum(current_vec)

        if baseline_sum > 0:
            baseline_vec = [x / baseline_sum for x in baseline_vec]
        if current_sum > 0:
            current_vec = [x / current_sum for x in current_vec]

        # Calculate KL divergence
        # KL(current || baseline) = sum(current * log(current / baseline))
        kl_div = entropy(current_vec, baseline_vec)

        # Determine if drifting
        is_drifting = kl_div > threshold

        # Identify significant changes (>5 percentage points)
        significant_changes = {}

        for theme in all_themes_sorted:
            base_freq = self.baseline.get(theme, 0)
            curr_freq = current_distribution.get(theme, 0)
            change = curr_freq - base_freq

            if abs(change) > 0.05:  # 5% threshold
                significant_changes[theme] = {
                    'baseline': float(base_freq),
                    'current': float(curr_freq),
                    'change': float(change),
                    'direction': 'increase' if change > 0 else 'decrease'
                }

        # Build alert message
        if is_drifting:
            alert_msg = f"⚠️ DRIFT DETECTED: KL divergence = {kl_div:.3f} (threshold: {threshold})"
            if significant_changes:
                top_change = max(significant_changes.items(),
                                 key=lambda x: abs(x[1]['change']))
                top_name, top = top_change
                alert_msg += (f"\nLargest change: {top_name} "
                              f"({top['direction']} by "
                              f"{abs(top['change']) * 100:.1f}%)")
        else:
            alert_msg = f"✓ No drift detected (KL divergence = {kl_div:.3f})"

        return {
            'is_drifting': is_drifting,
            'kl_divergence': float(kl_div),
            'threshold': threshold,
            'significant_changes': significant_changes,
            'alert': alert_msg
        }


class DetectDriftTool(Tool):
    """Tool for drift detection"""

    name: str = "detect_drift"
    description: str = "Checks if recent theme distribution has drifted from baseline. Use to identify emerging issues."
    parameters: Dict[str, Any] = {
        "current_themes": {
            "type": "array",
            "description": "Themes from recent reviews (optional - will use memory if not provided)"
        }
    }

    # Add fields for memory and detector
    memory: Optional[Any] = None
    _detector: Optional[DriftDetector] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, memory=None, **data):
        """
        Initialize drift tool

        Args:
            memory: AgentMemory instance for accessing history
        """
        # Set memory before super init
        data['memory'] = memory
        super().__init__(**data)

        # Set baseline from our dataset (these frequencies are from EDA)
        baseline = {
            'overtime': 0.55,
            'management': 0.40,
            'pay_benefits': 0.36,
            'workload': 0.33,
            'work_life_balance': 0.26,
            'culture': 0.15,
            'safety': 0.12,
            'career_growth': 0.10,
            'training': 0.08,
            'other': 0.20
        }

        # Store detector
        object.__setattr__(self, '_detector', DriftDetector(baseline))

    @property
    def detector(self) -> DriftDetector:
        """Get the detector instance."""
        return self._detector

    def execute(self, current_themes: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute drift detection

        Args:
            current_themes: Optional list of themes from recent reviews

        Returns:
            Drift analysis results
        """

        # Get current distribution
        if current_themes:
            # Calculate from provided themes
            theme_counts = {}
            for theme in current_themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            total = sum(theme_counts.values())
            current_dist = {t: c / total for t, c in theme_counts.items()}
        elif self.memory:
            # Use memory (last 50 analyses)
            current_dist = self.memory.get_theme_distribution(last_n=50)
        else:
            return {
                "tool": self.name,
                "error": "No data available for drift detection - provide current_themes or initialize with memory"
            }

        if not current_dist:
            return {
                "tool": self.name,
                "error": "No data available for drift detection"
            }

        # Detect drift
        drift_result = self.detector.detect_drift(current_dist, threshold=0.1)

        return {
            "tool": self.name,
            **drift_result
        }


# Example usage
if __name__ == '__main__':
    # Test drift detector
    baseline = {
        'overtime': 0.34,
        'pay_benefits': 0.28,
        'management': 0.25,
        'workload': 0.22
    }

    # Simulate drift: overtime increased significantly
    current = {
        'overtime': 0.55,  # +21% increase!
        'pay_benefits': 0.25,
        'management': 0.15,
        'workload': 0.20
    }

    detector = DriftDetector(baseline)
    result = detector.detect_drift(current, threshold=0.1)

    print("Drift Detection Test:")
    print(f"Is drifting: {result['is_drifting']}")
    print(f"KL divergence: {result['kl_divergence']:.3f}")
    print(f"Alert: {result['alert']}")

    if result['significant_changes']:
        print("\nSignificant changes:")
        for theme, change in result['significant_changes'].items():
            print(f"  {theme}: {change['baseline']:.2f} → {change['current']:.2f} ({change['direction']})")
