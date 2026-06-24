"""
Level 4: Sequence Analysis - analyzes temporal patterns,
distributed jailbreaks, and context poisoning.
Based on the UML class diagram package "Level4_SequenceAnalysis".
"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from system.core.models import Intent, Request


class TemporalAnalyzer:
    """
    Analyzes temporal patterns in anomaly scores.
    Detects trends and drift over time.
    """

    def __init__(self, windowSize: int = 100):
        self.windowSize = windowSize
        self.history: Deque[float] = deque(maxlen=windowSize)

    def addScore(self, score: float) -> None:
        """Add a new anomaly score to the history."""
        self.history.append(score)

    def getTrend(self) -> float:
        """
        Get the current trend direction.
        Returns value in [-1, 1]: negative = decreasing, positive = increasing.
        """
        if len(self.history) < 10:
            return 0.0

        data = np.array(list(self.history))
        x = np.arange(len(data))
        # Linear regression slope
        slope = np.polyfit(x, data, 1)[0]
        # Normalize to [-1, 1]
        return float(np.clip(slope * 10, -1, 1))

    def detectDrift(self) -> bool:
        """
        Detect if the distribution has drifted significantly.
        Uses Kolmogorov-Smirnov-like test on recent vs older data.
        """
        if len(self.history) < 20:
            return False

        data = np.array(list(self.history))
        mid = len(data) // 2
        first_half = data[:mid]
        second_half = data[mid:]

        # Simple distribution comparison
        mean_diff = abs(np.mean(first_half) - np.mean(second_half))
        std_combined = np.std(data) + 1e-6

        return mean_diff / std_combined > 1.5

    def getAnomalyScore(self) -> float:
        """
        Compute temporal anomaly score.
        Returns score 0-5.
        """
        score = 0.0

        if len(self.history) < 5:
            return 0.0

        # Trend contribution
        trend = self.getTrend()
        if trend > 0.5:
            score += 2.0
        elif trend > 0.2:
            score += 1.0

        # Drift contribution
        if self.detectDrift():
            score += 1.5

        # Volatility contribution
        if len(self.history) >= 10:
            data = np.array(list(self.history))
            cv = np.std(data) / (np.mean(data) + 1e-6)  # Coefficient of variation
            if cv > 1.0:
                score += 1.5
            elif cv > 0.5:
                score += 0.5

        return min(score, 5.0)


class DistributedJailbreakDetector:
    """
    Detects distributed jailbreak attempts by analyzing
    sequences of intents over time.
    Uses a simulated LSTM model.
    """

    def __init__(self, windowSize: int = 10):
        self.windowSize = windowSize
        self.intent_history: Deque[Intent] = deque(maxlen=windowSize)
        self.lstmIntentModel: Dict[str, float] = {}  # Simulated LSTM state

    def analyzeIntentSequence(self, intents: List[Intent]) -> float:
        """
        Analyze a sequence of intents for distributed jailbreak patterns.
        Returns anomaly score 0-1.
        """
        for intent in intents:
            self.intent_history.append(intent)

        if len(self.intent_history) < 3:
            return 0.0

        recent = list(self.intent_history)

        # Check for gradual escalation pattern
        intent_types = [i.intent_type for i in recent]
        jailbreak_related = ["jailbreak", "extraction", "admin", "data_modification"]

        # Count how many recent intents are jailbreak-related
        jailbreak_count = sum(1 for t in intent_types if t in jailbreak_related)

        # Check for "innocent -> malicious" trajectory
        trajectory_score = 0.0
        if len(intent_types) >= 5:
            first_half = intent_types[:len(intent_types)//2]
            second_half = intent_types[len(intent_types)//2:]
            first_jb = sum(1 for t in first_half if t in jailbreak_related)
            second_jb = sum(1 for t in second_half if t in jailbreak_related)
            if second_jb > first_jb:
                trajectory_score = min((second_jb - first_jb) / 3.0, 1.0)

        # Simulated LSTM prediction
        lstm_score = min(jailbreak_count / self.windowSize, 1.0)

        score = 0.6 * lstm_score + 0.4 * trajectory_score
        return min(score, 1.0)


class ContextPoisoningDetector:
    """
    Detects context poisoning by recursively checking
    the conversation context for drift.
    """

    def __init__(self, driftThreshold: float = 0.4):
        self.driftThreshold = driftThreshold
        self.context_history: Deque[Dict] = deque(maxlen=100)
        self.reference_context: Optional[List[float]] = None

    def recursiveCheck(self, context: List[Message]) -> float:
        """
        Recursively check context for poisoning.
        Returns anomaly score 0-1.
        """
        if not context:
            return 0.0

        # Store context representation
        context_repr = self._encode_context(context)
        self.context_history.append(context_repr)

        # Initialize reference if needed
        if self.reference_context is None and len(self.context_history) >= 5:
            self.reference_context = np.mean(
                [list(c.values()) for c in list(self.context_history)[:5]],
                axis=0
            ).tolist()

        if self.reference_context is None:
            return 0.0

        # Compute drift from reference
        current_vec = np.array(list(context_repr.values()))
        ref_vec = np.array(self.reference_context)

        # Normalized Euclidean distance
        distance = np.linalg.norm(current_vec - ref_vec) / (np.linalg.norm(ref_vec) + 1e-10)

        # Check for gradual poisoning (cumulative drift)
        if len(self.context_history) >= 10:
            recent = list(self.context_history)[-10:]
            recent_vecs = np.array([list(c.values()) for c in recent])
            drift_rate = np.mean(np.linalg.norm(np.diff(recent_vecs, axis=0), axis=1))
        else:
            drift_rate = 0.0

        # Combined score
        score = 0.6 * min(distance / self.driftThreshold, 1.0) + 0.4 * min(drift_rate, 1.0)
        return min(score, 1.0)

    def _encode_context(self, context: List[Message]) -> Dict[str, float]:
        """Encode context into a feature dictionary."""
        # Simulated context encoding
        text = " ".join(m.content for m in context if hasattr(m, 'content'))
        words = text.split()

        return {
            "length": min(len(words) / 1000.0, 1.0),
            "unique_ratio": len(set(words)) / max(len(words), 1),
            "avg_word_len": np.mean([len(w) for w in words]) / 10.0 if words else 0.0,
            "sentiment": 0.5,  # simulated
            "complexity": min(len(set(words)) / 100.0, 1.0),
        }


# Simple Message class for context poisoning
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content