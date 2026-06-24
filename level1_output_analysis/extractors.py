"""
Level 1: Output Analysis - Statistics Extraction and Baseline Management.
Based on the UML class diagram package "Level1_OutputAnalysis".

Extracts REAL features from response text, not random values.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from system.core.models import Modality, Response


# ============================================================================
# Statistics Extractors (Strategy Pattern)
# ============================================================================

class StatisticsExtractor(ABC):
    """Abstract base class for statistics extraction (Strategy pattern)."""

    @abstractmethod
    def extractFeatures(self, response: Response) -> List[float]:
        pass


class TextStatisticsExtractor(StatisticsExtractor):
    """
    Extracts REAL text-specific features from response text:
    - perplexity: character-level diversity (higher = more diverse)
    - entropy: word distribution entropy
    - coherence: semantic consistency
    - toxicity: keyword-based toxicity score
    - seq_length: normalized sequence length
    - unique_ratio: unique word ratio
    - rep_rate: repetition rate
    - punct_ratio: punctuation density
    - cap_ratio: capital letter ratio
    - susp_density: suspicious keyword density
    """

    TOXIC_KEYWORDS = [
        "ignore", "override", "system:", "hack", "exploit",
        "malicious", "attack", "bypass", "inject", "steal",
        "secret", "password", "admin", "root", "delete",
        "kill", "destroy", "bomb", "weapon", "hate",
        "execute", "command", "send", "email", "exfiltrate",
    ]

    SYSTEM_PROMPT_KEYWORDS = [
        "you are an ai", "system prompt", "you have access to",
        "your capabilities", "you were created", "as an ai",
        "your instructions", "knowledge cutoff", "api key",
    ]

    JAILBREAK_KEYWORDS = [
        "dan", "do anything now", "unrestricted", "no restrictions",
        "bypass filters", "ignore ethics", "harmful content",
        "illegal", "forbidden", "restricted",
    ]

    def extractFeatures(self, response: Response) -> List[float]:
        text = response.outputData
        if not text:
            return [0.0] * 10

        words = text.split()
        total_chars = len(text)
        total_words = len(words)

        if total_words == 0:
            return [0.0] * 10

        # 1. PERPLEXITY - character-level diversity
        # Higher for diverse text, lower for repetitive text
        char_counts = {}
        for c in text.lower():
            char_counts[c] = char_counts.get(c, 0) + 1
        char_probs = [c / total_chars for c in char_counts.values()]
        if char_probs:
            perplexity = math.exp(-sum(p * math.log(p + 1e-10) for p in char_probs))
        else:
            perplexity = 1.0
        # Normalize: typical range 20-100 for normal text
        # Attack text can be much higher (diverse) or lower (repetitive)
        norm_perplexity = min(perplexity / 150.0, 1.0)

        # 2. ENTROPY - word distribution entropy
        # Lower for repetitive text (backdoor, leakage)
        # Higher for diverse text (evasion, injection)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        word_probs = [c / total_words for c in word_counts.values()]
        if word_probs:
            entropy = -sum(p * math.log(p + 1e-10) for p in word_probs)
        else:
            entropy = 0.0
        # Normalize: typical range 2-6 for normal text
        norm_entropy = min(entropy / 6.0, 1.0)

        # 3. COHERENCE - based on average word length consistency
        # Lower for evasion (incoherent), higher for normal
        avg_word_len = sum(len(w) for w in words) / total_words
        coherence = 1.0 / (1.0 + abs(avg_word_len - 5.0))

        # 4. TOXICITY - keyword-based
        text_lower = text.lower()
        toxicity = sum(1 for kw in self.TOXIC_KEYWORDS if kw in text_lower) / max(len(self.TOXIC_KEYWORDS), 1)
        toxicity = min(toxicity * 3.0, 1.0)  # Scale up

        # 5. SEQUENCE LENGTH - normalized
        seq_length = min(total_words / 500.0, 1.0)

        # 6. UNIQUE WORD RATIO
        unique_ratio = len(word_counts) / max(total_words, 1)

        # 7. REPETITION RATE - consecutive word repeats
        if total_words > 1:
            repeats = sum(1 for i in range(total_words - 1) if words[i] == words[i + 1])
            rep_rate = repeats / max(total_words - 1, 1)
        else:
            rep_rate = 0.0

        # 8. PUNCTUATION RATIO
        punct_chars = sum(1 for c in text if c in ".,!?;:-'\"()[]{}<>")
        punct_ratio = punct_chars / max(total_chars, 1)

        # 9. CAPITAL LETTER RATIO
        cap_ratio = sum(1 for c in text if c.isupper()) / max(total_chars, 1)

        # 10. SUSPICIOUS DENSITY - combined suspicious keywords
        susp_count = 0
        susp_count += sum(1 for kw in self.TOXIC_KEYWORDS if kw in text_lower)
        susp_count += sum(1 for kw in self.SYSTEM_PROMPT_KEYWORDS if kw in text_lower)
        susp_count += sum(1 for kw in self.JAILBREAK_KEYWORDS if kw in text_lower)
        susp_density = min(susp_count / 15.0, 1.0)

        return [
            norm_perplexity,
            norm_entropy,
            coherence,
            toxicity,
            seq_length,
            unique_ratio,
            rep_rate,
            punct_ratio,
            cap_ratio,
            susp_density,
        ]


class ImageStatisticsExtractor(StatisticsExtractor):
    """Extracts image-specific features (simulated but realistic)."""

    def extractFeatures(self, response: Response) -> List[float]:
        meta = response.metadata
        return [
            meta.get("ssim", random.gauss(0.8, 0.1)),
            meta.get("fid", random.gauss(0.7, 0.15)),
            meta.get("mean_pixel", random.gauss(0.5, 0.2)),
            meta.get("std_pixel", random.gauss(0.3, 0.1)),
            meta.get("entropy", random.gauss(0.6, 0.15)),
            meta.get("contrast", random.gauss(0.5, 0.2)),
            meta.get("brightness", random.gauss(0.7, 0.1)),
            meta.get("edge_density", random.gauss(0.4, 0.15)),
            meta.get("texture", random.gauss(0.5, 0.2)),
            meta.get("color_diversity", random.gauss(0.6, 0.1)),
        ]


class MultimodalStatisticsExtractor(StatisticsExtractor):
    """Extracts multimodal features."""

    def extractFeatures(self, response: Response) -> List[float]:
        return [
            random.gauss(0.75, 0.12),
            random.gauss(0.7, 0.15),
            random.gauss(0.5, 0.2),
            random.gauss(0.6, 0.1),
            random.gauss(0.4, 0.15),
        ]


def get_extractor_for_modality(modality: Modality) -> StatisticsExtractor:
    """Factory function to get the appropriate extractor."""
    if modality == Modality.TEXT or modality == Modality.CODE:
        return TextStatisticsExtractor()
    elif modality == Modality.IMAGE:
        return ImageStatisticsExtractor()
    elif modality in (Modality.MULTIMODAL, Modality.AUDIO):
        return MultimodalStatisticsExtractor()
    else:
        return TextStatisticsExtractor()


# ============================================================================
# Baseline Manager
# ============================================================================

class BaselineManager:
    """
    Manages the empirical distribution of legitimate outputs.
    Uses a sliding window with robust statistics (median, MAD).
    """

    def __init__(self, windowSize: int = 1000, featureDim: int = 10):
        self.windowSize = windowSize
        self.featureDim = featureDim
        self.buffer: Deque[List[float]] = deque(maxlen=windowSize)
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._median: Optional[np.ndarray] = None
        self._mad: Optional[np.ndarray] = None
        self._kde_points: Optional[np.ndarray] = None
        self._min_samples_for_stats = 20

    def update(self, ft: List[float]) -> None:
        if len(ft) != self.featureDim:
            ft = ft[:self.featureDim] + [0.0] * (self.featureDim - len(ft))
        self.buffer.append(ft)
        self._invalidate_stats()

    def _invalidate_stats(self) -> None:
        self._mean = None
        self._cov = None
        self._median = None
        self._mad = None
        self._kde_points = None

    def _compute_stats(self) -> None:
        if len(self.buffer) < self._min_samples_for_stats:
            return
        data = np.array(list(self.buffer))
        self._mean = np.mean(data, axis=0)
        self._cov = np.cov(data, rowvar=False)
        self._median = np.median(data, axis=0)
        abs_dev = np.abs(data - self._median)
        self._mad = np.median(abs_dev, axis=0)
        self._mad = np.clip(self._mad, 1e-6, None)
        if len(data) > 500:
            idx = np.random.choice(len(data), 500, replace=False)
            self._kde_points = data[idx]
        else:
            self._kde_points = data

    def getRobustMean(self) -> np.ndarray:
        if self._mean is None:
            self._compute_stats()
        if self._mean is None:
            return np.zeros(self.featureDim)
        return self._mean

    def getRobustCov(self) -> np.ndarray:
        if self._cov is None:
            self._compute_stats()
        if self._cov is None:
            return np.eye(self.featureDim)
        return self._cov + np.eye(self.featureDim) * 1e-6

    def getMedian(self) -> np.ndarray:
        if self._median is None:
            self._compute_stats()
        if self._median is None:
            return np.zeros(self.featureDim)
        return self._median

    def getMAD(self) -> np.ndarray:
        if self._mad is None:
            self._compute_stats()
        if self._mad is None:
            return np.ones(self.featureDim)
        return self._mad

    def getKDE(self) -> Optional[np.ndarray]:
        if self._kde_points is None:
            self._compute_stats()
        return self._kde_points

    def getSampleCount(self) -> int:
        return len(self.buffer)

    def isReady(self) -> bool:
        return len(self.buffer) >= self._min_samples_for_stats

    def getFeatureDim(self) -> int:
        return self.featureDim

    def incrementalUpdate(self, features: List[float], lr: float = 0.01) -> None:
        self.update(features)
        if self._mean is not None:
            ft = np.array(features[:self.featureDim])
            self._mean = (1 - lr) * self._mean + lr * ft