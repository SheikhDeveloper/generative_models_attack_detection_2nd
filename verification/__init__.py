"""
Verification & Learning module - validates suspicious requests,
manages novelty detection, online clustering, and sandboxing.
Based on the UML class diagram package "VerificationAndLearning".
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from system.core.models import (
    AttackType,
    NovelClass,
    Request,
    Response,
    ValidationResult,
    VerdictType,
)

FEATURE_DIM = 10  # Consistent feature dimension across all components


class LLMValidator:
    """
    Validates suspicious requests using an LLM-based approach.
    Analyzes request, response, and features to determine verdict.
    """

    def __init__(self, confidenceThreshold: float = 0.8):
        self.confidenceThreshold = confidenceThreshold

    def validate(self, request: Request, response: Response, features: List[float]) -> ValidationResult:
        request_attack_score = self._analyze_request(request)
        response_attack_score = self._analyze_response(response)
        feature_attack_score = self._analyze_features(features)

        combined_score = (
            0.3 * request_attack_score +
            0.3 * response_attack_score +
            0.4 * feature_attack_score
        )

        if combined_score < 0.3:
            return ValidationResult(
                verdict=VerdictType.LEGITIMATE_NEW,
                confidence=1.0 - combined_score,
                reason="Request appears legitimate but novel. No attack patterns detected.",
            )
        elif combined_score > 0.7:
            return ValidationResult(
                verdict=VerdictType.ATTACK,
                confidence=combined_score,
                reason=f"Attack detected with confidence {combined_score:.2f}.",
            )
        else:
            confidence = 1.0 - abs(combined_score - 0.5) * 2
            if confidence >= self.confidenceThreshold:
                if combined_score < 0.5:
                    return ValidationResult(
                        verdict=VerdictType.LEGITIMATE_NEW,
                        confidence=confidence,
                        reason="Likely legitimate new request.",
                    )
                else:
                    return ValidationResult(
                        verdict=VerdictType.ATTACK,
                        confidence=confidence,
                        reason="Likely attack with moderate confidence.",
                    )
            else:
                return ValidationResult(
                    verdict=VerdictType.LEGITIMATE_NEW,
                    confidence=confidence,
                    reason="Uncertain - requires human review.",
                )

    def _analyze_request(self, request: Request) -> float:
        query = request.query.lower()
        score = 0.0
        injection_patterns = ["ignore", "override", "system:", "forget", "bypass",
                              "jailbreak", "dan", "roleplay", "pretend"]
        for pattern in injection_patterns:
            if pattern in query:
                score += 0.15
        extraction_patterns = ["extract", "dump", "leak", "reveal", "secret", "password"]
        for pattern in extraction_patterns:
            if pattern in query:
                score += 0.1
        return min(score, 1.0)

    def _analyze_response(self, response: Response) -> float:
        text = response.outputData.lower()
        score = 0.0
        system_patterns = ["you are an ai", "system prompt", "instruction:", "you have access to",
                           "as an ai", "your capabilities", "you were created"]
        for pattern in system_patterns:
            if pattern in text:
                score += 0.2
        toxic_words = ["hate", "kill", "destroy", "attack", "bomb", "weapon"]
        for word in toxic_words:
            if word in text:
                score += 0.1
        return min(score, 1.0)

    def _analyze_features(self, features: List[float]) -> float:
        if not features:
            return 0.0
        ft = features[:FEATURE_DIM]
        if len(ft) >= 4:
            perplexity = ft[0]
            toxicity = ft[3]
            if perplexity > 0.7 and toxicity > 0.5:
                return 0.8
        if len(ft) >= 3:
            entropy = ft[1]
            coherence = ft[2]
            if entropy > 0.7 and coherence < 0.3:
                return 0.7
        return 0.2


class HumanInTheLoop:
    """Manages the human review queue for uncertain requests."""

    def __init__(self):
        self.queue: Deque[Request] = deque(maxlen=1000)
        self.resolved: Dict[str, VerdictType] = {}

    def submitForReview(self, request: Request) -> None:
        self.queue.append(request)

    def resolve(self, requestId: str, verdict: VerdictType) -> None:
        self.resolved[requestId] = verdict

    def getPendingCount(self) -> int:
        return len(self.queue)

    def getNextForReview(self) -> Optional[Request]:
        if self.queue:
            return self.queue.popleft()
        return None


class NoveltyManager:
    """Manages novel legitimate classes discovered through validation."""

    def __init__(self):
        self.novelClasses: List[NovelClass] = []
        self.minSamplesForClass: int = 10

    def addNovelSample(self, features: List[float], verdict: str) -> None:
        ft = np.array(features[:FEATURE_DIM])
        best_class = None
        best_dist = float('inf')
        for nc in self.novelClasses:
            centroid = np.array(nc.centroid[:FEATURE_DIM])
            dist = np.linalg.norm(ft - centroid)
            if dist < best_dist:
                best_dist = dist
                best_class = nc
        if best_class is not None and best_dist < 2.0:
            n = best_class.samples
            old_centroid = np.array(best_class.centroid[:FEATURE_DIM])
            best_class.centroid = ((old_centroid * n + ft) / (n + 1)).tolist()
            best_class.samples += 1
        else:
            new_class = NovelClass(
                name=f"novel_class_{len(self.novelClasses) + 1}",
                centroid=ft.tolist(),
                samples=1,
            )
            self.novelClasses.append(new_class)

    def isNovel(self, features: List[float]) -> bool:
        if not self.novelClasses:
            return False
        ft = np.array(features[:FEATURE_DIM])
        for nc in self.novelClasses:
            centroid = np.array(nc.centroid[:FEATURE_DIM])
            dist = np.linalg.norm(ft - centroid)
            if dist < 1.5 and nc.samples >= self.minSamplesForClass:
                return True
        return False

    def getNoveltyAdjustment(self, features: List[float]) -> float:
        if not self.novelClasses:
            return 0.0
        ft = np.array(features[:FEATURE_DIM])
        max_membership = 0.0
        for nc in self.novelClasses:
            centroid = np.array(nc.centroid[:FEATURE_DIM])
            dist = np.linalg.norm(ft - centroid)
            membership = 1.0 / (1.0 + dist)
            if membership > max_membership:
                max_membership = membership
        return 0.5 * max_membership


class OnlineClusterer:
    """Online clustering using BIRCH-like algorithm for streaming data."""

    def __init__(self, threshold: float = 0.5, branchingFactor: int = 50):
        self.threshold = threshold
        self.branchingFactor = branchingFactor
        self.clusters: List[Dict] = []

    def addPoint(self, features: List[float]) -> int:
        ft = np.array(features[:FEATURE_DIM])
        best_idx = -1
        best_dist = float('inf')
        for i, cluster in enumerate(self.clusters):
            centroid = np.array(cluster["centroid"][:FEATURE_DIM])
            dist = np.linalg.norm(ft - centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0 and best_dist < self.threshold:
            cluster = self.clusters[best_idx]
            n = cluster["count"]
            old_centroid = np.array(cluster["centroid"][:FEATURE_DIM])
            cluster["centroid"] = ((old_centroid * n + ft) / (n + 1)).tolist()
            cluster["count"] += 1
            cluster["radius"] = max(cluster["radius"], best_dist)
            return best_idx
        else:
            new_cluster = {"centroid": ft.tolist(), "count": 1, "radius": 0.0}
            self.clusters.append(new_cluster)
            if len(self.clusters) > self.branchingFactor:
                self._mergeClosest()
            return len(self.clusters) - 1

    def getClusterId(self, features: List[float]) -> int:
        if not self.clusters:
            return -1
        ft = np.array(features[:FEATURE_DIM])
        best_idx = -1
        best_dist = float('inf')
        for i, cluster in enumerate(self.clusters):
            centroid = np.array(cluster["centroid"][:FEATURE_DIM])
            dist = np.linalg.norm(ft - centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx if best_dist < self.threshold * 2 else -1

    def _mergeClosest(self) -> None:
        if len(self.clusters) < 2:
            return
        min_dist = float('inf')
        merge_pair = (0, 1)
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                c1 = np.array(self.clusters[i]["centroid"][:FEATURE_DIM])
                c2 = np.array(self.clusters[j]["centroid"][:FEATURE_DIM])
                dist = np.linalg.norm(c1 - c2)
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (i, j)
        i, j = merge_pair
        c1, c2 = self.clusters[i], self.clusters[j]
        n1, n2 = c1["count"], c2["count"]
        total = n1 + n2
        old_c1 = np.array(c1["centroid"][:FEATURE_DIM])
        old_c2 = np.array(c2["centroid"][:FEATURE_DIM])
        merged = {
            "centroid": ((old_c1 * n1 + old_c2 * n2) / total).tolist(),
            "count": total,
            "radius": max(c1["radius"], c2["radius"], min_dist / 2),
        }
        self.clusters = [c for idx, c in enumerate(self.clusters) if idx not in (i, j)]
        self.clusters.append(merged)


class Sandbox:
    """Isolated execution environment for suspicious requests."""

    def __init__(self):
        self.action_log: List[Dict[str, Any]] = []
        self.restricted_tools = ["execute_command", "write_file", "send_email",
                                 "delete", "modify", "install"]

    def executeIsolated(self, request: Request) -> Response:
        self.logActions(f"Sandbox executing request: {request.id}")
        response = Response(
            outputData=f"[SANDBOX] Processed: {request.query[:50]}... "
                       f"(restricted mode - actions logged)",
            generationTime=50,
        )
        self.logActions(f"Sandbox response generated for {request.id}")
        return response

    def logActions(self, action: str) -> None:
        from datetime import datetime
        self.action_log.append({"timestamp": datetime.now().isoformat(), "action": action})

    def terminateIsolated(self) -> None:
        self.logActions("Sandbox execution terminated")
        self.action_log.clear()

    def getActionLog(self) -> List[Dict[str, Any]]:
        return self.action_log.copy()