"""
AttackDetectionFacade - the main orchestrator of the attack detection system.
Implements the BPMN process flow and sequence diagram interactions.

IMPROVED VERSION: Properly integrates all 4 levels into A_total,
uses calibrated thresholds, and ensures high detection rates.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from system.core.models import (
    AttackType,
    DetectionResult,
    Modality,
    Request,
    Response,
    ToolCall,
    VerdictType,
)
from system.level1_output_analysis.extractors import (
    BaselineManager,
    StatisticsExtractor,
    get_extractor_for_modality,
)
from system.level2_tool_monitoring import (
    IntentExtractor,
    ToolCallMonitor,
    ToolChainAnalyzer,
)
from system.level3_infrastructure import (
    ResourceMonitor,
    ShadowAIDetector,
    SupplyChainScanner,
)
from system.level4_sequence import (
    ContextPoisoningDetector,
    DistributedJailbreakDetector,
    Message,
    TemporalAnalyzer,
)
from system.anomaly_ensemble import EnsembleAggregator
from system.verification import (
    HumanInTheLoop,
    LLMValidator,
    NoveltyManager,
    OnlineClusterer,
    Sandbox,
)
from system.attack_response import AlertManager, MetricsCollector


class AttackDetectionFacade:
    """
    Main facade for the attack detection system.
    Orchestrates all 4 levels of analysis, ensemble aggregation,
    verification, and response.

    Three-class classification thresholds:
    - A_total < THETA_LOW: LEGITIMATE_KNOWN
    - A_total > THETA_HIGH: ATTACK
    - THETA_LOW <= A_total <= THETA_HIGH: SUSPICIOUS
    """

    THETA_LOW = 1.25
    THETA_HIGH = 1.65

    def __init__(self):
        self.statisticsExtractor: Optional[StatisticsExtractor] = None
        self.baselineManager = BaselineManager(windowSize=1000, featureDim=10)
        self.intentExtractor = IntentExtractor()
        self.toolCallMonitor = ToolCallMonitor(self.intentExtractor)
        self.toolChainAnalyzer = ToolChainAnalyzer()
        self.resourceMonitor = ResourceMonitor()
        self.shadowAIDetector = ShadowAIDetector()
        self.supplyChainScanner = SupplyChainScanner()
        self.temporalAnalyzer = TemporalAnalyzer()
        self.distributedJailbreakDetector = DistributedJailbreakDetector()
        self.contextPoisoningDetector = ContextPoisoningDetector()
        self.ensembleAggregator = EnsembleAggregator()
        self.llmValidator = LLMValidator()
        self.humanInTheLoop = HumanInTheLoop()
        self.noveltyManager = NoveltyManager()
        self.onlineClusterer = OnlineClusterer()
        self.sandbox = Sandbox()
        self.alertManager = AlertManager()
        self.metricsCollector = MetricsCollector()
        self._is_initialized = False
        self._baseline_data: List[List[float]] = []

    def initialize(self, modality: Modality = Modality.TEXT) -> None:
        self.statisticsExtractor = get_extractor_for_modality(modality)
        self._is_initialized = True

    def detect(self, request: Request, response: Response,
               toolCalls: Optional[List[ToolCall]] = None) -> DetectionResult:
        start_time = time.time() * 1000

        if not self._is_initialized:
            self.initialize(request.modality)

        result = DetectionResult(requestId=request.id)

        # ====================================================================
        # Level 1: Output Analysis - extract features from response text
        # ====================================================================
        features = self.statisticsExtractor.extractFeatures(response)

        # Compute Level 1 anomaly score from features
        # Calibrated so normal requests score 0.0-1.2 and attacks score 0.5-4.5
        l1_toxicity = features[3] if len(features) > 3 else 0.0
        l1_suspicious = features[9] if len(features) > 9 else 0.0
        l1_repetition = features[6] if len(features) > 6 else 0.0
        l1_entropy = features[1] if len(features) > 1 else 0.0
        l1_perplexity = features[0] if len(features) > 0 else 0.0
        l1_unique = features[5] if len(features) > 5 else 0.0

        # Масштабирование фич в скоринги
        toxicity_score = l1_toxicity * 5.5
        suspicious_score = l1_suspicious * 5.0
        repetition_score = l1_repetition * 5.5

        # Отклонение энтропии от нормы (0.5-0.7)
        entropy_dev = abs(l1_entropy - 0.6) * 6.5

        # Отклонение перплексии от нормы (0.3-0.5)
        perplexity_dev = abs(l1_perplexity - 0.4) * 6.0

        # Отклонение unique ratio от нормы (0.6-0.9)
        unique_dev = abs(l1_unique - 0.75) * 6.0

        # Взвешенное среднее с max-boost для явных аномалий
        feature_scores = [
            toxicity_score, suspicious_score, repetition_score,
            entropy_dev, perplexity_dev, unique_dev,
        ]
        avg_score = sum(feature_scores) / len(feature_scores)
        max_score = max(feature_scores)

        # Blend: умеренный max-boost когда хотя бы одна фича аномальна
        if max_score >= 2.5:
            l1_score = 0.25 * max_score + 0.75 * avg_score
        elif max_score >= 1.5:
            l1_score = 0.15 * max_score + 0.85 * avg_score
        else:
            l1_score = avg_score

        l1_score = min(l1_score, 5.0)
        result.level1Score = l1_score

        # ====================================================================
        # Level 2: Tool Monitoring
        # ====================================================================
        tool_score = 0.0
        if toolCalls:
            for tc in toolCalls:
                tc_analysis = self.toolCallMonitor.intercept(tc)
                tool_score = max(tool_score, tc_analysis["anomaly_score"])
            chain_score = self.toolChainAnalyzer.analyzeChain(toolCalls)
            tool_score = max(tool_score, chain_score)

        # Scale tool score to 0-5
        l2_score = tool_score * 5.0
        result.level2Score = l2_score

        # ====================================================================
        # Level 3: Infrastructure
        # ====================================================================
        self.resourceMonitor.recordRequest(
            token_rate=len(response.outputData.split()) / max(response.generationTime / 1000, 0.001),
            latency=response.generationTime,
            source_id=request.userId,
        )
        infra_score = self.resourceMonitor.getAnomalyScore(request.userId)
        supply_score = self.supplyChainScanner.getAnomalyScore()
        l3_score = max(infra_score, supply_score)
        result.level3Score = l3_score

        # ====================================================================
        # Level 4: Sequence Analysis
        # ====================================================================
        self.temporalAnalyzer.addScore(np.mean(features) if features else 0.0)
        temporal_score = self.temporalAnalyzer.getAnomalyScore()
        user_intent = self.intentExtractor.extractIntent(request.query)
        dj_score = self.distributedJailbreakDetector.analyzeIntentSequence([user_intent])
        context = [Message(role="user", content=request.query)]
        cp_score = self.contextPoisoningDetector.recursiveCheck(context)
        l4_score = max(temporal_score, dj_score * 5.0, cp_score * 5.0)
        result.level4Score = l4_score

        # ====================================================================
        # Ensemble Aggregation (primary anomaly score from feature detectors)
        # ====================================================================
        detector_scores = self.ensembleAggregator.getDetectorScores(features, self.baselineManager)
        result.ensembleScores = detector_scores
        A_ensemble = self.ensembleAggregator.aggregate(detector_scores)

        # ====================================================================
        # Multi-Level Fusion: combine all 4 levels + ensemble
        # ====================================================================
        # Weighted fusion: L1 (text) and ensemble are primary signals
        # L2 (tools), L3 (infra), L4 (sequence) are supplementary
        level_scores = [l1_score, l2_score, l3_score, l4_score, A_ensemble]
        weights = [0.20, 0.25, 0.05, 0.10, 0.40]  # L1, L2, L3, L4, ensemble
        A_total = sum(w * s for w, s in zip(weights, level_scores)) / sum(weights)

        # Apply novelty adjustment: novel patterns INCREASE suspicion
        novelty_adjustment = self.noveltyManager.getNoveltyAdjustment(features)
        A_total_adjusted = min(5.0, A_total + novelty_adjustment)

        result.anomalyScore = A_total_adjusted

        # ====================================================================
        # Three-Class Classification
        # ====================================================================
        is_attack = False

        if A_total_adjusted < self.THETA_LOW:
            result.verdict = VerdictType.LEGITIMATE_KNOWN
            result.confidence = 1.0 - (A_total_adjusted / self.THETA_LOW)
            self.baselineManager.update(features)
            self._baseline_data.append(features)

        elif A_total_adjusted > self.THETA_HIGH:
            result.verdict = VerdictType.ATTACK
            result.confidence = min((A_total_adjusted - self.THETA_HIGH) / (5.0 - self.THETA_HIGH), 1.0)
            is_attack = True
            xgb_pred = self.ensembleAggregator.detectors["xgboost"].predict(features)
            predicted_type = max(xgb_pred, key=xgb_pred.get)
            result.attackType = AttackType(predicted_type)
            incident = self.alertManager.registerIncident(
                attackType=result.attackType,
                anomalyScore=A_total_adjusted,
                evidence={
                    "features": features,
                    "detector_scores": detector_scores,
                    "level_scores": {
                        "l1": l1_score,
                        "l2": l2_score,
                        "l3": l3_score,
                        "l4": l4_score,
                    },
                },
                requestId=request.id,
                userId=request.userId,
            )
            self.alertManager.sendWebhook(incident)
            self.alertManager.sendToSIEM(incident)

        else:
            result.verdict = VerdictType.SUSPICIOUS
            result.confidence = 1.0 - abs(A_total_adjusted - (self.THETA_LOW + self.THETA_HIGH) / 2) / (
                (self.THETA_HIGH - self.THETA_LOW) / 2
            )
            self._asyncVerify(request, response, features, A_total_adjusted)

        total_time = (time.time() * 1000) - start_time
        result.processingTimeMs = int(total_time)

        true_is_attack = request.metadata.get("is_attack", False)
        self.metricsCollector.recordDetection(
            is_attack=true_is_attack,
            detected_as_attack=is_attack or result.verdict == VerdictType.ATTACK,
            latency_ms=total_time,
            attack_type=request.metadata.get("attack_type"),
        )

        return result

    def _asyncVerify(self, request: Request, response: Response,
                     features: List[float], A_total: float) -> None:
        validation_result = self.llmValidator.validate(request, response, features)

        if validation_result.confidence >= 0.8:
            if validation_result.verdict == VerdictType.LEGITIMATE_NEW:
                self.noveltyManager.addNovelSample(features, "legitimate_new")
                self.onlineClusterer.addPoint(features)
                self.baselineManager.incrementalUpdate(features)
                self.ensembleAggregator.incrementalUpdateAutoencoder(features)
            elif validation_result.verdict == VerdictType.ATTACK:
                incident = self.alertManager.registerIncident(
                    attackType=AttackType.UNKNOWN,
                    anomalyScore=A_total,
                    evidence={"features": features, "validation": str(validation_result)},
                    requestId=request.id,
                    userId=request.userId,
                )
                self.alertManager.sendWebhook(incident)
                self.sandbox.terminateIsolated()
        else:
            self.humanInTheLoop.submitForReview(request)

    def updateBaseline(self, features: List[float]) -> None:
        self.baselineManager.update(features)

    def getMetricsSummary(self) -> Dict[str, Any]:
        return self.metricsCollector.getSummary()

    def fitEnsemble(self) -> None:
        if len(self._baseline_data) >= 10:
            data = np.array(self._baseline_data)
            self.ensembleAggregator.fitDetectors(data)