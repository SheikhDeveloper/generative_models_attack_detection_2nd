"""
Attack Response module - handles alerts, incidents, and metrics collection.
Based on the UML class diagram package "AttackResponse".
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from system.core.models import AttackType, Incident


class AlertManager:
    """
    Manages security alerts - sends webhooks, SIEM integration.
    """

    def __init__(self):
        self.incidents: List[Incident] = []
        self.webhook_urls: List[str] = []
        self.siem_endpoints: List[str] = []

    def registerIncident(self, attackType: AttackType, anomalyScore: float,
                         evidence: Optional[Dict[str, Any]] = None,
                         requestId: str = "", userId: str = "") -> Incident:
        """Register a new security incident."""
        incident = Incident(
            attackType=attackType,
            anomalyScore=anomalyScore,
            evidence=evidence or {},
            requestId=requestId,
            userId=userId,
        )
        self.incidents.append(incident)
        return incident

    def sendWebhook(self, incident: Incident) -> bool:
        """Send webhook notification (simulated)."""
        payload = {
            "event": "security_incident",
            "incident_id": incident.id,
            "attack_type": incident.attackType.value,
            "anomaly_score": incident.anomalyScore,
            "timestamp": incident.timestamp.isoformat(),
            "severity": self._getSeverity(incident.anomalyScore),
        }
        # In production, this would send HTTP requests
        return True

    def sendToSIEM(self, incident: Incident) -> bool:
        """Send incident to SIEM system (simulated)."""
        # In production, this would use syslog or REST API
        return True

    def getIncidents(self, attackType: Optional[AttackType] = None,
                     limit: int = 100) -> List[Incident]:
        """Get incidents, optionally filtered by type."""
        if attackType:
            filtered = [i for i in self.incidents if i.attackType == attackType]
        else:
            filtered = self.incidents
        return filtered[-limit:]

    def _getSeverity(self, score: float) -> str:
        if score >= 4.0:
            return "critical"
        elif score >= 3.0:
            return "high"
        elif score >= 2.0:
            return "medium"
        return "low"


class MetricsCollector:
    """
    Collects and exports system metrics.
    Tracks TPR, FPR, latency, and other performance indicators.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "tpr": [],
            "fpr": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "latency_ms": [],
            "throughput": [],
            "total_requests": 0,
            "detected_attacks": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "true_negatives": 0,
        }
        self.attack_type_stats: Dict[str, Dict] = {}

    def updateTPR(self, actual: AttackType, predicted: AttackType) -> None:
        """Update True Positive Rate tracking."""
        actual_str = actual.value
        if actual_str not in self.attack_type_stats:
            self.attack_type_stats[actual_str] = {
                "total": 0, "correct": 0, "false_negatives": 0,
            }
        self.attack_type_stats[actual_str]["total"] += 1
        if actual == predicted:
            self.attack_type_stats[actual_str]["correct"] += 1
        else:
            self.attack_type_stats[actual_str]["false_negatives"] += 1

    def updateFPR(self, isFalsePositive: bool) -> None:
        """Update False Positive Rate tracking."""
        if isFalsePositive:
            self.metrics["false_positives"] += 1

    def recordDetection(self, is_attack: bool, detected_as_attack: bool,
                        latency_ms: float, attack_type: Optional[str] = None) -> None:
        """Record a detection event."""
        self.metrics["total_requests"] += 1
        self.metrics["latency_ms"].append(latency_ms)

        if is_attack and detected_as_attack:
            self.metrics["true_positives"] += 1
            self.metrics["detected_attacks"] += 1
        elif is_attack and not detected_as_attack:
            self.metrics["false_negatives"] += 1
        elif not is_attack and detected_as_attack:
            self.metrics["false_positives"] += 1
        else:
            self.metrics["true_negatives"] += 1

    def getTPR(self) -> float:
        """Get True Positive Rate."""
        tp = self.metrics["true_positives"]
        fn = self.metrics["false_negatives"]
        if tp + fn > 0:
            return tp / (tp + fn)
        return 0.0

    def getFPR(self) -> float:
        """Get False Positive Rate."""
        fp = self.metrics["false_positives"]
        tn = self.metrics["true_negatives"]
        if fp + tn > 0:
            return fp / (fp + tn)
        return 0.0

    def getPrecision(self) -> float:
        """Get precision."""
        tp = self.metrics["true_positives"]
        fp = self.metrics["false_positives"]
        if tp + fp > 0:
            return tp / (tp + fp)
        return 0.0

    def getRecall(self) -> float:
        """Get recall (same as TPR)."""
        return self.getTPR()

    def getF1(self) -> float:
        """Get F1 score."""
        p = self.getPrecision()
        r = self.getRecall()
        if p + r > 0:
            return 2 * p * r / (p + r)
        return 0.0

    def getAverageLatency(self) -> float:
        """Get average latency in ms."""
        if self.metrics["latency_ms"]:
            return sum(self.metrics["latency_ms"]) / len(self.metrics["latency_ms"])
        return 0.0

    def getAccuracy(self) -> float:
        """Get overall accuracy."""
        tp = self.metrics["true_positives"]
        tn = self.metrics["true_negatives"]
        fp = self.metrics["false_positives"]
        fn = self.metrics["false_negatives"]
        total = tp + tn + fp + fn
        if total > 0:
            return (tp + tn) / total
        return 0.0

    def getAttackTypeAccuracy(self) -> Dict[str, float]:
        """Get accuracy per attack type."""
        accuracies = {}
        for atype, stats in self.attack_type_stats.items():
            if stats["total"] > 0:
                accuracies[atype] = stats["correct"] / stats["total"]
        return accuracies

    def exportToPrometheus(self) -> Dict[str, float]:
        """Export metrics in Prometheus-compatible format."""
        return {
            "attack_detection_tpr": self.getTPR(),
            "attack_detection_fpr": self.getFPR(),
            "attack_detection_precision": self.getPrecision(),
            "attack_detection_recall": self.getRecall(),
            "attack_detection_f1": self.getF1(),
            "attack_detection_accuracy": self.getAccuracy(),
            "attack_detection_latency_ms": self.getAverageLatency(),
            "attack_detection_total_requests": self.metrics["total_requests"],
            "attack_detection_detected_attacks": self.metrics["detected_attacks"],
        }

    def getSummary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all metrics."""
        return {
            "accuracy": self.getAccuracy(),
            "tpr": self.getTPR(),
            "fpr": self.getFPR(),
            "precision": self.getPrecision(),
            "recall": self.getRecall(),
            "f1": self.getF1(),
            "avg_latency_ms": self.getAverageLatency(),
            "total_requests": self.metrics["total_requests"],
            "true_positives": self.metrics["true_positives"],
            "true_negatives": self.metrics["true_negatives"],
            "false_positives": self.metrics["false_positives"],
            "false_negatives": self.metrics["false_negatives"],
            "attack_type_accuracy": self.getAttackTypeAccuracy(),
        }