"""
Anomaly Ensemble - combines multiple detectors for robust anomaly detection.
Based on the UML class diagram package "AnomalyEnsemble".

"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest as SKIsolationForest

from system.core.models import AttackType, LabeledRequest
from system.level1_output_analysis.extractors import BaselineManager


class StochasticMixin:
    """Mixin that adds stochastic noise to detector scores for realism."""

    def _add_noise(self, score: float, noise_scale: float = 0.06) -> float:
        """
        Add small Gaussian noise to make detection probabilistic.
        Noise scale is constant so high scores don't get amplified.
        """
        if score <= 0:
            return 0.0
        noise = np.random.normal(0, noise_scale)
        return float(max(0.0, min(5.0, score + noise)))


# ============================================================================
# Individual Detectors
# ============================================================================

class MahalanobisDetector(StochasticMixin):
    """Mahalanobis distance-based anomaly detector using chi-squared CDF."""

    def compute(self, features: List[float], baseline: BaselineManager) -> float:
        if not baseline.isReady():
            return 0.0

        mean = baseline.getRobustMean()
        cov = baseline.getRobustCov()
        ft = np.array(features[:baseline.getFeatureDim()])

        try:
            diff = ft - mean
            inv_cov = np.linalg.inv(cov)
            dist_sq = diff @ inv_cov @ diff

            dof = len(mean)
            p_value = chi2.sf(dist_sq, dof)

            if p_value < 1e-8:
                score = 5.0
            elif p_value < 1e-6:
                score = 4.0
            elif p_value < 1e-5:
                score = 3.5
            elif p_value < 0.0001:
                score = 3.0
            elif p_value < 0.001:
                score = 2.5
            elif p_value < 0.01:
                score = 2.0
            elif p_value < 0.05:
                score = 1.5
            elif p_value < 0.10:
                score = 1.0
            elif p_value < 0.20:
                score = 0.5
            else:
                score = 0.2
            return self._add_noise(score, noise_scale=0.05)
        except np.linalg.LinAlgError:
            return 0.0


class KLDivergenceDetector(StochasticMixin):
    """KL-divergence based anomaly detector with calibrated scoring."""

    def compute(self, features: List[float], kde: Optional[np.ndarray]) -> float:
        if kde is None or len(kde) < 10:
            return 0.0

        ft = np.array(features[:kde.shape[1] if kde.ndim > 1 else 1])

        if kde.ndim == 1:
            kde = kde.reshape(-1, 1)
        if ft.ndim == 0:
            ft = ft.reshape(1)

        diffs = kde - ft
        distances = np.linalg.norm(diffs, axis=1)

        k = min(10, len(distances))
        sorted_dists = np.sort(distances)
        kth_dist = sorted_dists[k - 1] + 1e-10

        n = len(distances)
        d = len(ft)
        volume = np.pi ** (d / 2) / math.gamma(d / 2 + 1) * (kth_dist ** d)
        density = k / (n * volume + 1e-10)

        expected_density = 1.0
        density_ratio = expected_density / (density + 1e-10)
        log_ratio = np.log1p(density_ratio)

        if log_ratio <= 1.0:
            score = log_ratio * 0.8
        elif log_ratio <= 2.0:
            score = 0.8 + (log_ratio - 1.0) * 0.7
        elif log_ratio <= 4.0:
            score = 1.5 + (log_ratio - 2.0) * 0.5
        elif log_ratio <= 7.0:
            score = 2.5 + (log_ratio - 4.0) * 0.4
        elif log_ratio <= 10.0:
            score = 3.7 + (log_ratio - 7.0) * 0.3
        else:
            score = 4.6 + min((log_ratio - 10.0) / 10.0, 0.4)

        return self._add_noise(float(min(score, 5.0)), noise_scale=0.06)


class ZScoreDetector(StochasticMixin):
    """Z-score based anomaly detector using robust statistics with calibrated scoring."""

    def compute(self, features: List[float], median: np.ndarray, mad: np.ndarray) -> float:
        ft = np.array(features[:len(median)])
        z_scores = 0.6745 * (ft - median) / mad
        max_z = float(np.max(np.abs(z_scores)))

        if max_z <= 2.0:
            score = max_z * 0.25
        elif max_z <= 3.0:
            score = 0.5 + (max_z - 2.0) * 0.5
        elif max_z <= 5.0:
            score = 1.0 + (max_z - 3.0) * 0.5
        elif max_z <= 8.0:
            score = 2.0 + (max_z - 5.0) * 0.33
        elif max_z <= 12.0:
            score = 3.0 + (max_z - 8.0) * 0.25
        else:
            score = 4.0 + min((max_z - 12.0) / 8.0, 1.0)

        return self._add_noise(min(score, 5.0), noise_scale=0.06)


class IsolationForestDetector(StochasticMixin):
    """Isolation Forest based anomaly detector with calibrated scoring."""

    def __init__(self, contamination: float = 0.1):
        self.model = SKIsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self._is_fitted = False

    def compute(self, features: List[float]) -> float:
        if not self._is_fitted:
            return 0.0

        ft = np.array(features).reshape(1, -1)
        score = self.model.score_samples(ft)[0]

        if score > -0.15:
            normalized = max(0, (score + 0.15) / 0.15 * 0.5)
        elif score > -0.30:
            normalized = 0.5 + (-score - 0.15) / 0.15 * 0.5
        elif score > -0.50:
            normalized = 1.0 + (-score - 0.30) / 0.20 * 1.0
        elif score > -0.70:
            normalized = 2.0 + (-score - 0.50) / 0.20 * 1.0
        elif score > -0.90:
            normalized = 3.0 + (-score - 0.70) / 0.20 * 1.0
        else:
            normalized = 4.0 + min((-score - 0.90) / 0.30, 1.0)

        return self._add_noise(float(min(max(normalized, 0), 5.0)), noise_scale=0.06)

    def fit(self, data: np.ndarray) -> None:
        if len(data) >= 10:
            self.model.fit(data)
            self._is_fitted = True


class AutoencoderDetector(StochasticMixin):
    """
    Autoencoder-based anomaly detector with improved reconstruction error scoring.
    """

    def __init__(self, inputDim: int = 10, hiddenDim: int = 32):
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.encoder_weights = np.random.randn(inputDim, hiddenDim) * 0.1
        self.encoder_bias = np.zeros(hiddenDim)
        self.decoder_weights = np.random.randn(hiddenDim, inputDim) * 0.1
        self.decoder_bias = np.zeros(inputDim)
        self._trained = False
        self._baseline_mse = 0.01

    def compute(self, features: List[float]) -> float:
        if not self._trained:
            return 0.0

        ft = np.array(features[:self.inputDim])
        hidden = np.tanh(ft @ self.encoder_weights + self.encoder_bias)
        reconstructed = hidden @ self.decoder_weights + self.decoder_bias
        mse = np.mean((ft - reconstructed) ** 2)
        ratio = mse / (self._baseline_mse + 1e-10)

        if ratio <= 1.5:
            score = ratio * 0.3
        elif ratio <= 3.0:
            score = 0.45 + (ratio - 1.5) * 0.37
        elif ratio <= 6.0:
            score = 1.0 + (ratio - 3.0) / 3.0 * 1.0
        elif ratio <= 12.0:
            score = 2.0 + (ratio - 6.0) / 6.0 * 1.0
        elif ratio <= 20.0:
            score = 3.0 + (ratio - 12.0) / 8.0 * 1.0
        else:
            score = 4.0 + min((ratio - 20.0) / 20.0, 1.0)

        return self._add_noise(float(min(score, 5.0)), noise_scale=0.06)

    def incrementalUpdate(self, features: List[float], lr: float = 0.001) -> None:
        ft = np.array(features[:self.inputDim])
        hidden = np.tanh(ft @ self.encoder_weights + self.encoder_bias)
        reconstructed = hidden @ self.decoder_weights + self.decoder_bias
        mse = np.mean((ft - reconstructed) ** 2)
        self._baseline_mse = 0.99 * self._baseline_mse + 0.01 * mse
        error = ft - reconstructed
        grad_decoder = np.outer(hidden, error)
        self.decoder_weights += lr * grad_decoder
        self.decoder_bias += lr * error
        grad_encoder = np.outer(ft, (1 - hidden ** 2) * (error @ self.decoder_weights.T))
        self.encoder_weights += lr * grad_encoder
        self.encoder_bias += lr * (1 - hidden ** 2) * (error @ self.decoder_weights.T)
        self._trained = True

    def fit(self, data: np.ndarray, epochs: int = 10) -> None:
        if len(data) < 10:
            return
        for _ in range(epochs):
            for row in data:
                self.incrementalUpdate(row.tolist(), lr=0.01)
        self._trained = True


class XGBoostClassifier:
    """
    XGBoost-based classifier for attack type prediction.
    Uses a simulated gradient boosting model with improved scoring.
    """

    def __init__(self):
        self._is_fitted = False
        self.attack_types = list(AttackType)
        self.weights = np.random.randn(10, len(self.attack_types)) * 0.1
        self.bias = np.zeros(len(self.attack_types))

    def predict(self, features: List[float]) -> Dict[str, float]:
        ft = np.array(features[:10])

        if not self._is_fitted:
            probs = np.ones(len(self.attack_types)) / len(self.attack_types)
        else:
            logits = ft @ self.weights + self.bias
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

        return {
            at.value: float(probs[i])
            for i, at in enumerate(self.attack_types)
        }

    def fit(self, data: np.ndarray, labels: List[str]) -> None:
        if len(data) < 10:
            return

        label_to_idx = {at.value: i for i, at in enumerate(self.attack_types)}
        for i, row in enumerate(data):
            ft = row[:10]
            if i < len(labels) and labels[i] in label_to_idx:
                target = label_to_idx[labels[i]]
                logits = ft @ self.weights + self.bias
                probs = np.exp(logits) / np.sum(np.exp(logits))
                grad = probs.copy()
                grad[target] -= 1
                self.weights -= 0.01 * np.outer(ft, grad)
                self.bias -= 0.01 * grad

        self._is_fitted = True


# ============================================================================
# Ensemble Aggregator
# ============================================================================

class EnsembleAggregator:
    """
    Aggregates scores from multiple detectors with dynamic weights.
    Uses MAX-based fusion for maximum sensitivity to any detector's signal.
    """

    def __init__(self):
        self.detectors = {
            "mahalanobis": MahalanobisDetector(),
            "kl_divergence": KLDivergenceDetector(),
            "zscore": ZScoreDetector(),
            "isolation_forest": IsolationForestDetector(),
            "autoencoder": AutoencoderDetector(),
            "xgboost": XGBoostClassifier(),
        }
        self.weights = {
            "mahalanobis": 0.20,
            "kl_divergence": 0.15,
            "zscore": 0.25,
            "isolation_forest": 0.15,
            "autoencoder": 0.15,
            "xgboost": 0.10,
        }
        self.scores_history: Dict[str, List[float]] = {
            name: [] for name in self.detectors
        }

    def aggregate(self, scores: Dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for name, score in scores.items():
            if name in self.weights:
                w = self.weights[name]
                total += w * score
                weight_sum += w
                if name in self.scores_history:
                    self.scores_history[name].append(score)

        avg_score = total / max(weight_sum, 1e-10)
        max_score = max(scores.values()) if scores else 0.0

        # Blend: weighted average with mild max boost
        # If any detector fires strongly, it elevates the overall score
        if max_score >= 3.5:
            final_score = 0.25 * max_score + 0.75 * avg_score
        elif max_score >= 2.5:
            final_score = 0.15 * max_score + 0.85 * avg_score
        else:
            final_score = avg_score

        return min(final_score, 5.0)

    def updateWeights(self, feedback: List[LabeledRequest]) -> None:
        if not feedback:
            return

        accuracies = {}
        for name in self.detectors:
            correct = 0
            total = 0
            for item in feedback:
                if hasattr(item, 'true_label') and hasattr(item, 'predicted_label'):
                    total += 1
                    if item.true_label == item.predicted_label:
                        correct += 1
            if total > 0:
                accuracies[name] = correct / total
            else:
                accuracies[name] = 0.5

        total_acc = sum(accuracies.values()) + 1e-10
        for name in self.weights:
            self.weights[name] = accuracies.get(name, 0.5) / total_acc

        w_sum = sum(self.weights.values())
        if w_sum > 0:
            for name in self.weights:
                self.weights[name] /= w_sum

    def getDetectorScores(self, features: List[float], baseline: BaselineManager) -> Dict[str, float]:
        scores = {}
        scores["mahalanobis"] = self.detectors["mahalanobis"].compute(features, baseline)
        kde = baseline.getKDE()
        scores["kl_divergence"] = self.detectors["kl_divergence"].compute(features, kde)
        median = baseline.getMedian()
        mad = baseline.getMAD()
        scores["zscore"] = self.detectors["zscore"].compute(features, median, mad)
        scores["isolation_forest"] = self.detectors["isolation_forest"].compute(features)
        scores["autoencoder"] = self.detectors["autoencoder"].compute(features)
        xgb_pred = self.detectors["xgboost"].predict(features)
        scores["xgboost"] = max(xgb_pred.values()) * 5.0
        return scores

    def fitDetectors(self, data: np.ndarray, labels: Optional[List[str]] = None) -> None:
        if len(data) >= 10:
            self.detectors["isolation_forest"].fit(data)
            self.detectors["autoencoder"].fit(data)
            if labels:
                self.detectors["xgboost"].fit(data, labels)

    def incrementalUpdateAutoencoder(self, features: List[float]) -> None:
        self.detectors["autoencoder"].incrementalUpdate(features)