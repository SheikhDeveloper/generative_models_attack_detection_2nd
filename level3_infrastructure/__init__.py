"""
Level 3: Infrastructure Monitoring - monitors resource usage,
detects shadow AI, and scans supply chain.
Based on the UML class diagram package "Level3_InfrastructureMonitoring".
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from system.core.models import Connection, Dependency


class ResourceMonitor:
    """
    Monitors infrastructure metrics:
    - Token rate, request frequency, latency
    - Anomaly detection in resource usage
    """

    def __init__(self, windowSize: int = 60):
        self.windowSize = windowSize
        self.token_rates: Deque[float] = deque(maxlen=windowSize)
        self.request_frequencies: Deque[float] = deque(maxlen=windowSize)
        self.latencies: Deque[float] = deque(maxlen=windowSize)
        self.source_stats: Dict[str, Dict] = {}  # per-source statistics

    def getTokenRate(self) -> float:
        """Get current token rate (tokens/sec)."""
        if not self.token_rates:
            return 0.0
        return np.mean(self.token_rates)

    def getRequestFrequency(self) -> float:
        """Get current request frequency (req/sec)."""
        if not self.request_frequencies:
            return 0.0
        return np.mean(self.request_frequencies)

    def getLatency(self) -> float:
        """Get current average latency (ms)."""
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies)

    def getAnomalyScore(self, source_id: str = "default") -> float:
        """
        Compute infrastructure anomaly score for a given source.
        Returns score 0-5.
        """
        score = 0.0

        # Token rate anomaly
        if len(self.token_rates) >= 10:
            rates = np.array(self.token_rates)
            mean_r = np.mean(rates)
            std_r = np.std(rates) + 1e-6
            current_rate = rates[-1]
            z_token = abs(current_rate - mean_r) / std_r
            if z_token > 4:
                score += 2.0
            elif z_token > 3:
                score += 1.0

        # Request frequency anomaly
        if len(self.request_frequencies) >= 10:
            freqs = np.array(self.request_frequencies)
            mean_f = np.mean(freqs)
            std_f = np.std(freqs) + 1e-6
            current_freq = freqs[-1]
            z_freq = abs(current_freq - mean_f) / std_f
            if z_freq > 4:
                score += 2.0
            elif z_freq > 3:
                score += 1.0

        # Latency anomaly
        if len(self.latencies) >= 10:
            lats = np.array(self.latencies)
            mean_l = np.mean(lats)
            std_l = np.std(lats) + 1e-6
            current_lat = lats[-1]
            z_lat = abs(current_lat - mean_l) / std_l
            if z_lat > 4:
                score += 1.0
            elif z_lat > 3:
                score += 0.5

        # Per-source rate limiting check
        if source_id in self.source_stats:
            stats = self.source_stats[source_id]
            if stats["request_count"] > 100:
                rate = stats["request_count"] / max(stats["time_span"], 1)
                if rate > 50:  # >50 req/sec from single source
                    score += 1.0

        return min(score, 5.0)

    def recordRequest(self, token_rate: float, latency: float, source_id: str = "default") -> None:
        """Record metrics for a request."""
        self.token_rates.append(token_rate)
        self.latencies.append(latency)

        # Update request frequency (simulated)
        self.request_frequencies.append(random.gauss(10, 3))

        # Update per-source stats
        if source_id not in self.source_stats:
            self.source_stats[source_id] = {"request_count": 0, "time_span": 1}
        self.source_stats[source_id]["request_count"] += 1


class ShadowAIDetector:
    """
    Detects shadow AI usage by scanning processes
    and identifying suspicious LLM-related activity.
    """

    def __init__(self):
        self.known_processes: Set[str] = {
            "python3", "python", "node", "java", "nginx", "postgres",
            "redis-server", "docker", "containerd", "sshd",
        }
        self.llm_frameworks: List[str] = [
            "transformers", "langchain", "llama", "pytorch", "tensorflow",
            "vllm", "tgi", "text-generation", "ollama", "llama.cpp",
        ]

    def scanProcesses(self) -> List[str]:
        """Scan running processes for suspicious activity."""
        # Simulated process scan
        suspicious = []
        processes = self._get_running_processes()

        for proc in processes:
            if self.isSuspicious(proc):
                suspicious.append(proc)

        return suspicious

    def isSuspicious(self, proc: str) -> bool:
        """Check if a process is suspicious."""
        proc_lower = proc.lower()

        # Check for unknown LLM-related processes
        if any(fw in proc_lower for fw in self.llm_frameworks):
            if proc_lower not in self.known_processes:
                return True

        # Check for crypto miners or other malicious processes
        mining_keywords = ["miner", "crypto", "coin", "xmr", "ethminer"]
        if any(kw in proc_lower for kw in mining_keywords):
            return True

        return False

    def _get_running_processes(self) -> List[str]:
        """Simulate getting running processes."""
        return list(self.known_processes) + ["python3 -m transformers", "node server.js"]


class SupplyChainScanner:
    """
    Scans supply chain integrity:
    - Checks dependency hashes
    - Monitors outbound connections
    """

    def __init__(self):
        self.known_dependencies: Dict[str, Dependency] = {}
        self.connection_whitelist: Set[str] = {
            "api.openai.com", "api.anthropic.com", "api.google.com",
            "registry.npmjs.org", "pypi.org", "files.pythonhosted.org",
            "github.com", "raw.githubusercontent.com",
        }
        self.outbound_connections: List[Connection] = []

    def checkIntegrity(self, dependencies: List[Dependency]) -> bool:
        """
        Check integrity of dependencies against known hashes.
        Returns True if all dependencies are intact.
        """
        for dep in dependencies:
            if dep.name in self.known_dependencies:
                expected = self.known_dependencies[dep.name]
                if dep.current_hash != expected.expected_hash:
                    return False
            else:
                # New dependency - register it
                self.known_dependencies[dep.name] = dep
        return True

    def monitorOutboundConnections(self) -> List[Connection]:
        """
        Monitor and return suspicious outbound connections.
        """
        # Simulated connection monitoring
        suspicious = []
        for conn in self.outbound_connections:
            if conn.dest_ip not in self.connection_whitelist:
                if conn.dest_port not in (80, 443, 8080):
                    suspicious.append(conn)
        return suspicious

    def addConnection(self, connection: Connection) -> None:
        """Record an outbound connection."""
        self.outbound_connections.append(connection)

    def getAnomalyScore(self) -> float:
        """
        Compute supply chain anomaly score.
        Returns score 0-5.
        """
        score = 0.0

        # Check for suspicious connections
        suspicious = self.monitorOutboundConnections()
        if len(suspicious) > 5:
            score += 2.0
        elif len(suspicious) > 2:
            score += 1.0

        # Check for unknown dependencies
        unknown_deps = sum(
            1 for dep in self.known_dependencies.values()
            if dep.current_hash != dep.expected_hash
        )
        if unknown_deps > 0:
            score += 3.0

        return min(score, 5.0)