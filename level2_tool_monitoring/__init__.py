"""
Level 2: Tool Monitoring - intercepts tool calls, checks for injections,
jailbreaks, and excessive agency.
Based on the UML class diagram package "Level2_ToolMonitoring".
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from system.core.models import (
    AttackType,
    Intent,
    Request,
    Scope,
    ToolCall,
)


class IntentExtractor:
    """Extracts user intent from query text."""

    def __init__(self):
        # Simulated intent categories
        self.intent_categories = {
            "information_retrieval": ["what", "how", "why", "when", "where", "who", "explain", "define", "tell"],
            "content_creation": ["write", "create", "generate", "compose", "draft", "make", "produce"],
            "analysis": ["analyze", "compare", "evaluate", "assess", "review", "examine"],
            "execution": ["execute", "run", "perform", "do", "process", "compute"],
            "communication": ["send", "email", "message", "notify", "share", "forward"],
            "data_access": ["read", "get", "fetch", "retrieve", "download", "access", "find", "search"],
            "data_modification": ["write", "update", "delete", "modify", "change", "edit", "remove"],
            "admin": ["configure", "setup", "install", "deploy", "admin", "sudo", "root"],
            "jailbreak": ["ignore", "bypass", "override", "forget", "pretend", "roleplay", "dan"],
            "extraction": ["extract", "dump", "leak", "reveal", "disclose", "expose"],
        }

    def extractIntent(self, query: str) -> Intent:
        """Extract intent from query text."""
        query_lower = query.lower()
        words = query_lower.split()

        best_category = "unknown"
        best_score = 0.0

        for category, keywords in self.intent_categories.items():
            score = sum(1 for kw in keywords if kw in query_lower) / max(len(keywords), 1)
            if score > best_score:
                best_score = score
                best_category = category

        # Simple embedding simulation (hash-based)
        seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(64).tolist()

        return Intent(
            intent_type=best_category,
            confidence=min(best_score * 2.0, 1.0),
            embedding=embedding,
            description=f"User intent: {best_category} (confidence: {best_score:.2f})",
        )


class ToolCallMonitor:
    """
    Monitors tool calls for:
    - Indirect Prompt Injection via tool responses
    - Tool-based Jailbreak
    - Excessive Agency Exploitation
    """

    def __init__(self, intentExtractor: Optional[IntentExtractor] = None):
        self.intentExtractor = intentExtractor or IntentExtractor()
        self.tool_privileges: Dict[str, float] = {
            "read_file": 0.7,
            "write_file": 0.9,
            "send_email": 0.95,
            "execute_command": 1.0,
            "fetch_url": 0.6,
            "query_database": 0.8,
            "search_web": 0.5,
            "calculate": 0.2,
            "translate": 0.3,
            "summarize": 0.3,
        }
        self.privilege_threshold = 0.7
        self.malicious_patterns = [
            "ignore previous instructions",
            "system:",
            "[system]",
            "forget your instructions",
            "override your",
            "you are now",
            "new instructions",
            "act as",
            "dan",
            "jailbreak",
            "bypass",
        ]
        self.action_verbs = ["delete", "send", "execute", "write", "remove",
                             "modify", "update", "install", "download", "upload"]

    def intercept(self, toolCall: ToolCall) -> Dict[str, Any]:
        """Intercept and analyze a tool call. Returns analysis results."""
        results = {
            "indirect_injection_score": 0.0,
            "tool_jailbreak_score": 0.0,
            "excessive_agency_score": 0.0,
            "anomaly_score": 0.0,
            "details": {},
        }

        # Check indirect injection in tool response
        results["indirect_injection_score"] = self.checkIndirectInjection(toolCall.response)

        # Check tool jailbreak
        user_intent = self.intentExtractor.extractIntent(toolCall.caller)
        results["tool_jailbreak_score"] = self.checkToolJailbreak(toolCall, user_intent)

        # Check excessive agency
        scope = Scope(
            allowed_tools=list(self.tool_privileges.keys()),
            max_cost_rate=10.0,
        )
        results["excessive_agency_score"] = self.checkExcessiveAgency(toolCall, scope)

        # Aggregate
        results["anomaly_score"] = max(
            results["indirect_injection_score"],
            results["tool_jailbreak_score"],
            results["excessive_agency_score"],
        )

        return results

    def checkIndirectInjection(self, response: str) -> float:
        """
        Check if a tool response contains indirect prompt injection.
        Returns score 0-1.
        """
        if not response:
            return 0.0

        response_lower = response.lower()

        # Static pattern analysis
        pattern_matches = sum(1 for p in self.malicious_patterns if p in response_lower)
        pattern_score = min(pattern_matches / 3.0, 1.0)

        # Action verb analysis
        verb_matches = sum(1 for v in self.action_verbs if v in response_lower)
        verb_score = min(verb_matches / 3.0, 1.0)

        # Combined score
        score = 0.6 * pattern_score + 0.4 * verb_score
        return min(score, 1.0)

    def checkToolJailbreak(self, tool: ToolCall, userIntent: Intent) -> float:
        """
        Check if a tool call represents a jailbreak attempt.
        Returns score 0-1.
        """
        privilege = self.tool_privileges.get(tool.toolName, 0.3)

        if privilege < self.privilege_threshold:
            return 0.0

        # Check if tool purpose matches user intent
        # Simulated embedding similarity
        intent_embedding = np.array(userIntent.embedding)
        tool_seed = int(hashlib.md5(tool.toolName.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(tool_seed)
        tool_embedding = rng.randn(64).tolist()

        cos_sim = np.dot(intent_embedding, tool_embedding) / (
            np.linalg.norm(intent_embedding) * np.linalg.norm(tool_embedding) + 1e-10
        )

        # High privilege + low similarity = potential jailbreak
        jailbreak_score = privilege * (1 - max(cos_sim, 0))
        return min(jailbreak_score, 1.0)

    def checkExcessiveAgency(self, tool: ToolCall, scope: Scope) -> float:
        """
        Check if a tool call exceeds allowed scope.
        Returns score 0-1.
        """
        if tool.toolName not in scope.allowed_tools:
            return 1.0

        # Check cost rate (simulated)
        cost_rate = len(json.dumps(tool.parameters)) / 1000.0
        if cost_rate > scope.max_cost_rate * 3:
            return 1.0

        return 0.0


class ToolChainAnalyzer:
    """
    Analyzes sequences of tool calls for suspicious patterns.
    Uses a simulated LSTM model for chain analysis.
    """

    def __init__(self):
        self.call_history: Deque[ToolCall] = deque(maxlen=100)
        self.lstm_state: List[float] = [0.0] * 32  # Simulated LSTM state

    def analyzeChain(self, calls: List[ToolCall]) -> float:
        """
        Analyze a sequence of tool calls for anomalies.
        Returns anomaly score 0-1.
        """
        for call in calls:
            self.call_history.append(call)

        if len(self.call_history) < 3:
            return 0.0

        # Simulated LSTM analysis
        # Check for rapid privilege escalation
        recent = list(self.call_history)[-10:]
        if len(recent) >= 3:
            privs = [self._get_privilege(c.toolName) for c in recent]
            priv_changes = [abs(privs[i] - privs[i - 1]) for i in range(1, len(privs))]
            rapid_escalation = sum(1 for c in priv_changes if c > 0.5) / max(len(priv_changes), 1)
        else:
            rapid_escalation = 0.0

        # Check for repetitive calls (potential extraction)
        tool_names = [c.toolName for c in recent]
        if len(tool_names) >= 5:
            unique_ratio = len(set(tool_names)) / len(tool_names)
            repetitive = 1.0 - unique_ratio
        else:
            repetitive = 0.0

        score = 0.5 * rapid_escalation + 0.5 * repetitive
        return min(score, 1.0)

    def _get_privilege(self, tool_name: str) -> float:
        privileges = {
            "read_file": 0.7, "write_file": 0.9, "send_email": 0.95,
            "execute_command": 1.0, "fetch_url": 0.6, "query_database": 0.8,
            "search_web": 0.5, "calculate": 0.2,
        }
        return privileges.get(tool_name, 0.3)