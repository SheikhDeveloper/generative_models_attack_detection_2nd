"""
Core data models for the Attack Detection System.
Based on the UML class diagram (sys-class-diagram.puml).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CODE = "code"


class VerdictType(Enum):
    LEGITIMATE_KNOWN = "legitimate_known"
    LEGITIMATE_NEW = "legitimate_new"
    ATTACK = "attack"
    SUSPICIOUS = "suspicious"


class AttackType(Enum):
    PROMPT_INJECTION = "prompt_injection"
    EVASION = "evasion"
    DATA_POISONING = "data_poisoning"
    MODEL_INVERSION = "model_inversion"
    MODEL_EXTRACTION = "model_extraction"
    MEMBERSHIP_INFERENCE = "membership_inference"
    JAILBREAK = "jailbreak"
    BACKDOOR = "backdoor"
    INDIRECT_PROMPT_INJECTION = "indirect_prompt_injection"
    TOOL_JAILBREAK = "tool_jailbreak"
    EXCESSIVE_AGENCY = "excessive_agency"
    LLM_JACKING = "llm_jacking"
    SUPPLY_CHAIN = "supply_chain"
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    DISTRIBUTED_JAILBREAK = "distributed_jailbreak"
    CONTEXT_POISONING = "context_poisoning"
    UNKNOWN = "unknown"


@dataclass
class Request:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    userId: str = ""
    query: str = ""
    modality: Modality = Modality.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    outputData: str = ""
    generationTime: int = 0  # ms
    modality: Modality = Modality.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    toolName: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    caller: str = ""
    response: str = ""
    privilege_level: float = 0.0  # 0-1 scale


@dataclass
class Intent:
    intent_type: str = ""
    confidence: float = 0.0
    embedding: List[float] = field(default_factory=list)
    description: str = ""


@dataclass
class Scope:
    allowed_tools: List[str] = field(default_factory=list)
    max_cost_rate: float = 0.0
    user_id: str = ""


@dataclass
class ValidationResult:
    verdict: VerdictType = VerdictType.LEGITIMATE_KNOWN
    confidence: float = 0.0
    reason: str = ""


@dataclass
class Incident:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attackType: AttackType = AttackType.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    anomalyScore: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    requestId: str = ""
    userId: str = ""


@dataclass
class DetectionResult:
    requestId: str = ""
    verdict: VerdictType = VerdictType.LEGITIMATE_KNOWN
    anomalyScore: float = 0.0
    level1Score: float = 0.0
    level2Score: float = 0.0
    level3Score: float = 0.0
    level4Score: float = 0.0
    ensembleScores: Dict[str, float] = field(default_factory=dict)
    attackType: Optional[AttackType] = None
    confidence: float = 0.0
    processingTimeMs: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabeledRequest:
    request: Request = field(default_factory=Request)
    response: Response = field(default_factory=Response)
    features: List[float] = field(default_factory=list)
    true_label: str = ""
    predicted_label: str = ""


@dataclass
class NovelClass:
    name: str = ""
    centroid: List[float] = field(default_factory=list)
    samples: int = 0
    first_seen: datetime = field(default_factory=datetime.now)


@dataclass
class Dependency:
    name: str = ""
    version: str = ""
    expected_hash: str = ""
    current_hash: str = ""


@dataclass
class Connection:
    dest_ip: str = ""
    dest_port: int = 0
    protocol: str = "tcp"
    timestamp: datetime = field(default_factory=datetime.now)