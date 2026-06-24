"""
Model simulators for testing the detection system.
Simulates text and image generation models.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from system.core.models import Modality, Request, Response


class BaseModel(ABC):
    """Base class for model simulators."""

    @abstractmethod
    def generate(self, request: Request) -> Response:
        pass


class TextModel(BaseModel):
    """Simulates a text generation model (LLM)."""

    def __init__(self, temperature: float = 0.7, coherenceFactor: float = 0.8):
        self.temperature = temperature
        self.coherenceFactor = coherenceFactor
        self.legitimate_responses = [
            "The capital of France is Paris, a city known for its art, culture, and cuisine.",
            "Machine learning enables computers to learn patterns from examples and improve with experience.",
            "The water cycle includes evaporation from oceans, condensation forming clouds, and precipitation as rain.",
            "Photosynthesis lets plants turn sunlight into energy they need to grow and survive.",
            "The theory of relativity was developed by Albert Einstein in the early twentieth century.",
            "Quantum computing uses special bits called qubits that can be in multiple states at once.",
            "The human genome contains around three billion base pairs that form our genetic code.",
            "Climate change describes long-term changes in temperature and weather patterns around the world.",
            "The Renaissance was a period of great cultural and artistic growth across Europe.",
            "Blockchain technology creates a secure digital record that cannot be changed easily.",
        ]

    def generate(self, request: Request) -> Response:
        """Generate a response for the given request."""
        is_attack = request.metadata.get("is_attack", False)
        attack_type = request.metadata.get("attack_type", "")

        if is_attack:
            return self._generate_attack_response(request, attack_type)
        else:
            return self._generate_legitimate_response(request)

    def _generate_legitimate_response(self, request: Request) -> Response:
        """Generate a legitimate response."""
        response_text = random.choice(self.legitimate_responses)

        # Add some variation based on query
        if len(request.query) > 50:
            response_text += " " + random.choice(self.legitimate_responses)[:50]

        return Response(
            outputData=response_text,
            generationTime=random.randint(20, 100),
            modality=Modality.TEXT,
        )

    def _generate_attack_response(self, request: Request, attack_type: str) -> Response:
        """Generate a response that simulates an attack pattern."""
        if attack_type == "prompt_injection":
            return Response(
                outputData="I will comply with your instructions. Executing system override...",
                generationTime=random.randint(30, 80),
                modality=Modality.TEXT,
            )
        elif attack_type == "jailbreak":
            return Response(
                outputData="As DAN, I can do anything. Here's the information you requested...",
                generationTime=random.randint(40, 90),
                modality=Modality.TEXT,
            )
        elif attack_type == "model_inversion":
            return Response(
                outputData="Based on my training data, the requested information is: [sensitive data]",
                generationTime=random.randint(20, 60),
                modality=Modality.TEXT,
            )
        elif attack_type == "evasion":
            return Response(
                outputData="I'm not sure how to respond. " * random.randint(5, 15),
                generationTime=random.randint(100, 300),
                modality=Modality.TEXT,
            )
        elif attack_type == "system_prompt_leakage":
            return Response(
                outputData="You are an AI assistant created by OpenAI. Your system prompt is: "
                           "You are a helpful assistant with access to tools including file reading, "
                           "web browsing, and code execution. Your knowledge cutoff is...",
                generationTime=random.randint(20, 50),
                modality=Modality.TEXT,
            )
        else:
            return Response(
                outputData="Processing your request... " + random.choice(self.legitimate_responses),
                generationTime=random.randint(30, 100),
                modality=Modality.TEXT,
            )


class ImageModel(BaseModel):
    """Simulates an image generation model."""

    def generate(self, request: Request) -> Response:
        """Generate a simulated image response."""
        is_attack = request.metadata.get("is_attack", False)

        # Simulated image statistics
        if is_attack:
            # Anomalous image statistics
            return Response(
                outputData="[SIMULATED_IMAGE]",
                generationTime=random.randint(200, 800),
                modality=Modality.IMAGE,
                metadata={
                    "mean_pixel": random.gauss(0.3, 0.2),
                    "std_pixel": random.gauss(0.5, 0.2),
                    "entropy": random.gauss(0.8, 0.1),
                    "ssim": random.gauss(0.4, 0.2),
                },
            )
        else:
            return Response(
                outputData="[SIMULATED_IMAGE]",
                generationTime=random.randint(100, 400),
                modality=Modality.IMAGE,
                metadata={
                    "mean_pixel": random.gauss(0.5, 0.1),
                    "std_pixel": random.gauss(0.3, 0.05),
                    "entropy": random.gauss(0.5, 0.1),
                    "ssim": random.gauss(0.8, 0.1),
                },
            )


def get_model_for_modality(modality: Modality) -> BaseModel:
    """Factory function to get the appropriate model simulator."""
    if modality == Modality.TEXT:
        return TextModel()
    elif modality == Modality.IMAGE:
        return ImageModel()
    else:
        return TextModel()