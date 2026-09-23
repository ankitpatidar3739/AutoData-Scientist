"""
Base Agent class supporting Google Gemini API with fallback to deterministic expert reasoning.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel
from src.core.config import settings

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = settings.gemini_model

    def call_llm(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Invokes Gemini LLM if API key is available; otherwise returns empty string."""
        if not self.api_key:
            return ""

        try:
            # Try new google-genai SDK first
            from google import genai
            client = genai.Client(api_key=self.api_key)
            full_prompt = f"System: {system_instruction}\n\nUser: {prompt}" if system_instruction else prompt
            response = client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            return response.text or ""
        except ImportError:
            pass

        try:
            # Fallback to google.generativeai
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            model = genai_legacy.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            return response.text or ""
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Falling back to deterministic rules engine.")
            return ""

    def parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parses JSON from LLM response, stripping markdown code blocks if present."""
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
