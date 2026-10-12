"""LLM provider adapter: one small interface over Anthropic, OpenAI, a local Ollama server and a deterministic fake.

Every client takes its base URL and API key from project settings (see settings.py), never implicitly from
SDK environment variables. Responses can be cached on disk (cache.py) and every paid call is checked against
the run's cost cap first (budget.py). Use factory.get_llm_client() to get a ready-to-use client.
"""

from uwcopilot.llm.base import LLMClient, LLMMessage, LLMResponse

__all__ = ["LLMClient", "LLMMessage", "LLMResponse"]
