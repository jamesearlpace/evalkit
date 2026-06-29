"""Provider-agnostic LLM client wrappers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from .spec import LLMSpec


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot complete a request."""


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Return a chat-completion response as text."""


@dataclass
class OpenAIChatClient:
    model: str
    api_key: str | None = None
    base_url: str | None = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("Install the 'openai' package to use the OpenAI provider.") from exc

        client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"), base_url=self.base_url)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


@dataclass
class AzureOpenAIChatClient:
    model: str
    endpoint: str | None = None
    api_key: str | None = None
    api_version: str | None = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise LLMError("Install the 'openai' package to use the Azure OpenAI provider.") from exc

        client = AzureOpenAI(
            azure_endpoint=self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=self.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


@dataclass
class MockLLMClient:
    """Small test/demo client that returns a configured response."""

    response: str = '{"score": 1.0, "reasoning": "mock response", "failure_mode": null}'

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return self.response


def build_llm_client(spec: LLMSpec | None) -> LLMClient | None:
    if spec is None:
        return None

    provider = spec.provider.lower()
    if provider in {"none", "disabled"}:
        return None
    if provider == "mock":
        return MockLLMClient(response=str(spec.params.get("response", MockLLMClient.response)))
    if provider == "openai":
        model = spec.model or os.getenv("EVALKIT_JUDGE_MODEL") or "gpt-4o-mini"
        return OpenAIChatClient(
            model=model,
            api_key=spec.params.get("api_key"),
            base_url=spec.params.get("base_url"),
        )
    if provider in {"azure", "azure_openai", "azure-openai"}:
        model = spec.model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not model:
            raise LLMError("Azure OpenAI provider requires llm.model or AZURE_OPENAI_DEPLOYMENT.")
        return AzureOpenAIChatClient(
            model=model,
            endpoint=spec.params.get("endpoint"),
            api_key=spec.params.get("api_key"),
            api_version=spec.params.get("api_version"),
        )
    raise LLMError(f"Unsupported LLM provider '{spec.provider}'.")
