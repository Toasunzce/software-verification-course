from typing import Protocol

import httpx

from app.core.exceptions import LLMError


class LLMClient(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str: ...


class GroqClient:
    def __init__(self, api_key: str, model: str, url: str, timeout: float = 30.0):
        self._api_key = api_key
        self._model = model
        self._url = url
        self._http = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str | None = None) -> str:
        if not self._api_key:
            raise LLMError("GROQ_API_KEY wasn't defined")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._http.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "messages": messages, "temperature": 0.2},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise LLMError(f"Groq request error: {e}") from e