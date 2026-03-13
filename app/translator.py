from __future__ import annotations

import json
import sys

import httpx


PROMPT_TEMPLATE = """You are a professional German (de-AT) to English (en) translator. Your goal is to accurately convey the meaning and nuances of the original German text while adhering to English grammar, vocabulary, and cultural sensitivities.
Produce only the English translation, without any additional explanations or commentary. Please translate the following German text into English:


{german_text}
"""


class OllamaTranslator:
    """Translate German transcript segments through a local Ollama instance."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "translategemma",
        request_timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(request_timeout_seconds),
        )

    async def healthcheck(self) -> None:
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        available_names = {
            model.get("name", "").split(":")[0]
            for model in models
            if isinstance(model, dict) and model.get("name")
        }
        if self.model_name not in available_names:
            raise RuntimeError(
                f"Ollama model '{self.model_name}' is not available. "
                f"Run `ollama pull {self.model_name}` first."
            )

    async def close(self) -> None:
        await self._client.aclose()

    async def translate(self, german_text: str, *, stream_output: bool = False) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(german_text=german_text.strip()),
                }
            ],
            "stream": True,
        }

        translated_parts: list[str] = []
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue

                message = json.loads(line)
                if "error" in message:
                    raise RuntimeError(message["error"])

                content = message.get("message", {}).get("content", "")
                if not content:
                    continue

                translated_parts.append(content)
                if stream_output:
                    sys.stdout.write(content)
                    sys.stdout.flush()

        return "".join(translated_parts).strip()
