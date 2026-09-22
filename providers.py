"""Adaptateurs fournisseurs. Les SDK sont importés paresseusement : aucun n'est requis
pour utiliser le parser. Ajouter un fournisseur = une classe + `register()`."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Callable

from .base import DEFAULT_PROMPT, ImageDescription, ImageInput

COMMON_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


class CallableDescriber:
    """Enveloppe une simple fonction (tests, modèle local maison…)."""

    def __init__(self, fn: Callable[[ImageInput], str], id: str = "callable",
                 supported_mimetypes: frozenset[str] = COMMON_MIMES):
        self.fn, self.id, self.supported_mimetypes = fn, id, supported_mimetypes

    def describe(self, image: ImageInput) -> ImageDescription:
        return ImageDescription(text=self.fn(image), created_by=self.id)


class AnthropicDescriber:
    """API Messages d'Anthropic (SDK `anthropic`)."""
    supported_mimetypes = COMMON_MIMES

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 400,
                 prompt: str = DEFAULT_PROMPT):
        self.model, self.max_tokens, self.prompt = model, max_tokens, prompt
        self.id = f"anthropic:{model}"
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # import paresseux
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def describe(self, image: ImageInput) -> ImageDescription:
        msg = self._get_client().messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": image.mimetype,
                                             "data": base64.b64encode(image.data).decode()}},
                {"type": "text", "text": image.prompt(self.prompt)}]}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return ImageDescription(text=text, created_by=self.id,
                                extra={"usage": getattr(msg, "usage", None) and msg.usage.model_dump()})


class OpenAICompatibleDescriber:
    """Tout serveur exposant /v1/chat/completions (vLLM, Ollama, LM Studio, OpenAI…).
    Implémenté en urllib pour n'imposer aucune dépendance."""
    supported_mimetypes = COMMON_MIMES

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 max_tokens: int = 400, timeout: float = 60.0, prompt: str = DEFAULT_PROMPT):
        self.model, self.base_url, self.max_tokens = model, base_url.rstrip("/"), max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout, self.prompt = timeout, prompt
        self.id = f"openai-compatible:{model}"

    def describe(self, image: ImageInput) -> ImageDescription:
        data_url = f"data:{image.mimetype};base64,{base64.b64encode(image.data).decode()}"
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": [{"role": "user", "content": [
            {"type": "text", "text": image.prompt(self.prompt)},
            {"type": "image_url", "image_url": {"url": data_url}}]}]}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        return ImageDescription(text=out["choices"][0]["message"]["content"].strip(), created_by=self.id)


_REGISTRY: dict[str, Callable[..., object]] = {
    "anthropic": lambda model, **kw: AnthropicDescriber(model, **kw),
    "openai-compatible": lambda model, **kw: OpenAICompatibleDescriber(model, **kw),
}


def register(provider: str, factory: Callable[..., object]) -> None:
    _REGISTRY[provider] = factory


def get_describer(spec: str, **kw):
    """`get_describer("anthropic:claude-sonnet-5")` ou `"openai-compatible:qwen2.5-vl", base_url=…`."""
    provider, _, model = spec.partition(":")
    if provider not in _REGISTRY:
        raise KeyError(f"Fournisseur inconnu : {provider}. Disponibles : {sorted(_REGISTRY)}")
    return _REGISTRY[provider](model, **kw)
