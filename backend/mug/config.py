"""Runtime configuration (environment-driven)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"  # filled by scripts/download_index.py


def _provider_keys() -> dict[str, bool]:
    return {"openai": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY"))}


def _default_model() -> str:
    """gpt-4o-mini unless only an Anthropic key is configured."""
    p = os.environ.get("MUG_PROVIDER", "auto").lower()
    keys = _provider_keys()
    if p == "anthropic" or (p == "auto" and keys["anthropic"] and not keys["openai"]):
        return "claude-haiku-4-5"
    return "gpt-4o-mini"


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("MUG_DATA_DIR", _DEFAULT_DATA)))
    # copilot | openai | anthropic | auto (route each model to OpenAI or Anthropic by name when that
    # provider's API key is set, otherwise to the GitHub Copilot endpoint).
    provider: str = os.environ.get("MUG_PROVIDER", "auto").lower()
    copilot_base_url: str = os.environ.get("MUG_COPILOT_BASE_URL", "https://api.githubcopilot.com")
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_base_url: str = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    answer_model: str = os.environ.get("MUG_ANSWER_MODEL") or _default_model()
    # Probe model for logprob-based signals (equivalence / groundedness). Should expose logprobs;
    # without them the probes fall back to parsing the text answer.
    probe_model: str = os.environ.get("MUG_PROBE_MODEL") or _default_model()
    max_concurrency: int = int(os.environ.get("MUG_MAX_CONCURRENCY", "8"))
    k_textbook: int = int(os.environ.get("MUG_K_TEXTBOOK", "4"))
    k_pubmed: int = int(os.environ.get("MUG_K_PUBMED", "3"))
    tau_m: float = float(os.environ.get("MUG_TAU_M", "0.15"))
    tau_g: float = float(os.environ.get("MUG_TAU_G", "0.5"))
    rewrite_deadline_s: float = float(os.environ.get("MUG_REWRITE_DEADLINE_S", "3.0"))
    ncbi_api_key: str | None = os.environ.get("NCBI_API_KEY")
    enable_textbooks: bool = os.environ.get("MUG_DISABLE_TEXTBOOKS", "0") != "1"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()

# Models that return token logprobs (Copilot endpoint verified Nov 2026; OpenAI API documented).
# Anthropic's API does not return logprobs.
LOGPROB_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"}
COPILOT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "claude-haiku-4.5", "claude-haiku-5.5",
                  "claude-sonnet-5", "claude-sonnet-5.5", "claude-opus-5.5", "gemini-3.8-flash"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
ANTHROPIC_MODELS = ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-1"]


def provider_for(model: str, provider: str | None = None) -> str:
    """Which API serves ``model``: an explicit MUG_PROVIDER wins; ``auto`` routes by model name to a
    provider whose key is set and falls back to the GitHub Copilot endpoint."""
    p = (provider or settings.provider).lower()
    if p in ("copilot", "openai", "anthropic"):
        return p
    keys = _provider_keys()
    if model.startswith("claude"):
        return "anthropic" if keys["anthropic"] else "copilot"
    return "openai" if keys["openai"] else "copilot"


def available_models() -> list[str]:
    """Models offered in the UI and accepted by the API for the configured provider(s)."""
    p = settings.provider
    keys = _provider_keys()
    if p == "openai":
        out = list(OPENAI_MODELS)
    elif p == "anthropic":
        out = list(ANTHROPIC_MODELS)
    elif p == "copilot":
        out = list(COPILOT_MODELS)
    else:
        out = []
        if keys["openai"]:
            out += OPENAI_MODELS
        if keys["anthropic"]:
            out += ANTHROPIC_MODELS
        if not out:
            out = list(COPILOT_MODELS)
    extra = [m.strip() for m in os.environ.get("MUG_EXTRA_MODELS", "").split(",") if m.strip()]
    for m in [settings.answer_model, settings.probe_model, *extra]:
        if m not in out:
            out.append(m)
    return out


CHAT_MODELS = available_models()


def supports_logprobs(model: str) -> bool:
    if provider_for(model) == "anthropic":
        return False
    return model in LOGPROB_MODELS or any(model.startswith(m + "-") for m in LOGPROB_MODELS)
