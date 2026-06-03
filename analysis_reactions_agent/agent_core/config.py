from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict


PROVIDER_DEFAULTS = {
    "gemini": {
        "model": "gemini-2.5-flash",
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "api_key_file": "gemini_api_key.txt",
        "env_var": "GEMINI_API_KEY",
    },
    "deepseek": {
        "model": "deepseek-v4-flash",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_file": "deepseek_api_key.txt",
        "env_var": "DEEPSEEK_API_KEY",
    },
}


def load_runtime_config(script_dir: Path, config_path: Path) -> Dict[str, object]:
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object: {0}".format(config_path))
    return data


def resolve_provider_name(args, runtime_config: Dict[str, object], default_provider: str) -> str:
    provider = (args.provider or runtime_config.get("provider") or default_provider).strip().lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(
            "Unsupported provider '{0}', expected one of: {1}".format(
                provider, ", ".join(sorted(PROVIDER_DEFAULTS.keys()))
            )
        )
    return provider


def resolve_provider_settings(provider: str, runtime_config: Dict[str, object]) -> Dict[str, str]:
    settings = dict(PROVIDER_DEFAULTS[provider])
    configured_providers = runtime_config.get("providers")
    if isinstance(configured_providers, dict):
        provider_settings = configured_providers.get(provider)
        if isinstance(provider_settings, dict):
            for key in ("model", "api_url", "api_key_file"):
                value = provider_settings.get(key)
                if value:
                    settings[key] = str(value).strip()
    return settings


def resolve_runtime_options(args, runtime_config: Dict[str, object], script_dir: Path, default_provider: str):
    provider = resolve_provider_name(args, runtime_config, default_provider)
    provider_settings = resolve_provider_settings(provider, runtime_config)
    args.provider = provider
    args.model = (args.model or provider_settings["model"]).strip()
    args.api_url = (args.api_url or provider_settings["api_url"]).strip()
    args.api_key_file = Path(provider_settings["api_key_file"])
    if not args.api_key_file.is_absolute():
        args.api_key_file = script_dir / args.api_key_file
    args.api_key_env_var = provider_settings["env_var"]
    args.api_key = (args.api_key or "").strip()
    if not args.api_key and args.api_key_file.exists():
        args.api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not args.api_key:
        args.api_key = os.getenv(args.api_key_env_var, "").strip()
    return args
