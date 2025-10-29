"""
Configuration utilities for loading secret credentials such as the OpenAI API key.

Supports multiple strategies:
1. Standard environment variables (export OPENAI_API_KEY=...).
2. A local .env file loaded via python-dotenv (if installed).
3. Streamlit's secrets.toml (if running inside Streamlit).
4. An optional cloud secrets manager (currently AWS Secrets Manager via boto3).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _load_dotenv_if_available() -> None:
    """Load variables from a .env file when python-dotenv is present."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return

    # Attempt project-level .env, then default search behaviour.
    project_env = Path(__file__).resolve().parent / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=False)
    load_dotenv(override=False)


def _get_streamlit_secret(key: str) -> Optional[str]:
    """Fetch secret from Streamlit if available."""
    if key in os.environ:
        return os.environ[key]

    try:
        import streamlit as st  # type: ignore
    except ImportError:
        return None

    if "secrets" not in dir(st):
        return None

    try:
        return st.secrets[key]  # type: ignore[index]
    except Exception:
        return None


def _get_aws_secret(secret_id: str) -> Optional[str]:
    """
    Retrieve a secret string from AWS Secrets Manager.

    Requires boto3 and AWS credentials to be configured. Returns None if boto3
    is unavailable or the secret cannot be fetched.
    """
    try:
        import boto3  # type: ignore
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
    except ImportError:
        raise RuntimeError(
            "boto3 is required to fetch secrets from AWS. "
            "Install it with `pip install boto3` or disable the OPENAI_API_KEY_SECRET_ID setting."
        )

    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_id)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to retrieve secret '{secret_id}' from AWS: {exc}") from exc

    secret_string = response.get("SecretString")
    if not secret_string:
        raise RuntimeError(f"Secret '{secret_id}' does not contain a string payload.")

    return secret_string


@lru_cache(maxsize=1)
def get_openai_api_key() -> str:
    """
    Determine the OpenAI API key using the supported strategies.

    Priority order:
    1. Already-present environment variable OPENAI_API_KEY.
    2. Values found by loading .env files (python-dotenv).
    3. Streamlit secrets file key named OPENAI_API_KEY.
    4. AWS Secrets Manager secret referenced by OPENAI_API_KEY_SECRET_ID.
    """
    _load_dotenv_if_available()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    streamlit_key = _get_streamlit_secret("OPENAI_API_KEY")
    if streamlit_key:
        os.environ["OPENAI_API_KEY"] = streamlit_key  # cache for downstream usage
        return streamlit_key

    secret_id = os.getenv("OPENAI_API_KEY_SECRET_ID")
    if secret_id:
        secret_value = _get_aws_secret(secret_id)
        os.environ["OPENAI_API_KEY"] = secret_value
        return secret_value

    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY, add it to a .env file, "
        "configure Streamlit secrets, or supply OPENAI_API_KEY_SECRET_ID for AWS Secrets Manager."
    )


def get_backend_base_url(default: str = "http://127.0.0.1:8000") -> str:
    """Return the backend URL, allowing overrides via env or .env."""
    _load_dotenv_if_available()
    return os.getenv("BACKEND_BASE_URL", default)


__all__ = ["get_openai_api_key", "get_backend_base_url"]
