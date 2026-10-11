"""Reading project settings the same way everywhere.

Every settings module (LLM, RAG, agent, risk service) follows the same two rules:

- only the variable names it asks for are read, so a variable set by the hosting platform (for example
  ANTHROPIC_BASE_URL) never changes anything by accident;
- a non-empty value in the repo .env file wins over the same name in the process environment.

env_reader() implements those rules once. is_http_url() is the shared check for service URLs.
"""

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

SettingReader = Callable[[str], str | None]


def env_reader(env_file: Path | None, environ: Mapping[str, str] | None = None) -> SettingReader:
    """Return read(name): the stripped, non-empty value of name from env_file, else from environ, else None.

    env_file=None skips the .env file (tests do this). environ=None means the real process environment.
    """
    process_env = os.environ if environ is None else environ
    file_values: dict[str, str | None] = {}
    if env_file is not None and env_file.exists():
        file_values = dict(dotenv_values(env_file))

    def read(name: str) -> str | None:
        """The non-empty value of name from the .env file, else from the process environment."""
        for source in (file_values, process_env):
            value = source.get(name)
            if value is not None and value.strip():
                return value.strip()
        return None

    return read


def is_http_url(url: str) -> bool:
    """True for an absolute http:// or https:// URL that names a host."""
    try:
        parts = urlsplit(url)
        has_host = bool(parts.hostname)
    except ValueError:  # for example an invalid port or a malformed IPv6 address
        return False
    return parts.scheme in ("http", "https") and has_host
