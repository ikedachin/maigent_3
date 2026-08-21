import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


SENSITIVE_KEYS = {
    "api_key",
    "openai_api_key",
    "openrouter_api_key",
    "openai_compatible_api_key",
    "azure_api_key",
    "azure_openai_api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
}

PROVIDER_ORDER = ["openai", "openai_compatible", "ollama", "lmstudio", "openrouter", "azure", "bedrock"]

PROVIDER_ENV_KEYS = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"),
    "bedrock": ("AWS_BEARER_TOKEN_BEDROCK", ""),
}

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible", "ollama", "lmstudio", "openrouter"}
CONTROL_CONFIG_NAMES = {"initial_clarifier", "tool_selector", "dynamic_replanner", "dynamic_finalizer"}


@dataclass(frozen=True)
class RuntimeConfig:
    values: dict[str, Any]
    sources: list[str]

    @property
    def model(self) -> str:
        provider = self.active_provider
        if not provider:
            return ""
        provider_config = self.provider_config(provider)
        return str(
            provider_config.get("model")
            or provider_config.get("default_model")
            or self.values.get("model")
            or self.values.get("default_model")
            or ""
        ).strip()

    @property
    def api_key(self) -> str:
        provider = self.active_provider
        if not provider:
            return ""
        provider_config = self.provider_config(provider)
        env_key, _ = PROVIDER_ENV_KEYS.get(provider, ("", ""))
        provider_keys = ["api_key", f"{provider}_api_key"]
        if provider == "openai":
            provider_keys.append("openai_api_key")
        if provider == "azure":
            provider_keys.append("azure_openai_api_key")
        for key in provider_keys:
            value = provider_config.get(key)
            if value:
                return str(value).strip()
        for key in provider_keys:
            value = self.values.get(key)
            if value:
                return str(value).strip()
        return os.environ.get(env_key, "").strip()

    @property
    def base_url(self) -> str:
        provider = self.active_provider
        if not provider:
            return ""
        provider_config = self.provider_config(provider)
        _, env_key = PROVIDER_ENV_KEYS.get(provider, ("", ""))
        default = {
            "ollama": "http://localhost:11434/v1",
            "lmstudio": "http://localhost:1234/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }.get(provider, "")
        return str(
            provider_config.get("base_url")
            or provider_config.get("openai_base_url")
            or self.values.get("base_url")
            or self.values.get("openai_base_url")
            or os.environ.get(env_key, "")
            or default
        ).strip()

    @property
    def api_mode(self) -> str:
        provider = self.active_provider
        provider_config = self.provider_config(provider) if provider else {}
        default = "chat"
        value = str(provider_config.get("api_mode") or self.values.get("api_mode") or self.values.get("openai_api_mode") or default).strip().lower()
        return value if value in {"auto", "responses", "chat"} else default

    @property
    def providers(self) -> dict[str, Any]:
        providers = self.values.get("providers") or self.values.get("llm_providers") or {}
        return providers if isinstance(providers, dict) else {}

    @property
    def active_provider(self) -> str:
        providers = self.providers
        for name in PROVIDER_ORDER:
            config = providers.get(name, {})
            if isinstance(config, dict) and _as_bool(config.get("enabled", False)):
                return name
        if providers:
            return ""
        return "openai"

    def provider_config(self, name: str) -> dict[str, Any]:
        providers = self.providers
        config = providers.get(name, {})
        return dict(config) if isinstance(config, dict) else {}

    @property
    def is_openai_compatible_provider(self) -> bool:
        return self.active_provider in OPENAI_COMPATIBLE_PROVIDERS

    @property
    def azure_endpoint(self) -> str:
        config = self.provider_config("azure")
        return str(
            config.get("azure_endpoint")
            or config.get("endpoint")
            or self.values.get("azure_endpoint")
            or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        ).strip()

    @property
    def azure_api_version(self) -> str:
        config = self.provider_config("azure")
        return str(config.get("api_version") or self.values.get("azure_api_version") or "2024-02-15-preview").strip()

    @property
    def bedrock_region(self) -> str:
        config = self.provider_config("bedrock")
        return str(config.get("region") or config.get("region_name") or self.values.get("aws_region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip()

    @property
    def bedrock_profile(self) -> str:
        config = self.provider_config("bedrock")
        return str(config.get("profile") or config.get("profile_name") or self.values.get("aws_profile") or os.environ.get("AWS_PROFILE", "")).strip()

    @property
    def bedrock_credentials(self) -> dict[str, str]:
        config = self.provider_config("bedrock")
        mapping = {
            "aws_access_key_id": "aws_access_key_id",
            "aws_secret_access_key": "aws_secret_access_key",
            "aws_session_token": "aws_session_token",
        }
        credentials = {}
        for config_key, boto_key in mapping.items():
            value = config.get(config_key) or self.values.get(config_key)
            if value:
                credentials[boto_key] = str(value).strip()
        return credentials

    @property
    def tools(self) -> dict[str, Any]:
        tools = self.values.get("tools", {})
        return tools if isinstance(tools, dict) else {}

    @property
    def final_evaluation(self) -> dict[str, Any]:
        settings = self.values.get("final_evaluation", {})
        return settings if isinstance(settings, dict) else {}

    @property
    def final_evaluation_enabled(self) -> bool:
        return _as_bool(self.final_evaluation.get("enabled", False))

    @property
    def final_evaluation_max_retries(self) -> int:
        try:
            return max(0, min(3, int(self.final_evaluation.get("max_retries", 3))))
        except (TypeError, ValueError):
            return 3

    @property
    def final_evaluation_max_output_tokens(self) -> int:
        try:
            return max(1, int(self.final_evaluation.get("max_output_tokens", 8192)))
        except (TypeError, ValueError):
            return 8192

    @property
    def final_evaluation_reasoning_effort(self) -> str:
        value = str(self.final_evaluation.get("reasoning_effort", "none")).strip().lower()
        return value if value in {"none", "minimal", "low", "medium", "high"} else "none"

    @property
    def sandbox_code_generation(self) -> dict[str, Any]:
        value = self.values.get("sandbox_code_generation", {})
        return value if isinstance(value, dict) else {}

    @property
    def sandbox_code_generation_enabled(self) -> bool:
        return _as_bool(self.sandbox_code_generation.get("enabled", True))

    @property
    def sandbox_code_generation_max_output_tokens(self) -> int:
        try:
            return max(1, int(self.sandbox_code_generation.get("max_output_tokens", 32768)))
        except (TypeError, ValueError):
            return 32768

    @property
    def sandbox_code_generation_reasoning_effort(self) -> str:
        value = self.sandbox_code_generation.get("reasoning_effort", "none")
        if isinstance(value, bool):
            return "medium" if value else "none"
        normalized = str(value).strip().lower()
        if normalized in {"true", "on", "yes"}:
            return "medium"
        if normalized in {"false", "off", "no"}:
            return "none"
        return normalized if normalized in {"none", "minimal", "low", "medium", "high", "xhigh"} else "none"

    @property
    def logging(self) -> dict[str, Any]:
        logging_config = self.values.get("logging", {})
        return logging_config if isinstance(logging_config, dict) else {}

    @property
    def llm_log_tail_chars(self) -> int | None:
        value = self.logging.get("llm_tail_chars", 100)
        if isinstance(value, str) and value.strip().lower() in {"full", "all", "unlimited"}:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 100
        return None if parsed <= 0 else max(1, min(200000, parsed))

    def control_config(self, name: str) -> dict[str, Any]:
        config = self.values.get(name, {})
        if isinstance(config, dict):
            return config
        legacy = self.tools.get(name, {})
        return legacy if isinstance(legacy, dict) else {}

    def tool_enabled(self, name: str, default: bool = False) -> bool:
        if name == "file_batch" and name not in self.tools:
            if "rag" not in self.tools:
                return False
            rag = self.tools.get("rag", {})
            return isinstance(rag, dict) and _as_bool(rag.get("enabled", False))
        if name not in CONTROL_CONFIG_NAMES and name not in self.tools:
            return False
        config = self.control_config(name) if name in CONTROL_CONFIG_NAMES else self.tools.get(name, {})
        if not isinstance(config, dict):
            return default
        return _as_bool(config.get("enabled", default))

    @property
    def enabled_tool_names(self) -> set[str]:
        names = {
            str(name)
            for name, config in self.tools.items()
            if isinstance(config, dict) and _as_bool(config.get("enabled", False))
        }
        if self.tool_enabled("file_batch"):
            names.add("file_batch")
        return names

    @property
    def sandbox_image(self) -> str:
        sandbox = self.tools.get("sandbox", {})
        if not isinstance(sandbox, dict):
            return "python:3.11-slim"
        return str(sandbox.get("image") or "python:3.11-slim").strip()

    @property
    def sandbox_allowed_libraries(self) -> list[str]:
        sandbox = self.tools.get("sandbox", {})
        if not isinstance(sandbox, dict):
            return []
        libraries = sandbox.get("allowed_libraries", [])
        if isinstance(libraries, str):
            return [libraries]
        if isinstance(libraries, list):
            return [str(item).strip() for item in libraries if str(item).strip()]
        return []

    @property
    def sandbox_install_libraries_on_run(self) -> bool:
        sandbox = self.tools.get("sandbox", {})
        if not isinstance(sandbox, dict):
            return False
        return _as_bool(sandbox.get("install_libraries_on_run", False))

    @property
    def sandbox_timeout_seconds(self) -> int:
        sandbox = self.tools.get("sandbox", {})
        if not isinstance(sandbox, dict):
            return 20
        try:
            return max(1, min(600, int(sandbox.get("timeout_seconds", 20))))
        except (TypeError, ValueError):
            return 20

    @property
    def sandbox_memory_limit_mb(self) -> int:
        sandbox = self.tools.get("sandbox", {})
        if not isinstance(sandbox, dict):
            return 512
        try:
            return max(64, min(8192, int(sandbox.get("memory_limit_mb", 512))))
        except (TypeError, ValueError):
            return 512

    @property
    def sandbox_pids_limit(self) -> int:
        sandbox = self.tools.get("sandbox", {})
        if not isinstance(sandbox, dict):
            return 128
        try:
            return max(16, min(2048, int(sandbox.get("pids_limit", 128))))
        except (TypeError, ValueError):
            return 128

    @property
    def sandbox_cpus(self) -> float:
        sandbox = self.tools.get("sandbox", {})
        if not isinstance(sandbox, dict):
            return 1.0
        try:
            return max(0.1, min(8.0, float(sandbox.get("cpus", 1.0))))
        except (TypeError, ValueError):
            return 1.0

    @property
    def web_search_api_key(self) -> str:
        web_search = self.tools.get("web_search", {})
        config_value = web_search.get("api_key") if isinstance(web_search, dict) else None
        return str(config_value or os.environ.get("TAVILY_API_KEY", "")).strip()

    @property
    def web_search_max_results(self) -> int:
        web_search = self.tools.get("web_search", {})
        if not isinstance(web_search, dict):
            return 5
        try:
            return max(1, min(10, int(web_search.get("max_results", 5))))
        except (TypeError, ValueError):
            return 5

    @property
    def web_search_timeout_seconds(self) -> int:
        web_search = self.tools.get("web_search", {})
        if not isinstance(web_search, dict):
            return 10
        try:
            return max(1, min(30, int(web_search.get("timeout_seconds", 10))))
        except (TypeError, ValueError):
            return 10

    @property
    def web_search_max_retries(self) -> int:
        web_search = self.tools.get("web_search", {})
        if not isinstance(web_search, dict):
            return 1
        try:
            return max(0, min(5, int(web_search.get("max_retries", 1))))
        except (TypeError, ValueError):
            return 1

    @property
    def multi_agent(self) -> dict[str, Any]:
        value = self.values.get("multi_agent", {})
        return value if isinstance(value, dict) else {}

    @property
    def multi_agent_enabled(self) -> bool:
        return _as_bool(self.multi_agent.get("enabled", True))

    @property
    def multi_agent_max_workers(self) -> int:
        try:
            return max(1, min(5, int(self.multi_agent.get("max_workers", 3))))
        except (TypeError, ValueError):
            return 3

    @property
    def multi_agent_parallel_tools(self) -> bool:
        return _as_bool(self.multi_agent.get("parallel_tools", True))

    @property
    def multi_agent_progress_visible(self) -> bool:
        return _as_bool(self.multi_agent.get("progress_visible", True))

    def redacted(self) -> dict[str, Any]:
        safe = _redact_sensitive(self.values)
        if os.environ.get("OPENAI_API_KEY") and not any(k in self.values for k in SENSITIVE_KEYS):
            safe["api_key"] = "******** (env)"
        return safe


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in SENSITIVE_KEYS and item:
                redacted[key] = "********"
            else:
                redacted[key] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data if isinstance(data, dict) else {}


def _read_yaml_subset(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: tuple[int, dict[str, Any], str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            value = _parse_scalar(stripped[2:].strip())
            if not isinstance(parent, list):
                if pending_key is None:
                    continue
                pending_indent, pending_parent, key = pending_key
                if pending_indent != indent:
                    continue
                new_list: list[Any] = []
                pending_parent[key] = new_list
                stack.append((indent - 1, new_list))
                parent = new_list
            parent.append(value)
            continue
        if ":" not in stripped or not isinstance(parent, dict):
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            pending_key = None
            continue
        child: dict[str, Any] = {}
        parent[key] = child
        stack.append((indent, child))
        pending_key = (indent + 2, parent, key)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_config_pair(directory: Path) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    sources: list[str] = []
    toml_path = directory / "config.toml"
    toml_values = _read_toml(toml_path)
    if toml_values:
        _deep_update(values, toml_values)
        sources.append(str(toml_path))
    yaml_path = directory / "config.yaml"
    yaml_values = _read_yaml_subset(yaml_path)
    if yaml_values:
        _deep_update(values, yaml_values)
        sources.append(str(yaml_path))
    return values, sources


def load_runtime_config(project_path: str = "") -> RuntimeConfig:
    values: dict[str, Any] = {}
    sources: list[str] = []

    user_values, user_sources = _load_config_pair(Path.home() / ".maigent")
    if user_values:
        _deep_update(values, user_values)
        sources.extend(user_sources)

    app_dir = Path(settings.BASE_DIR) / ".maigent"
    app_values, app_sources = _load_config_pair(app_dir)
    if app_values:
        _deep_update(values, app_values)
        sources.extend(app_sources)

    if project_path:
        project_dir = Path(project_path).expanduser() / ".maigent"
        if project_dir != app_dir:
            project_values, project_sources = _load_config_pair(project_dir)
            if project_values:
                _deep_update(values, project_values)
                sources.extend(project_sources)

    env_model = os.environ.get("OPENAI_MODEL")
    if env_model and not values.get("model") and not values.get("default_model"):
        values["model"] = env_model
        sources.append("OPENAI_MODEL")

    return RuntimeConfig(values=values, sources=sources)


def _maigent_layer_dirs(project_path: str = "") -> list[Path]:
    dirs = [Path.home() / ".maigent", Path(settings.BASE_DIR) / ".maigent"]
    if project_path:
        project_dir = Path(project_path).expanduser() / ".maigent"
        if project_dir != dirs[-1]:
            dirs.append(project_dir)
    return dirs


def load_agents_md(project_path: str = "") -> str:
    """Read AGENTS.md from each .maigent/ layer (user, app, project) and
    concatenate whichever are present, in that precedence order."""
    parts: list[str] = []
    for directory in _maigent_layer_dirs(project_path):
        agents_path = directory / "AGENTS.md"
        try:
            if agents_path.is_file():
                text = agents_path.read_text(encoding="utf-8").strip()
                if text:
                    parts.append(text)
        except OSError:
            continue
    return "\n\n---\n\n".join(parts)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: str


def _parse_skill_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a SKILL.md file: optional `---`-delimited frontmatter with flat
    `key: value` pairs, followed by the markdown body. Returns (frontmatter, body)."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    frontmatter: dict[str, str] = {}
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    if end_index is None:
        return {}, text.strip()
    body = "\n".join(lines[end_index + 1 :]).strip()
    return frontmatter, body


def _load_skills_from_dir(skills_dir: Path) -> list[Skill]:
    skills: list[Skill] = []
    if not skills_dir.is_dir():
        return skills
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_path = entry / "SKILL.md"
        if not skill_path.is_file():
            continue
        try:
            text = skill_path.read_text(encoding="utf-8")
        except OSError:
            continue
        frontmatter, body = _parse_skill_frontmatter(text)
        if not body:
            continue
        name = frontmatter.get("name") or entry.name
        description = frontmatter.get("description") or ""
        skills.append(Skill(name=name, description=description, body=body, path=str(skill_path)))
    return skills


def load_skills(project_path: str = "") -> list[Skill]:
    """Discover .maigent/skills/<name>/SKILL.md across the user, app, and
    project layers. Later layers override earlier ones with the same skill name."""
    skills: dict[str, Skill] = {}
    for directory in _maigent_layer_dirs(project_path):
        for skill in _load_skills_from_dir(directory / "skills"):
            skills[skill.name] = skill
    return list(skills.values())
