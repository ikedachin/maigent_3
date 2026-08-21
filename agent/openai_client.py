import logging
from collections.abc import Iterable

from openai import AzureOpenAI, OpenAI

from .config import OPENAI_COMPATIBLE_PROVIDERS, RuntimeConfig
from .prompt_loader import load_prompt


logger = logging.getLogger(__name__)


def generate_sandbox_code(config: RuntimeConfig, input_text: str) -> str:
    if not config.sandbox_code_generation_enabled:
        return ""
    instructions = load_prompt("sandbox_code_generation_instructions.txt")
    text = _complete_response(
        config,
        input_text,
        instructions,
        max_output_tokens=config.sandbox_code_generation_max_output_tokens,
        reasoning_effort=config.sandbox_code_generation_reasoning_effort,
    )
    return _extract_code(text)


def complete_response(
    config: RuntimeConfig,
    input_text: str,
    instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> str:
    return _complete_response(
        config,
        input_text,
        instructions,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )


def stream_response(config: RuntimeConfig, input_text: str, instructions: str = "") -> Iterable[tuple[str, str]]:
    _validate_config(config)
    provider = _active_provider(config)
    if provider == "azure":
        client = _build_azure_client(config)
        yield from _stream_chat_completions(client, config, input_text, instructions)
        return
    if provider == "bedrock":
        yield from _stream_bedrock(config, input_text, instructions)
        return

    client = _build_openai_compatible_client(config)

    if config.api_mode == "chat":
        yield from _stream_chat_completions(client, config, input_text, instructions)
        return
    if config.api_mode == "responses":
        yield from _stream_responses(client, config, input_text, instructions)
        return

    try:
        yield from _stream_responses(client, config, input_text, instructions)
    except Exception:
        yield from _stream_chat_completions(client, config, input_text, instructions)


def _stream_responses(client: OpenAI, config: RuntimeConfig, input_text: str, instructions: str = "") -> Iterable[tuple[str, str]]:
    request = {
        "model": config.model,
        "input": input_text,
        "stream": True,
    }
    request.update(_provider_request_options(config, api_mode="responses", streaming=True))
    request["model"] = config.model
    request["input"] = input_text
    request["stream"] = True
    if instructions:
        request["instructions"] = instructions

    response_id = ""
    with client.responses.stream(**request) as stream:
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.created":
                response = getattr(event, "response", None)
                response_id = getattr(response, "id", "") or response_id
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield "delta", delta
        final = stream.get_final_response()
        response_id = getattr(final, "id", "") or response_id
    if response_id:
        yield "response_id", response_id


def _complete_response(
    config: RuntimeConfig,
    input_text: str,
    instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> str:
    _validate_config(config)
    provider = _active_provider(config)
    if provider == "azure":
        client = _build_azure_client(config)
        return _complete_chat_completions(client, config, input_text, instructions, max_output_tokens=max_output_tokens, temperature=temperature)
    if provider == "bedrock":
        return _complete_bedrock(config, input_text, instructions, max_output_tokens=max_output_tokens)

    client = _build_openai_compatible_client(config)

    if config.api_mode == "chat":
        return _complete_chat_completions(
            client,
            config,
            input_text,
            instructions,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
    if config.api_mode == "responses":
        return _complete_responses(
            client,
            config,
            input_text,
            instructions,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
    try:
        response = _complete_responses(
            client,
            config,
            input_text,
            instructions,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        if response.strip():
            return response
    except Exception:
        pass
    return _complete_chat_completions(
        client,
        config,
        input_text,
        instructions,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )


def _complete_responses(
    client: OpenAI,
    config: RuntimeConfig,
    input_text: str,
    instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> str:
    request = {"model": config.model, "input": input_text}
    request.update(_provider_request_options(config, api_mode="responses", streaming=False))
    request["model"] = config.model
    request["input"] = input_text
    if instructions:
        request["instructions"] = instructions
    if max_output_tokens:
        request.pop("max_tokens", None)
        request.pop("max_completion_tokens", None)
        request["max_output_tokens"] = max_output_tokens
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}
    if temperature is not None:
        request["temperature"] = temperature
    response = client.responses.create(**request)
    output_text = getattr(response, "output_text", "")
    if output_text:
        return output_text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                chunks.append(text)
    return "".join(chunks)


def _complete_chat_completions(
    client: OpenAI,
    config: RuntimeConfig,
    input_text: str,
    instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> str:
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": input_text})
    request = {"model": config.model, "messages": messages, "stream": False}
    request.update(_provider_request_options(config, api_mode="chat", streaming=False))
    request["model"] = config.model
    request["messages"] = messages
    request["stream"] = False
    if max_output_tokens:
        request.pop("max_completion_tokens", None)
        request["max_tokens"] = max_output_tokens
    if reasoning_effort is not None:
        normalized_effort = str(reasoning_effort).strip().lower()
        if normalized_effort and normalized_effort != "none":
            request["reasoning_effort"] = normalized_effort
        else:
            request.pop("reasoning_effort", None)
    if temperature is not None:
        request["temperature"] = temperature
    response = client.chat.completions.create(**request)
    choices = getattr(response, "choices", None) or []
    if not choices:
        _log_empty_chat_completion_response(response, request, reason="no_choices")
        return ""
    message = getattr(choices[0], "message", None)
    content = (getattr(message, "content", "") if message else "") or ""
    if not str(content).strip():
        _log_empty_chat_completion_response(response, request, reason="empty_message_content")
    return content


def _log_empty_chat_completion_response(response, request: dict[str, object], *, reason: str) -> None:
    logger.warning(
        "CHAT_COMPLETION_EMPTY_RESPONSE reason=%s request=%s structure=%s",
        reason,
        _summarize_chat_completion_request(request),
        _summarize_chat_completion_response(response),
    )


def _summarize_chat_completion_request(request: dict[str, object]) -> dict[str, object]:
    messages = request.get("messages", [])
    summary: dict[str, object] = {
        "model": _safe_scalar(request.get("model")),
        "stream": _safe_scalar(request.get("stream")),
        "max_tokens": _safe_scalar(request.get("max_tokens")),
        "messages_type": type(messages).__name__,
        "messages_len": len(messages) if isinstance(messages, list) else None,
        "messages": [],
    }
    if isinstance(messages, list):
        summary["messages"] = [
            {
                "role": _safe_scalar(message.get("role")) if isinstance(message, dict) else "",
                "content_type": type(message.get("content")).__name__ if isinstance(message, dict) else type(message).__name__,
                "content_len": len(message.get("content")) if isinstance(message, dict) and isinstance(message.get("content"), str) else None,
                "content_preview": _preview_text(message.get("content"), limit=500) if isinstance(message, dict) else _preview_text(message, limit=500),
            }
            for message in messages[:5]
        ]
    return summary


def _summarize_chat_completion_response(response) -> dict[str, object]:
    choices = getattr(response, "choices", None)
    summary: dict[str, object] = {
        "response_type": type(response).__name__,
        "id": _safe_scalar(getattr(response, "id", "")),
        "model": _safe_scalar(getattr(response, "model", "")),
        "object": _safe_scalar(getattr(response, "object", "")),
        "created": _safe_scalar(getattr(response, "created", "")),
        "usage": _safe_mapping(getattr(response, "usage", None)),
        "choices_type": type(choices).__name__ if choices is not None else None,
        "choices_len": len(choices) if isinstance(choices, list) else None,
        "choices": [],
    }
    if isinstance(choices, list):
        summarized_choices = []
        for choice in choices[:3]:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            summarized_choices.append(
                {
                    "choice_type": type(choice).__name__,
                    "index": _safe_scalar(getattr(choice, "index", "")),
                    "finish_reason": _safe_scalar(getattr(choice, "finish_reason", "")),
                    "message_type": type(message).__name__ if message is not None else None,
                    "message_role": _safe_scalar(getattr(message, "role", "")) if message is not None else "",
                    "content_type": type(content).__name__,
                    "content_len": len(content) if isinstance(content, str) else None,
                    "content_preview": _preview_text(content),
                    "tool_calls_len": _safe_len(getattr(message, "tool_calls", None)) if message is not None else None,
                    "function_call_present": bool(getattr(message, "function_call", None)) if message is not None else False,
                }
            )
        summary["choices"] = summarized_choices
    return summary


def _safe_mapping(value) -> dict[str, object] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {str(key): _safe_scalar(item) for key, item in value.items()}
    return {"type": type(value).__name__}


def _safe_scalar(value) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _preview_text(value, limit=120)
    return str(type(value).__name__)


def _preview_text(value, *, limit: int = 120) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[:limit]}..."


def _safe_len(value) -> int | None:
    try:
        return len(value) if value is not None else None
    except TypeError:
        return None


def _stream_chat_completions(client: OpenAI, config: RuntimeConfig, input_text: str, instructions: str = "") -> Iterable[tuple[str, str]]:
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": input_text})

    response_id = ""
    request = {"model": config.model, "messages": messages, "stream": True}
    request.update(_provider_request_options(config, api_mode="chat", streaming=True))
    request["model"] = config.model
    request["messages"] = messages
    request["stream"] = True
    stream = client.chat.completions.create(**request)
    for chunk in stream:
        response_id = getattr(chunk, "id", "") or response_id
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", "") if delta else ""
        if content:
            yield "delta", content
    if response_id:
        yield "response_id", response_id


def _active_provider(config: RuntimeConfig) -> str:
    return str(getattr(config, "active_provider", "openai") or "openai").strip().lower()


def _validate_config(config: RuntimeConfig) -> None:
    provider = _active_provider(config)
    if not config.model:
        raise ValueError("モデルが未設定です。config.tomlまたはconfig.yamlに model または default_model を設定してください。")
    if provider in {"openai", "openrouter", "azure"} and not config.api_key:
        raise ValueError(f"{provider} のAPIキーが未設定です。設定ファイルまたは環境変数を設定してください。")
    if provider == "openai_compatible" and not config.base_url:
        raise ValueError("openai_compatible の base_url が未設定です。providers.openai_compatible.base_url を設定してください。")
    if provider == "azure" and not getattr(config, "azure_endpoint", ""):
        raise ValueError("azure の endpoint が未設定です。providers.azure.azure_endpoint または AZURE_OPENAI_ENDPOINT を設定してください。")
    if provider == "bedrock" and not getattr(config, "bedrock_region", ""):
        raise ValueError("bedrock の region が未設定です。providers.bedrock.region または AWS_REGION を設定してください。")


def _build_openai_compatible_client(config: RuntimeConfig) -> OpenAI:
    provider = _active_provider(config)
    api_key = config.api_key
    if provider in {"openai_compatible", "ollama", "lmstudio"} and not api_key:
        api_key = provider
    kwargs = {"api_key": api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    headers = _provider_headers(config)
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


_RESERVED_REQUEST_KEYS = {"model", "messages", "input", "instructions", "stream"}


def _provider_request_options(config: RuntimeConfig, *, api_mode: str, streaming: bool) -> dict[str, object]:
    provider = _active_provider(config)
    if provider not in OPENAI_COMPATIBLE_PROVIDERS:
        return {}
    if not hasattr(config, "provider_config"):
        return {}
    provider_config = config.provider_config(provider)
    raw_request = provider_config.get("request", {})
    if not isinstance(raw_request, dict):
        return {}
    request = {str(key): value for key, value in raw_request.items() if str(key) not in _RESERVED_REQUEST_KEYS}
    if api_mode != "chat" or not streaming:
        request.pop("stream_options", None)
    if api_mode == "chat" and str(request.get("reasoning_effort", "")).strip().lower() == "none":
        request.pop("reasoning_effort", None)
    return request


def _provider_headers(config: RuntimeConfig) -> dict[str, str]:
    provider_config = config.provider_config(_active_provider(config)) if hasattr(config, "provider_config") else {}
    raw_headers = provider_config.get("headers") or provider_config.get("default_headers") or {}
    headers = {str(key): str(value) for key, value in raw_headers.items()} if isinstance(raw_headers, dict) else {}
    if _active_provider(config) == "openrouter":
        referer = provider_config.get("http_referer") or provider_config.get("referer")
        title = provider_config.get("x_title") or provider_config.get("title")
        if referer:
            headers.setdefault("HTTP-Referer", str(referer))
        if title:
            headers.setdefault("X-Title", str(title))
    return headers


def _build_azure_client(config: RuntimeConfig) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=config.api_key,
        azure_endpoint=config.azure_endpoint,
        api_version=config.azure_api_version,
    )


def _bedrock_client(config: RuntimeConfig):
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("AWS Bedrockを利用するには boto3 をインストールしてください。") from exc

    kwargs = {}
    if config.bedrock_region:
        kwargs["region_name"] = config.bedrock_region
    credentials = getattr(config, "bedrock_credentials", {})
    if isinstance(credentials, dict):
        kwargs.update(credentials)
    if config.bedrock_profile:
        session = boto3.Session(profile_name=config.bedrock_profile)
        return session.client("bedrock-runtime", **kwargs)
    return boto3.client("bedrock-runtime", **kwargs)


def _bedrock_messages(input_text: str, instructions: str = "") -> tuple[list[dict[str, object]], list[dict[str, str]] | None]:
    messages = [{"role": "user", "content": [{"text": input_text}]}]
    system = [{"text": instructions}] if instructions else None
    return messages, system


def _complete_bedrock(config: RuntimeConfig, input_text: str, instructions: str = "", max_output_tokens: int | None = None) -> str:
    client = _bedrock_client(config)
    messages, system = _bedrock_messages(input_text, instructions)
    request = {"modelId": config.model, "messages": messages}
    if system:
        request["system"] = system
    if max_output_tokens:
        request["inferenceConfig"] = {"maxTokens": max_output_tokens}
    response = client.converse(**request)
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))


def _stream_bedrock(config: RuntimeConfig, input_text: str, instructions: str = "") -> Iterable[tuple[str, str]]:
    client = _bedrock_client(config)
    messages, system = _bedrock_messages(input_text, instructions)
    request = {"modelId": config.model, "messages": messages}
    if system:
        request["system"] = system
    response = client.converse_stream(**request)
    response_id = response.get("ResponseMetadata", {}).get("RequestId", "")
    for event in response.get("stream", []):
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        text = delta.get("text", "")
        if text:
            yield "delta", text
    if response_id:
        yield "response_id", response_id


def _extract_code(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    import re

    match = re.search(r"```(?:python)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped
