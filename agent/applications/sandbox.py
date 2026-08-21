import ast
import json
import logging
import re
from pathlib import Path

from ..access import is_path_allowed
from ..models import Thread
from ..openai_client import generate_sandbox_code
from ..tooling import SandboxDataset, make_sandbox_dataset, run_sandbox, sandbox_dataset_manifest
from .llm_helpers import _llm_response_max_retries

logger = logging.getLogger("agent")

SANDBOX_DATASET_MAX_CHARS = 500_000
SANDBOX_DATASET_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".txt": "text",
    ".md": "text",
}


def _format_sandbox_message(ok: bool, output: str) -> str:
    status = "成功" if ok else "失敗"
    output = _strip_sandbox_artifact_payloads(output).strip()
    if ok and not output:
        output = "成果物を生成しました。"
    return f"Sandbox実行結果: {status}\n\n```text\n{output}\n```"


def _strip_sandbox_artifact_payloads(output: str) -> str:
    output = _strip_marked_sandbox_artifact_payloads(output)

    def strip_json_block(match):
        block = match.group(1).strip()
        return "" if _is_sandbox_artifact_payload(block) else match.group(0)

    text = re.sub(r"```json\s*(.*?)```", strip_json_block, output, flags=re.DOTALL | re.IGNORECASE)
    lines = []
    for line in text.splitlines():
        if _is_sandbox_artifact_payload(line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _strip_marked_sandbox_artifact_payloads(output: str) -> str:
    marker = "<MAIGENT_ARTIFACT>"
    text = output
    while marker in text:
        marker_index = text.find(marker)
        object_start = text.find("{", marker_index + len(marker))
        if object_start < 0:
            break
        decoder = json.JSONDecoder()
        try:
            payload, object_end = decoder.raw_decode(text[object_start:])
        except json.JSONDecodeError:
            break
        if not (isinstance(payload, dict) and isinstance(payload.get("maigent_artifacts"), list)):
            break
        text = text[:marker_index] + text[object_start + object_end :]
    return text


def _is_sandbox_artifact_payload(text: str) -> bool:
    if not text or ("maigent_artifacts" not in text and "maigent_sandbox_result" not in text):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("maigent_artifacts"), list):
        return True
    result = payload.get("maigent_sandbox_result")
    return isinstance(result, dict) and isinstance(result.get("artifacts"), list)


def _is_sandbox_result_adequate(ok: bool, output: str) -> bool:
    return ok and bool(output.strip()) and "traceback" not in output.lower()


def _generate_sandbox_code_with_retries(config, input_text: str) -> str:
    if getattr(config, "sandbox_code_generation_enabled", True) is False:
        logger.debug("sandbox_code_generation_skipped reason=disabled")
        return ""
    attempts = _llm_response_max_retries(config, "sandbox_code_generation") + 1
    last_error = ""
    prompt = input_text
    for attempt in range(1, attempts + 1):
        try:
            generated_code = str(generate_sandbox_code(config, prompt) or "").strip()
        except Exception as exc:
            last_error = str(exc)
            logger.exception("sandbox_code_generation_error attempt=%s/%s", attempt, attempts)
            continue
        if generated_code:
            policy_error = _sandbox_code_policy_violation(generated_code)
            if policy_error:
                last_error = policy_error
                logger.debug(
                    "sandbox_code_generation_policy_rejected attempt=%s/%s reason=%s code_preview=%r",
                    attempt,
                    attempts,
                    policy_error,
                    generated_code[:240],
                )
                prompt = (
                    input_text
                    + "\n\nPrevious generated code was rejected before execution.\n"
                    + f"Reason: {policy_error}\n"
                    + "Regenerate executable Python that reads only embedded RAG/message text, never local files. "
                    + "When host-provided sandbox datasets are listed, use load_dataset(dataset_id) and do not paste rows "
                    + "into Python string literals. "
                    + "For images, render to io.BytesIO, base64-encode the image bytes, and print one "
                    + "maigent_sandbox_result JSON object with artifacts[].content_base64. Do not call os.path.exists, open, "
                    + "Path.read_text, pandas read_* with file paths, or savefig with a filesystem path.\n"
                )
                continue
            if attempt > 1:
                logger.debug("sandbox_code_generation_retry_succeeded attempt=%s/%s", attempt, attempts)
            return generated_code
        logger.debug("sandbox_code_generation_empty_response attempt=%s/%s", attempt, attempts)
    if last_error:
        logger.debug("sandbox_code_generation_failed_after_retries attempts=%s last_error=%r", attempts, last_error[:240])
    return ""


def _retry_generated_sandbox_after_execution_failure(
    config,
    input_text: str,
    generated_code: str,
    result,
    thread_id: int,
    datasets: tuple[SandboxDataset, ...] = (),
):
    if getattr(config, "sandbox_code_generation_enabled", True) is False:
        return result, generated_code
    attempts = _llm_response_max_retries(config, "sandbox_code_generation")
    for attempt in range(1, attempts + 1):
        prompt = (
            input_text
            + "\n\nPrevious generated sandbox code failed during execution.\n"
            + "Previous code:\n"
            + "```python\n"
            + generated_code
            + "\n```\n"
            + "Execution output / traceback:\n"
            + "```text\n"
            + result.output[:4000]
            + "\n```\n"
            + "Regenerate corrected executable Python. Fix missing imports and runtime errors. "
            + "Use only embedded RAG/message text, never local files. When host-provided sandbox datasets are listed, "
            + "use load_dataset(dataset_id) and do not paste rows into Python string literals. For images, render to io.BytesIO, "
            + "base64-encode the image bytes, and print one maigent_sandbox_result JSON object with "
            + "artifacts[].content_base64.\n"
        )
        logger.debug(
            "sandbox_code_execution_retry_start thread_id=%s attempt=%s/%s output_preview=%r",
            thread_id,
            attempt,
            attempts,
            result.output[:240],
        )
        repaired_code = _generate_sandbox_code_with_retries(config, prompt)
        if not repaired_code.strip():
            logger.debug(
                "sandbox_code_execution_retry_no_code thread_id=%s attempt=%s/%s",
                thread_id,
                attempt,
                attempts,
            )
            continue
        generated_code = repaired_code
        result = run_sandbox(input_text, config, code=generated_code, datasets=datasets)
        if _is_sandbox_result_adequate(result.ok, result.output):
            logger.debug(
                "sandbox_code_execution_retry_succeeded thread_id=%s attempt=%s/%s",
                thread_id,
                attempt,
                attempts,
            )
            break
    return result, generated_code


def _sandbox_code_policy_violation(code: str) -> str:
    if "maigent_artifacts" in code or "<MAIGENT_ARTIFACT>" in code:
        return "legacy artifact payload format is not allowed; use maigent_sandbox_result"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    if _contains_embedded_tabular_literal(tree):
        return "embedded tabular data copies are not allowed; use load_dataset(dataset_id)"
    string_assignments = _constant_string_assignments(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in {"open", "Path.open", "Path.read_text", "Path.read_bytes"}:
            return f"local file access is not allowed: {name}"
        if name in {
            "os.path.exists",
            "os.path.isfile",
            "os.path.isdir",
            "Path.exists",
            "Path.is_file",
            "Path.is_dir",
        } and node.args and _string_argument_value(node.args[0], string_assignments) is not None:
            return f"local filesystem checks are not allowed: {name}"
        if name in {
            "pd.read_csv",
            "pandas.read_csv",
            "read_csv",
            "pd.read_table",
            "pandas.read_table",
            "read_table",
            "pd.read_excel",
            "pandas.read_excel",
            "read_excel",
        } and node.args and _string_argument_value(node.args[0], string_assignments) is not None:
            return f"local file reads are not allowed: {name}"
        if name in {"plt.savefig", "matplotlib.pyplot.savefig", "savefig"}:
            if node.args and _string_argument_value(node.args[0], string_assignments) is not None:
                return f"sandbox file writes are not allowed: {name}"
            for keyword in node.keywords:
                if keyword.arg == "fname" and _string_argument_value(keyword.value, string_assignments) is not None:
                    return f"sandbox file writes are not allowed: {name}"
    return ""


def _contains_embedded_tabular_literal(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        lines = [line for line in node.value.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        comma_like_lines = sum(1 for line in lines if "," in line or "，" in line or "\t" in line)
        if comma_like_lines >= 3:
            return True
    return False


def _constant_string_assignments(tree: ast.AST) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                assignments[node.target.id] = node.value.value
    return assignments


def _string_argument_value(node: ast.AST, assignments: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return assignments.get(node.id)
    return None


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    if isinstance(func, ast.Call):
        return _call_name(func.func)
    return ""


def _append_sandbox_dataset_manifest(input_text: str, datasets: tuple[SandboxDataset, ...]) -> str:
    manifest = sandbox_dataset_manifest(datasets)
    if not manifest:
        return input_text
    return (
        input_text
        + "\n\nHost-provided sandbox dataset API:\n"
        + manifest
        + "\n\nGenerated code must use load_dataset(\"rag_1\") or another listed dataset id for these files. "
        + "Do not paste CSV/TSV/JSON rows into Python string literals."
    )


def _sandbox_datasets_from_rag_context(thread: Thread, input_text: str) -> tuple[SandboxDataset, ...]:
    datasets: list[SandboxDataset] = []
    for path_text in _rag_context_file_paths(input_text):
        if len(datasets) >= 5:
            break
        try:
            path = Path(path_text).expanduser().resolve()
        except OSError:
            continue
        kind = SANDBOX_DATASET_EXTENSIONS.get(path.suffix.lower())
        if not kind or not is_path_allowed(thread.project, str(path), write=False):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        truncated = len(text) > SANDBOX_DATASET_MAX_CHARS
        if truncated:
            text = text[:SANDBOX_DATASET_MAX_CHARS]
        datasets.append(
            make_sandbox_dataset(
                f"rag_{len(datasets) + 1}",
                path.name,
                str(path),
                kind,
                text,
                truncated=truncated,
            )
        )
    if datasets:
        logger.debug(
            "sandbox_datasets_built thread_id=%s datasets=%s",
            thread.id,
            [{"id": item.id, "name": item.name, "kind": item.kind, "rows": item.row_count} for item in datasets],
        )
    return tuple(datasets)


def _rag_context_file_paths(input_text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"^(?:Auto-selected file|File):\s+(.+?)\s*$", input_text, flags=re.MULTILINE):
        path = match.group(1).strip()
        if path and path not in paths:
            paths.append(path)
    return paths
