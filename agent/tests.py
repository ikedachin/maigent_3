import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from .access import is_path_allowed, normalize_access_path
from .config import RuntimeConfig, load_agents_md, load_runtime_config, load_skills
from .openai_client import stream_response
from .models import AgentRun, AgentTaskRecord, AgentWorkerRun, AppSetting, ApprovalRequest, Automation, FeatureFlag, Message, Project, ProjectAccessPath, Thread
from .tooling import (
    AgentPlan,
    AgentPlanStep,
    SandboxResult,
    build_agent_worker_specs,
    _parse_typed_sandbox_output,
    _prepend_sandbox_runtime_prelude,
    _prepend_sandbox_datasets,
    make_sandbox_dataset,
    _python_for_tabular_task,
    build_agent_plan,
    evaluation_criterion,
    can_build_sandbox_program,
    requires_llm_sandbox_program,
    run_sandbox,
    select_tool,
)
from .applications.rag import (
    _collect_candidate_documents,
    _extract_attachment_paths,
    _format_file_list_for_display,
    _search_terms,
)
from .applications.planning import _available_tool_specs, _initial_clarifier_max_output_tokens, _parse_plan_tasks
from .applications.llm_helpers import _control_config_max_output_tokens, _final_evaluation_max_output_tokens
from .applications.sandbox import (
    _generate_sandbox_code_with_retries,
    _sandbox_code_policy_violation,
    _strip_sandbox_artifact_payloads,
)
from .views import (
    _allowed_plan_tools,
    _avoid_failed_plan,
    _evaluate_final_answer,
    _multi_agent_enabled,
    _persist_sandbox_artifacts,
    _request_initial_clarification_if_needed,
    _serialize_plan_queue,
    _build_instructions,
    _build_file_batch_input,
    _execute_multi_agent_plan,
)
from .views import AgentState, TaskExecutionRecord, _execute_agent_plan, _replan_after_step, _route_final_output


@override_settings(STATICFILES_DIRS=[])
class ConfigTests(TestCase):
    def test_app_config_overrides_user_config(self):
        with TemporaryDirectory() as home_dir, TemporaryDirectory() as app_dir:
            home = Path(home_dir)
            app = Path(app_dir)
            (home / ".maigent").mkdir()
            (app / ".maigent").mkdir()
            (home / ".maigent" / "config.toml").write_text('model = "user-model"\napi_key = "user-key"\n')
            (app / ".maigent" / "config.toml").write_text('model = "app-model"\nbase_url = "http://local"\n')

            with patch("agent.config.Path.home", return_value=home), override_settings(BASE_DIR=app):
                config = load_runtime_config("")

            self.assertEqual(config.model, "app-model")
            self.assertEqual(config.api_key, "user-key")
            self.assertEqual(config.base_url, "http://local")
            self.assertEqual(len(config.sources), 2)

    def test_project_config_overrides_app_config(self):
        with TemporaryDirectory() as home_dir, TemporaryDirectory() as app_dir, TemporaryDirectory() as project_dir:
            home = Path(home_dir)
            app = Path(app_dir)
            project = Path(project_dir)
            (app / ".maigent").mkdir()
            (project / ".maigent").mkdir()
            (app / ".maigent" / "config.toml").write_text('model = "app-model"\napi_key = "app-key"\n')
            (project / ".maigent" / "config.toml").write_text('model = "project-model"\nbase_url = "http://project"\n')

            with patch("agent.config.Path.home", return_value=home), override_settings(BASE_DIR=app):
                config = load_runtime_config(str(project))

            self.assertEqual(config.model, "project-model")
            self.assertEqual(config.api_key, "app-key")
            self.assertEqual(config.base_url, "http://project")
            self.assertEqual(len(config.sources), 2)

    def test_api_mode_defaults_to_chat_and_preserves_explicit_auto(self):
        with TemporaryDirectory() as app_dir:
            app = Path(app_dir)
            (app / ".maigent").mkdir()
            (app / ".maigent" / "config.toml").write_text('model = "m"\n')

            with override_settings(BASE_DIR=app):
                config = load_runtime_config("")

            self.assertEqual(config.api_mode, "chat")

        explicit = RuntimeConfig(values={"model": "m", "api_key": "key", "api_mode": "auto"}, sources=[])
        self.assertEqual(explicit.api_mode, "auto")

    def test_openai_compatible_is_selected_after_openai(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "openai": {"enabled": False, "model": "gpt"},
                    "openai_compatible": {
                        "enabled": True,
                        "model": "Qwen/Qwen3.8-27B",
                        "base_url": "http://localhost:8000/v1",
                    },
                    "ollama": {"enabled": True, "model": "llama3.1"},
                }
            },
            sources=[],
        )

        self.assertEqual(config.active_provider, "openai_compatible")
        self.assertEqual(config.model, "Qwen/Qwen3.8-27B")
        self.assertEqual(config.base_url, "http://localhost:8000/v1")
        self.assertEqual(config.api_mode, "chat")
        self.assertEqual(config.api_key, "")

    def test_provider_config_selects_first_enabled_provider(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "openai": {"enabled": False, "model": "gpt"},
                    "ollama": {"enabled": True, "model": "llama3.1"},
                    "openrouter": {"enabled": True, "model": "openai/gpt-4o-mini", "api_key": "router-key"},
                }
            },
            sources=[],
        )

        self.assertEqual(config.active_provider, "ollama")
        self.assertEqual(config.model, "llama3.1")
        self.assertEqual(config.base_url, "http://localhost:11434/v1")
        self.assertEqual(config.api_mode, "chat")
        self.assertEqual(config.api_key, "")

    def test_openrouter_provider_uses_provider_api_key_and_redacts_it(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "openrouter": {
                        "enabled": True,
                        "model": "openai/gpt-4o-mini",
                        "api_key": "router-key",
                    }
                }
            },
            sources=[],
        )

        self.assertEqual(config.active_provider, "openrouter")
        self.assertEqual(config.api_key, "router-key")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.redacted()["providers"]["openrouter"]["api_key"], "********")

    def test_bedrock_provider_exposes_credentials_and_redacts_them(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "bedrock": {
                        "enabled": True,
                        "model": "anthropic.model",
                        "region": "ap-northeast-1",
                        "aws_access_key_id": "access",
                        "aws_secret_access_key": "secret",
                        "aws_session_token": "token",
                    }
                }
            },
            sources=[],
        )

        self.assertEqual(
            config.bedrock_credentials,
            {
                "aws_access_key_id": "access",
                "aws_secret_access_key": "secret",
                "aws_session_token": "token",
            },
        )
        safe = config.redacted()["providers"]["bedrock"]
        self.assertEqual(safe["aws_access_key_id"], "********")
        self.assertEqual(safe["aws_secret_access_key"], "********")
        self.assertEqual(safe["aws_session_token"], "********")

    def test_all_providers_disabled_disables_model_resolution(self):
        config = RuntimeConfig(
            values={
                "model": "legacy-model",
                "providers": {
                    "openai": {"enabled": False, "model": "gpt"},
                    "bedrock": {"enabled": False, "model": "anthropic.model"},
                },
            },
            sources=[],
        )

        self.assertEqual(config.active_provider, "")
        self.assertEqual(config.model, "")

    def test_yaml_config_loads_tools_and_sandbox_libraries(self):
        with TemporaryDirectory() as app_dir:
            app = Path(app_dir)
            (app / ".maigent").mkdir()
            (app / ".maigent" / "config.yaml").write_text(
                "\n".join(
                    [
                        "final_evaluation:",
                        "  enabled: true",
                        "  max_retries: 2",
                        "  llm_max_retries: 2",
                        "  max_output_tokens: 72",
                        "  reasoning_effort: low",
                        "llm:",
                        "  max_retries: 1",
                        "logging:",
                        "  llm_tail_chars: full",
                        "tools:",
                        "  rag:",
                        "    enabled: true",
                        "  sandbox:",
                        "    enabled: true",
                        "    image: python:3.12-slim",
                        "    timeout_seconds: 300",
                        "    memory_limit_mb: 1024",
                        "    pids_limit: 256",
                        "    cpus: 2",
                        "    install_libraries_on_run: true",
                        "    allowed_libraries:",
                        "      - numpy",
                        "      - pandas",
                        "tool_selector:",
                        "  enabled: true",
                        "  max_output_tokens: 96",
                        "  reasoning_effort: none",
                        "  max_retries: 2",
                        "initial_clarifier:",
                        "  enabled: true",
                        "  max_output_tokens: 192",
                        "  reasoning_effort: low",
                        "  llm_max_retries: 2",
                        "dynamic_replanner:",
                        "  enabled: true",
                        "  max_output_tokens: 128",
                        "  llm_max_retries: 2",
                        "  reasoning_effort: true",
                        "dynamic_finalizer:",
                        "  enabled: true",
                        "  max_output_tokens: 80",
                        "  llm_max_retries: 2",
                        "  reasoning_effort: false",
                        "sandbox_code_generation:",
                        "  enabled: true",
                        "  max_output_tokens: 32768",
                        "  llm_max_retries: 3",
                        "  reasoning_effort: xhigh",
                    ]
                ),
                encoding="utf-8",
            )

            with override_settings(BASE_DIR=app):
                config = load_runtime_config("")

            self.assertTrue(config.tool_enabled("sandbox"))
            self.assertEqual(config.sandbox_image, "python:3.12-slim")
            self.assertEqual(config.sandbox_timeout_seconds, 300)
            self.assertEqual(config.sandbox_memory_limit_mb, 1024)
            self.assertEqual(config.sandbox_pids_limit, 256)
            self.assertEqual(config.sandbox_cpus, 2.0)
            self.assertTrue(config.sandbox_install_libraries_on_run)
            self.assertEqual(config.sandbox_allowed_libraries, ["numpy", "pandas"])
            self.assertTrue(config.final_evaluation_enabled)
            self.assertEqual(config.final_evaluation_max_retries, 2)
            self.assertEqual(config.final_evaluation_max_output_tokens, 72)
            self.assertEqual(config.final_evaluation_reasoning_effort, "low")
            self.assertIsNone(config.llm_log_tail_chars)
            self.assertTrue(config.tool_enabled("tool_selector"))
            self.assertTrue(config.tool_enabled("initial_clarifier"))
            self.assertTrue(config.tool_enabled("dynamic_replanner"))
            self.assertTrue(config.tool_enabled("dynamic_finalizer"))
            self.assertEqual(config.control_config("tool_selector")["max_retries"], 2)
            self.assertEqual(config.control_config("initial_clarifier")["max_output_tokens"], 192)
            self.assertEqual(config.control_config("initial_clarifier")["llm_max_retries"], 2)
            self.assertEqual(config.final_evaluation["llm_max_retries"], 2)
            self.assertEqual(config.control_config("dynamic_replanner")["llm_max_retries"], 2)
            self.assertTrue(config.sandbox_code_generation_enabled)
            self.assertEqual(config.sandbox_code_generation_max_output_tokens, 32768)
            self.assertEqual(config.sandbox_code_generation_reasoning_effort, "xhigh")
            self.assertEqual(config.sandbox_code_generation["llm_max_retries"], 3)
            self.assertEqual(config.enabled_tool_names, {"rag", "file_batch", "sandbox"})
            self.assertTrue(config.tool_enabled("file_batch"))
            self.assertFalse(config.tool_enabled("web_search"))

    def test_runtime_config_sandbox_resource_limits_default_and_clamp(self):
        default_config = RuntimeConfig(values={}, sources=[])

        self.assertEqual(default_config.sandbox_memory_limit_mb, 512)
        self.assertEqual(default_config.sandbox_pids_limit, 128)
        self.assertEqual(default_config.sandbox_cpus, 1.0)

        clamped_config = RuntimeConfig(
            values={"tools": {"sandbox": {"memory_limit_mb": 999999, "pids_limit": 999999, "cpus": 999}}},
            sources=[],
        )

        self.assertEqual(clamped_config.sandbox_memory_limit_mb, 8192)
        self.assertEqual(clamped_config.sandbox_pids_limit, 2048)
        self.assertEqual(clamped_config.sandbox_cpus, 8.0)

    def test_runtime_config_web_search_settings_default_and_clamp(self):
        default_config = RuntimeConfig(values={}, sources=[])

        self.assertEqual(default_config.web_search_api_key, "")
        self.assertEqual(default_config.web_search_max_results, 5)
        self.assertEqual(default_config.web_search_timeout_seconds, 10)
        self.assertEqual(default_config.web_search_max_retries, 1)

        configured = RuntimeConfig(
            values={"tools": {"web_search": {"api_key": "tvly-abc", "max_results": 999, "timeout_seconds": 999, "max_retries": 999}}},
            sources=[],
        )

        self.assertEqual(configured.web_search_api_key, "tvly-abc")
        self.assertEqual(configured.web_search_max_results, 10)
        self.assertEqual(configured.web_search_timeout_seconds, 30)
        self.assertEqual(configured.web_search_max_retries, 5)

    def test_runtime_config_web_search_api_key_falls_back_to_env(self):
        import os

        config = RuntimeConfig(values={"tools": {"web_search": {}}}, sources=[])

        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-env-key"}):
            self.assertEqual(config.web_search_api_key, "tvly-env-key")

    def test_runtime_config_tools_must_be_declared_in_yaml_tools(self):
        config = RuntimeConfig(values={"tools": {"sandbox": {"enabled": True}}}, sources=[])

        self.assertEqual(config.enabled_tool_names, {"sandbox"})
        self.assertTrue(config.tool_enabled("sandbox"))
        self.assertFalse(config.tool_enabled("rag", default=True))
        self.assertFalse(config.tool_enabled("web_search", default=True))
        self.assertFalse(config.tool_enabled("file_batch", default=True))

    def test_runtime_config_enables_file_batch_with_existing_rag_config(self):
        config = RuntimeConfig(values={"tools": {"rag": {"enabled": True}}}, sources=[])
        disabled = RuntimeConfig(values={"tools": {"rag": {"enabled": True}, "file_batch": {"enabled": False}}}, sources=[])

        self.assertTrue(config.tool_enabled("file_batch"))
        self.assertIn("file_batch", config.enabled_tool_names)
        self.assertFalse(disabled.tool_enabled("file_batch"))
        self.assertNotIn("file_batch", disabled.enabled_tool_names)

    def test_initial_clarifier_max_output_tokens_defaults_to_8192_and_uses_config_value(self):
        default_config = RuntimeConfig(values={}, sources=[])
        configured = RuntimeConfig(values={"initial_clarifier": {"max_output_tokens": 32768}}, sources=[])
        invalid = RuntimeConfig(values={"initial_clarifier": {"max_output_tokens": "invalid"}}, sources=[])

        self.assertEqual(_initial_clarifier_max_output_tokens(default_config), 8192)
        self.assertEqual(_initial_clarifier_max_output_tokens(configured), 32768)
        self.assertEqual(_initial_clarifier_max_output_tokens(invalid), 8192)

    def test_dynamic_max_output_tokens_defaults_to_8192_and_uses_config_value(self):
        default_config = RuntimeConfig(values={}, sources=[])
        configured = RuntimeConfig(values={"dynamic_replanner": {"max_output_tokens": 32768}}, sources=[])
        invalid = RuntimeConfig(values={"dynamic_finalizer": {"max_output_tokens": "invalid"}}, sources=[])

        self.assertEqual(_control_config_max_output_tokens(default_config, "dynamic_replanner"), 8192)
        self.assertEqual(_control_config_max_output_tokens(configured, "dynamic_replanner"), 32768)
        self.assertEqual(_control_config_max_output_tokens(invalid, "dynamic_finalizer"), 8192)

    def test_final_evaluation_max_output_tokens_defaults_to_8192_and_uses_config_value(self):
        default_config = RuntimeConfig(values={}, sources=[])
        configured = RuntimeConfig(values={"final_evaluation": {"max_output_tokens": 32768}}, sources=[])
        invalid = RuntimeConfig(values={"final_evaluation": {"max_output_tokens": "invalid"}}, sources=[])

        self.assertEqual(_final_evaluation_max_output_tokens(default_config), 8192)
        self.assertEqual(configured.final_evaluation_max_output_tokens, 32768)
        self.assertEqual(_final_evaluation_max_output_tokens(configured), 32768)
        self.assertEqual(invalid.final_evaluation_max_output_tokens, 8192)

    def test_plan_task_parser_and_storage_filter_to_allowed_tools(self):
        allowed = {"sandbox", "final"}
        tasks = _parse_plan_tasks(
            [
                {"tool": "rag", "purpose": "Search context."},
                {"tool": "sandbox", "purpose": "Run calculation."},
                {"tool": "unknown", "purpose": "Do not run."},
                {"tool": "final", "purpose": "Answer."},
            ],
            allowed,
        )

        self.assertEqual([task.tool for task in tasks], ["sandbox", "final"])
        serialized = _serialize_plan_queue(
            [AgentPlanStep("rag", "Search context."), AgentPlanStep("sandbox", "Run calculation.")],
            allowed,
        )
        self.assertEqual(serialized, [{"tool": "sandbox", "purpose": "Run calculation."}])

    def test_allowed_plan_tools_uses_yaml_tools_plus_internal_final(self):
        config = RuntimeConfig(values={"tools": {"sandbox": {"enabled": True}, "rag": {"enabled": False}}}, sources=[])

        self.assertEqual(_allowed_plan_tools(config), {"sandbox", "final"})

    def test_llm_log_tail_chars_clamps_numeric_config(self):
        config = RuntimeConfig(values={"logging": {"llm_tail_chars": 250000}}, sources=[])

        self.assertEqual(config.llm_log_tail_chars, 200000)

    def test_log_tail_uses_configured_limit_or_full_text(self):
        from .views import _log_tail

        class TailConfig:
            llm_log_tail_chars = 4

        class FullConfig:
            llm_log_tail_chars = None

        with self.assertLogs("agent", level="DEBUG") as tail_logs:
            _log_tail("sample", "abcdef", config=TailConfig())
        self.assertIn("tail='cdef'", tail_logs.output[0])
        self.assertNotIn("tail='abcdef'", tail_logs.output[0])

        with self.assertLogs("agent", level="DEBUG") as full_logs:
            _log_tail("sample", "abcdef", config=FullConfig())
        self.assertIn("tail='abcdef'", full_logs.output[0])

    def test_final_evaluation_config_clamps_retries(self):
        with TemporaryDirectory() as app_dir:
            app = Path(app_dir)
            (app / ".maigent").mkdir()
            (app / ".maigent" / "config.toml").write_text(
                "\n".join(
                    [
                        "[final_evaluation]",
                        "enabled = true",
                        "max_retries = 9",
                        "max_output_tokens = 96",
                        'reasoning_effort = "minimal"',
                    ]
                ),
                encoding="utf-8",
            )

            with override_settings(BASE_DIR=app):
                config = load_runtime_config("")

            self.assertTrue(config.final_evaluation_enabled)
            self.assertEqual(config.final_evaluation_max_retries, 3)
            self.assertEqual(config.final_evaluation_max_output_tokens, 96)
            self.assertEqual(config.final_evaluation_reasoning_effort, "minimal")

    def test_multi_agent_config_defaults_and_clamps_workers(self):
        self.assertTrue(RuntimeConfig(values={}, sources=[]).multi_agent_enabled)
        config = RuntimeConfig(
            values={
                "multi_agent": {
                    "enabled": True,
                    "max_workers": 9,
                    "parallel_tools": False,
                    "progress_visible": False,
                }
            },
            sources=[],
        )

        self.assertTrue(config.multi_agent_enabled)
        self.assertEqual(config.multi_agent_max_workers, 5)
        self.assertFalse(config.multi_agent_parallel_tools)
        self.assertFalse(config.multi_agent_progress_visible)

    def test_multi_agent_can_be_disabled_for_sequential_fallback(self):
        config = RuntimeConfig(values={"multi_agent": {"enabled": False}}, sources=[])

        self.assertFalse(config.multi_agent_enabled)
        self.assertFalse(_multi_agent_enabled(config))


@override_settings(STATICFILES_DIRS=[])
class ChatFlowTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Repo", path="")
        self.thread = Thread.objects.create(project=self.project, title="Main")

    @patch("agent.views.load_runtime_config")
    def test_model_missing_returns_error_without_stream_url(self, mock_config):
        mock_config.return_value.model = ""
        mock_config.return_value.api_key = ""
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "hello"})

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertTrue(payload["error"])
        self.assertNotIn("stream_url", payload)
        self.assertEqual(Message.objects.filter(thread=self.thread).count(), 2)
        self.assertEqual(Message.objects.filter(role="assistant").first().status, "error")

    def test_update_rag_settings_clamps_top_k(self):
        response = self.client.post(
            reverse("update_rag_settings"),
            {
                "rag_top_k": "99",
                "final_evaluation_enabled": "on",
                "final_evaluation_max_retries": "9",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(AppSetting.objects.get(key="rag_top_k").value, "10")
        self.assertEqual(AppSetting.objects.get(key="final_evaluation_enabled").value, "true")
        self.assertEqual(AppSetting.objects.get(key="final_evaluation_max_retries").value, "3")

    def test_dashboard_creates_builtin_file_write_flag(self):
        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 200)
        flag = FeatureFlag.objects.get(name="file_write")
        self.assertTrue(flag.enabled)

    def test_update_feature_flag_toggles_flag(self):
        flag = FeatureFlag.objects.create(name="file_write", enabled=True)

        response = self.client.post(reverse("update_feature_flag", args=[flag.id]), {"action": "disable"})

        self.assertEqual(response.status_code, 302)
        flag.refresh_from_db()
        self.assertFalse(flag.enabled)

    def test_slash_status_returns_assistant_message(self):
        FeatureFlag.objects.create(name="web_search", enabled=True)
        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/status"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["command"])
        self.assertIn("Project: Repo", payload["content"])
        self.assertIn("web_search=on", payload["content"])

    def test_features_command_toggles_flag(self):
        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/features enable beta"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(FeatureFlag.objects.get(name="beta").enabled)

    def test_memories_command_toggles_thread(self):
        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/memories"})

        self.assertEqual(response.status_code, 200)
        self.thread.refresh_from_db()
        self.assertTrue(self.thread.memory_enabled)

    def test_compact_updates_only_current_thread_summary(self):
        other = Thread.objects.create(project=self.project, title="Other", summary="keep this")
        Message.objects.create(thread=other, role="user", content="other thread secret")
        Message.objects.create(thread=self.thread, role="user", content="main thread fact")
        Message.objects.create(thread=self.thread, role="assistant", content="main thread answer")

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/compact"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.thread.refresh_from_db()
        other.refresh_from_db()
        self.assertIn("main thread fact", self.thread.summary)
        self.assertIn("main thread answer", self.thread.summary)
        self.assertNotIn("other thread secret", self.thread.summary)
        self.assertEqual(other.summary, "keep this")
        self.assertIn("スレッド要約を更新しました", payload["content"])
        self.assertIn(self.thread.summary, payload["content"])
        self.assertEqual(payload["thread_summary"], self.thread.summary)

    def test_compact_automatically_enables_thread_memory(self):
        self.assertFalse(self.thread.memory_enabled)
        Message.objects.create(thread=self.thread, role="user", content="main thread fact")

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/compact"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("メモリ: on", response.json()["content"])
        self.thread.refresh_from_db()
        self.assertTrue(self.thread.memory_enabled)

    def test_memories_command_can_disable_after_compact(self):
        Message.objects.create(thread=self.thread, role="user", content="main thread fact")
        self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/compact"})
        self.thread.refresh_from_db()
        self.assertTrue(self.thread.memory_enabled)

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/memories"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "Memories: off")
        self.thread.refresh_from_db()
        self.assertFalse(self.thread.memory_enabled)
        self.assertTrue(self.thread.summary)

    def test_compact_ignores_slash_commands_and_responses(self):
        Message.objects.create(thread=self.thread, role="user", content="/status")
        Message.objects.create(thread=self.thread, role="assistant", content="Project: Repo")
        Message.objects.create(thread=self.thread, role="user", content="remember this")

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/compact"})

        self.assertEqual(response.status_code, 200)
        self.thread.refresh_from_db()
        self.assertIn("remember this", self.thread.summary)
        self.assertNotIn("/status", self.thread.summary)
        self.assertNotIn("Project: Repo", self.thread.summary)
        self.assertNotIn("/compact", self.thread.summary)

    def test_build_instructions_uses_current_thread_memory_only(self):
        other = Thread.objects.create(project=self.project, title="Other", memory_enabled=True, summary="other memory")
        self.thread.memory_enabled = True
        self.thread.summary = "current memory"
        self.thread.save(update_fields=["memory_enabled", "summary"])

        instructions = _build_instructions(self.thread)

        self.assertIn("current memory", instructions)
        self.assertNotIn(other.summary, instructions)

    def test_read_command_reads_allowed_file(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            file_path = root / "allowed.txt"
            file_path.write_text("hello file", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": f"/read {file_path}"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("hello file", response.json()["content"])

    def test_read_command_rejects_unallowed_file(self):
        with TemporaryDirectory() as root_dir:
            file_path = Path(root_dir) / "blocked.txt"
            file_path.write_text("secret", encoding="utf-8")

            response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": f"/read {file_path}"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("読み取りが許可されていません", response.json()["content"])

    def test_ls_command_lists_allowed_folder(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "sub").mkdir()
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": f"/ls {root}"})

        self.assertEqual(response.status_code, 200)
        content = response.json()["content"]
        self.assertIn("[dir] sub", content)
        self.assertIn("[file] a.txt", content)

    def test_file_command_without_path_lists_allowed_paths(self):
        ProjectAccessPath.objects.create(project=self.project, path="/tmp/example", mode="read")

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/file"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("許可済みファイル/フォルダ", response.json()["content"])
        self.assertIn("/tmp/example", response.json()["content"])

    def test_file_command_reads_file_or_lists_folder(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            file_path = root / "note.txt"
            file_path.write_text("note body", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            file_response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": f"/file {file_path}"})
            folder_response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": f"/file {root}"})

        self.assertIn("note body", file_response.json()["content"])
        self.assertIn("[file] note.txt", folder_response.json()["content"])

    def test_write_command_writes_allowed_file(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            file_path = root / "created.txt"
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])

            response = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"/write {file_path} -- hello write"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("書き込みしました", response.json()["content"])
            self.assertEqual(file_path.read_text(encoding="utf-8"), "hello write")

    def test_write_command_resolves_relative_path_inside_output_folder(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])

            response = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": "/write created.txt -- hello write"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("書き込みしました", response.json()["content"])
            self.assertEqual((root / "created.txt").read_text(encoding="utf-8"), "hello write")

    def test_write_command_rejects_disabled_file_write_flag(self):
        FeatureFlag.objects.create(name="file_write", enabled=False)
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            file_path = root / "blocked.txt"
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])

            response = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"/write {file_path} -- blocked"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("ファイル書き込み機能が無効です", response.json()["content"])
            self.assertFalse(file_path.exists())

    def test_write_command_rejects_missing_output_folder(self):
        with TemporaryDirectory() as root_dir:
            file_path = Path(root_dir) / "blocked.txt"

            response = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"/write {file_path} -- blocked"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("書き出し先フォルダが未設定です", response.json()["content"])
            self.assertFalse(file_path.exists())

    def test_write_command_rejects_path_outside_output_folder(self):
        with TemporaryDirectory() as output_dir, TemporaryDirectory() as other_dir:
            self.project.output_path = output_dir
            self.project.save(update_fields=["output_path"])
            file_path = Path(other_dir) / "blocked.txt"

            response = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"/write {file_path} -- blocked"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("書き込みが許可されていません", response.json()["content"])
            self.assertFalse(file_path.exists())

    def test_append_command_appends_allowed_file(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            file_path = root / "note.txt"
            file_path.write_text("first", encoding="utf-8")
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])

            response = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"/append {file_path} -- second"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("追記しました", response.json()["content"])
            self.assertEqual(file_path.read_text(encoding="utf-8"), "firstsecond")

    def test_fork_command_copies_messages(self):
        Message.objects.create(thread=self.thread, role="user", content="before")
        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/fork"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Thread.objects.count(), 2)
        fork = Thread.objects.exclude(id=self.thread.id).get()
        self.assertGreaterEqual(fork.messages.count(), 2)

    @patch("agent.views.load_runtime_config")
    def test_model_command_returns_current_model(self, mock_config):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.sources = []

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/model"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "Current model: test-model")

    def test_resume_command_lists_recent_threads(self):
        other = Thread.objects.create(project=self.project, title="Other thread")

        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/resume"})

        self.assertEqual(response.status_code, 200)
        content = response.json()["content"]
        self.assertIn(f"#{other.id} Other thread", content)
        self.assertIn(f"#{self.thread.id} Main", content)

    def test_status_only_stub_commands_return_placeholder(self):
        for command in ["/experimental", "/agent", "/theme", "/apps"]:
            with self.subTest(command=command):
                response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": command})

                self.assertEqual(response.status_code, 200)
                content = response.json()["content"]
                self.assertIn(command, content)
                self.assertIn("初回版では状態表示のみ", content)

    def test_unknown_slash_command_returns_error_message(self):
        response = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "/nonexistent"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "未対応のスラッシュコマンドです: /nonexistent")

    def test_delete_thread_redirects_to_remaining_thread(self):
        other = Thread.objects.create(project=self.project, title="Other")

        response = self.client.post(reverse("delete_thread", args=[self.thread.id]))

        self.assertEqual(response.status_code, 302)
        self.assertFalse(Thread.objects.filter(id=self.thread.id).exists())
        self.assertEqual(response.url, reverse("thread", args=[other.id]))

    def test_delete_last_thread_recreates_main_thread(self):
        response = self.client.post(reverse("delete_thread", args=[self.thread.id]))

        self.assertEqual(response.status_code, 302)
        replacement = Thread.objects.get(project=self.project)
        self.assertEqual(replacement.title, "Main thread")
        self.assertEqual(response.url, reverse("thread", args=[replacement.id]))

    def test_agent_worker_run_cascades_and_task_record_can_link_worker(self):
        user = Message.objects.create(thread=self.thread, role="user", content="work")
        assistant = Message.objects.create(thread=self.thread, role="assistant", content="")
        run = AgentRun.objects.create(
            thread=self.thread,
            user_message=user,
            assistant_message=assistant,
            goal="goal",
        )
        worker = AgentWorkerRun.objects.create(run=run, name="research", role="research", purpose="Search")
        AgentTaskRecord.objects.create(
            run=run,
            worker=worker,
            sequence=1,
            tool="rag",
            purpose="Search",
            status="ok",
        )

        self.assertEqual(run.worker_runs.get(), worker)
        self.assertEqual(AgentTaskRecord.objects.get(run=run).worker, worker)
        run.delete()
        self.assertFalse(AgentWorkerRun.objects.filter(id=worker.id).exists())

    def test_build_agent_worker_specs_splits_rag_sandbox_and_verify(self):
        plan = AgentPlan(
            goal="Answer",
            evaluation_criteria=[],
            summary="rag -> sandbox -> final",
            steps=[
                AgentPlanStep("rag", "Find context."),
                AgentPlanStep("sandbox", "Compute."),
                AgentPlanStep("final", "Answer."),
            ],
        )

        workers = build_agent_worker_specs(plan, max_workers=3, parallel_tools=True)

        self.assertEqual([worker.name for worker in workers], ["research", "compute", "verify"])
        self.assertEqual([step.tool for step in workers[0].steps], ["rag"])
        self.assertEqual([step.tool for step in workers[1].steps], ["sandbox"])

    def test_build_agent_worker_specs_includes_file_batch_worker(self):
        plan = AgentPlan(
            goal="Answer",
            evaluation_criteria=[],
            summary="file_batch -> final",
            steps=[
                AgentPlanStep("file_batch", "Summarize files."),
                AgentPlanStep("final", "Answer."),
            ],
        )

        workers = build_agent_worker_specs(plan, max_workers=3, parallel_tools=True)

        self.assertEqual([worker.name for worker in workers], ["file_batch"])
        self.assertEqual([step.tool for step in workers[0].steps], ["file_batch"])

    @patch("agent.views._execute_agent_task")
    def test_multi_agent_execution_records_workers_and_synthesizes_results(self, mock_execute_task):
        class Config:
            multi_agent_enabled = True
            multi_agent_max_workers = 3
            multi_agent_parallel_tools = True
            multi_agent_progress_visible = True

            def tool_enabled(self, name, default=False):
                return name in {"rag", "sandbox"}

        sandbox_inputs = []

        def fake_execute_task(thread, user_text, config, input_text, plan_trace, step, preferred_rag_query="", progress=None):
            if progress:
                yield ""
            if step.tool == "rag":
                return {"ok": True, "input_text": input_text + "\nRAG context", "final_message": ""}
            sandbox_inputs.append(input_text)
            return {"ok": True, "input_text": input_text, "final_message": "Sandbox実行結果: 成功\n\n```text\n4\n```"}

        mock_execute_task.side_effect = fake_execute_task
        user = Message.objects.create(thread=self.thread, role="user", content="numbers")
        assistant = Message.objects.create(thread=self.thread, role="assistant", content="")
        plan = AgentPlan(
            goal="Answer numbers",
            evaluation_criteria=["Use worker results."],
            summary="rag -> sandbox -> final",
            steps=[
                AgentPlanStep("rag", "Find context."),
                AgentPlanStep("sandbox", "Compute."),
                AgentPlanStep("final", "Answer."),
            ],
        )
        run = AgentRun.objects.create(
            thread=self.thread,
            user_message=user,
            assistant_message=assistant,
            goal=plan.goal,
            initial_plan_summary=plan.summary,
        )
        events = []

        def progress(message):
            events.append(message)
            return {"progress": message, "progress_tail": events[-3:]}

        runner = _execute_multi_agent_plan(self.thread, "numbers", Config(), plan, progress, run=run, input_text="numbers")
        result = None
        chunks = []
        while True:
            try:
                chunks.append(next(runner))
            except StopIteration as exc:
                result = exc.value
                break

        self.assertTrue(result["ok"])
        self.assertIn("Worker results:", result["input_text"])
        self.assertIn("[research]", result["input_text"])
        self.assertIn("[compute]", result["input_text"])
        self.assertIn('"agent": "research"', "".join(chunks))
        self.assertIn('"agent_status": "running"', "".join(chunks))
        self.assertEqual(sandbox_inputs, ["numbers\nRAG context"])
        self.assertEqual(AgentWorkerRun.objects.filter(run=run).count(), 3)
        self.assertEqual(AgentTaskRecord.objects.filter(run=run, worker__isnull=False).count(), 2)

    @patch("agent.applications.file_batch._complete_response_with_retries")
    def test_file_batch_builds_map_reduce_context_for_allowed_folder(self, mock_complete):
        class Config:
            multi_agent_enabled = False
            multi_agent_parallel_tools = False
            multi_agent_max_workers = 3

            def tool_enabled(self, name, default=False):
                return name == "file_batch"

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir).resolve()
            (root / "alpha.txt").write_text("Alpha project notes.\nDetails.", encoding="utf-8")
            (root / "beta.md").write_text("# Beta\nRelease checklist.", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_complete.return_value = json.dumps(
                [
                    {"path": str(root / "alpha.txt"), "summary": "Alphaのメモです。", "status": "ok"},
                    {"path": str(root / "beta.md"), "summary": "Betaのチェックリストです。", "status": "ok"},
                ],
                ensure_ascii=False,
            )

            runner = _build_file_batch_input(
                self.thread,
                "フォルダの中のファイル名と一言要約を表にしてください",
                Config(),
                "answer base",
            )
            result = None
            while True:
                try:
                    next(runner)
                except StopIteration as exc:
                    result = exc.value
                    break

        self.assertTrue(result.ok)
        self.assertIn("File batch map-reduce context", result.input_text)
        self.assertIn("Alphaのメモです。", result.input_text)
        self.assertIn("Betaのチェックリストです。", result.input_text)
        self.assertIn('"map_batches": 1', result.input_text)
        self.assertEqual(len(result.paths), 2)
        self.assertTrue(any(path.endswith("alpha.txt") for path in result.paths))
        self.assertTrue(any(path.endswith("beta.md") for path in result.paths))
        mock_complete.assert_called_once()

    def test_extract_attachment_paths_reads_known_prefixes(self):
        attachments = [
            "File: /allowed/root/a.txt\n```text\ncontent\n```",
            "Folder: /allowed/root\n[file] a.txt",
            "Auto-selected file: /allowed/root/b.csv\n```text\ncontent\n```",
            "Unrelated block with no recognized prefix",
        ]

        paths = _extract_attachment_paths(attachments)

        self.assertEqual(paths, ("/allowed/root/a.txt", "/allowed/root", "/allowed/root/b.csv"))

    def test_format_file_list_for_display_truncates_with_remainder(self):
        paths = tuple(f"/allowed/root/file{i}.txt" for i in range(7))

        display = _format_file_list_for_display(paths, limit=5)

        self.assertEqual(display, "file0.txt, file1.txt, file2.txt, file3.txt, file4.txt, +2 more")
        self.assertEqual(_format_file_list_for_display(()), "")

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_saves_complete_assistant_message(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "hel"), ("delta", "lo"), ("response_id", "resp_1")])

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "hello"})
        self.assertEqual(create.status_code, 200)
        payload = create.json()

        stream = self.client.get(payload["stream_url"])
        body = b"".join(stream.streaming_content).decode()

        self.assertIn('"progress_tail"', body)
        self.assertIn("Goal: Answer the user's request: hello", body)
        self.assertIn("Criteria:", body)
        self.assertIn("Plan: Direct answer; no tool use needed.", body)
        self.assertIn('"done": true', body)
        self.assertIn('"elapsed_ms":', body)
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.content, "hello")
        self.assertEqual(assistant.status, "complete")
        self.assertEqual(assistant.openai_response_id, "resp_1")
        run = AgentRun.objects.get(assistant_message=assistant)
        self.assertEqual(run.status, "complete")
        self.assertEqual(run.user_message_id, payload["user_id"])
        self.assertEqual(run.goal, "Answer the user's request: hello")
        self.assertEqual(run.current_plan_queue, [])
        self.assertEqual(run.final_message, "hello")

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_plain_save_request_writes_final_answer_to_output_folder(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "保存する本文")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "この内容を保存してください"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

            saved_path = root / "maigent-output.txt"
            self.assertEqual(saved_path.read_text(encoding="utf-8"), "保存する本文\n")

        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("回答の保存結果", assistant.content)
        self.assertIn("OK:", assistant.content)

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_plain_save_request_reports_missing_output_folder(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "保存する本文")])

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "この内容を保存してください"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        b"".join(stream.streaming_content)

        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("回答の保存結果", assistant.content)
        self.assertIn("NG:", assistant.content)
        self.assertIn("書き出し先フォルダが未設定です", assistant.content)

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_includes_full_thread_conversation_history(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        Message.objects.create(thread=self.thread, role="user", content="最初の条件はAです")
        Message.objects.create(thread=self.thread, role="assistant", content="Aを覚えました")

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "それを使って答えて"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("Thread conversation history:", input_text)
        self.assertIn("user:\n最初の条件はAです", input_text)
        self.assertIn("assistant:\nAを覚えました", input_text)
        self.assertIn("Latest user message:\nそれを使って答えて", input_text)
        self.assertEqual(input_text.count("それを使って答えて"), 1)
        self.assertNotIn("pending", input_text)

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_initial_clarifier_returns_questions_before_planning(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = False
            initial_clarifier = {"max_output_tokens": 192, "reasoning_effort": "none", "llm_max_retries": 1}

            def tool_enabled(self, name, default=False):
                return name == "initial_clarifier"

        mock_config.return_value = Config()
        mock_complete.return_value = json.dumps(
            {
                "needs_clarification": True,
                "reason": "対象が不明なため確認が必要です。",
                "questions": ["どのファイルを対象にしますか？", "出力形式は何にしますか？", "期限はありますか？", "余分な質問"],
            },
            ensure_ascii=False,
        )

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "いい感じにまとめて"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        body = b"".join(stream.streaming_content).decode()

        self.assertNotIn("Goal:", body)
        mock_stream.assert_not_called()
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.status, "complete")
        self.assertIn("対象が不明なため確認が必要です。", assistant.content)
        self.assertIn("1. どのファイルを対象にしますか？", assistant.content)
        self.assertIn("3. 期限はありますか？", assistant.content)
        self.assertIn("どのファイルを対象にしますか？", assistant.content)
        self.assertNotIn("余分な質問", assistant.content)
        self.assertEqual(AgentRun.objects.filter(assistant_message=assistant).count(), 0)
        self.assertEqual(mock_complete.call_args.kwargs["max_output_tokens"], 192)
        self.assertEqual(mock_complete.call_args.kwargs["reasoning_effort"], "none")

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_initial_clarifier_allows_planning_when_no_clarification_needed(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = False
            initial_clarifier = {"max_output_tokens": 192, "reasoning_effort": "none", "llm_max_retries": 1}

            def tool_enabled(self, name, default=False):
                return name == "initial_clarifier"

        mock_config.return_value = Config()
        mock_complete.return_value = '{"needs_clarification": false, "reason": "clear", "questions": []}'
        mock_stream.return_value = iter([("delta", "planned answer")])

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "この件について教えてください。"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        body = b"".join(stream.streaming_content).decode()

        self.assertIn("Plan: Direct answer; no tool use needed.", body)
        self.assertIn("planned answer", body)
        mock_stream.assert_called_once()
        self.assertEqual(mock_complete.call_count, 1)

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_initial_clarifier_falls_back_on_invalid_or_empty_question_response(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = False
            initial_clarifier = {"llm_max_retries": 0}

            def tool_enabled(self, name, default=False):
                return name == "initial_clarifier"

        mock_config.return_value = Config()
        mock_complete.return_value = '{"needs_clarification": true, "reason": "missing"}'
        mock_stream.return_value = iter([("delta", "fallback answer")])

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "この件について教えてください。"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        body = b"".join(stream.streaming_content).decode()

        self.assertIn("fallback answer", body)
        mock_stream.assert_called_once()

    @patch("agent.applications.llm_helpers.complete_response")
    def test_initial_clarifier_skips_obvious_actionable_tool_plan(self, mock_complete):
        class Config:
            initial_clarifier = {"enabled": True}

            def tool_enabled(self, name, default=False):
                return name in {"initial_clarifier", "rag", "sandbox"}

        decision = _request_initial_clarification_if_needed(
            Config(),
            "Thread conversation history:\n\nLatest user message:\ntest.csvの点を科目ごとに計算してください。",
            latest_user_text="test.csvの点を科目ごとに計算してください。",
        )

        self.assertIsNone(decision)
        mock_complete.assert_not_called()

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_precheck_runs_clarifier_and_tool_selector_together(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = False

            def tool_enabled(self, name, default=False):
                return name in {"initial_clarifier", "tool_selector"}

        def fake_complete(config, prompt, instructions, **kwargs):
            if "ツールを選択" in instructions:
                return '{"steps": [], "reason": "no tool needed"}'
            return '{"needs_clarification": false, "reason": "clear", "questions": []}'

        mock_config.return_value = Config()
        mock_complete.side_effect = fake_complete
        mock_stream.return_value = iter([("delta", "hi there")])

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "hello"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        body = b"".join(stream.streaming_content).decode()

        self.assertIn("hi there", body)
        self.assertEqual(mock_complete.call_count, 2)
        mock_stream.assert_called_once()

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_tool_selector_result_overrides_clarifier_when_actionable(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = False

            def tool_enabled(self, name, default=False):
                return name in {"initial_clarifier", "tool_selector", "rag"}

        def fake_complete(config, prompt, instructions, **kwargs):
            if "ツールを選択" in instructions:
                return '{"steps": [{"tool": "rag", "purpose": "look up local context"}], "rag_query": "alpha", "reason": "named entity may be local"}'
            return '{"needs_clarification": true, "reason": "ambiguous request", "questions": ["どのファイルですか？"]}'

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "alpha.txt").write_text("alpha is a sealed book in the deep sea library.", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_config.return_value = Config()
            mock_complete.side_effect = fake_complete
            mock_stream.return_value = iter([("delta", "alpha is a sealed book.")])

            # This message has no rag/folder/file keyword, so the cheap rule-based
            # planner alone would not know a tool is available and would let the
            # clarifier's "needs_clarification" stand. The (mocked) tool-selector
            # LLM finds a concrete actionable plan instead, which should win.
            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "灰島ノエについて教えて"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            body = b"".join(stream.streaming_content).decode()

        self.assertNotIn("どのファイルですか", body)
        self.assertIn("alpha is a sealed book.", body)
        mock_stream.assert_called_once()
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.content, "alpha is a sealed book.")

    @patch("agent.views.search_web")
    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_web_search_still_runs_after_rag_finds_no_local_context(self, mock_config, mock_stream, mock_complete, mock_search_web):
        from .applications.web_search import WebSearchItem, WebSearchResult

        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = False

            def tool_enabled(self, name, default=False):
                return name in {"tool_selector", "rag", "web_search"}

        def fake_complete(config, prompt, instructions, **kwargs):
            return (
                '{"steps": [{"tool": "rag", "purpose": "check local files first"}, '
                '{"tool": "web_search", "purpose": "search the web as explicitly requested"}], "reason": "user explicitly asked for web"}'
            )

        mock_config.return_value = Config()
        mock_complete.side_effect = fake_complete
        mock_search_web.return_value = WebSearchResult(
            ok=True,
            query="今治西高校 校歌",
            results=(WebSearchItem(title="今治西高校の校歌", url="https://example.com", snippet="..."),),
        )
        mock_stream.return_value = iter([("delta", "今治西高校の校歌についての情報です。")])

        # An access path exists (so rag is offered to tool_selector at all) but
        # its only file is unrelated, so the real rag search finds no adequate
        # context. Before the fix, rag's "no context" result would end the
        # whole plan here and web_search would never run.
        with TemporaryDirectory() as root_dir:
            (Path(root_dir) / "unrelated.txt").write_text("quarterly budget notes", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=root_dir, mode="read")

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]), {"message": "今治西校の校歌についてwebを調べてみてよ"}
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        mock_search_web.assert_called_once()
        mock_stream.assert_called_once()
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.content, "今治西高校の校歌についての情報です。")
        self.assertNotIn("見つかりませんでした", assistant.content)
        self.assertEqual(AgentRun.objects.filter(assistant_message=assistant).count(), 1)

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_final_evaluation_skip_when_no_tools_used(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = True
            final_evaluation_max_retries = 2

            def tool_enabled(self, name, default=False):
                return False

        mock_config.return_value = Config()
        mock_stream.return_value = iter([("delta", "direct answer")])

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "日本の首都は？"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        body = b"".join(stream.streaming_content).decode()

        self.assertIn("no tools were used; skipping final evaluation", body)
        self.assertIn("direct answer", body)
        mock_complete.assert_not_called()
        mock_stream.assert_called_once()
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.content, "direct answer")

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_final_evaluation_retries_from_plan_when_inadequate(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = True
            final_evaluation_max_retries = 1
            final_evaluation_max_output_tokens = 80
            final_evaluation_reasoning_effort = "minimal"
            final_evaluation = {"llm_max_retries": 1}

            def tool_enabled(self, name, default=False):
                return name == "rag"

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "alpha.txt").write_text("alpha is a sealed book in the deep sea library.", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_config.return_value = Config()
            mock_stream.side_effect = [
                iter([("delta", "bad answer"), ("response_id", "resp_1")]),
                iter([("delta", "good answer"), ("response_id", "resp_2")]),
            ]
            mock_complete.side_effect = [
                None,
                '{"adequate": false, "reason": "too vague"}',
                '{"adequate": true, "reason": "answers the question"}',
            ]

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "alphaについて要約して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            body = b"".join(stream.streaming_content).decode()

        self.assertIn("good answer", body)
        self.assertIn("final evaluation failed; replanning with a different plan", body)
        self.assertIn("Revised local file context after failed final evaluation", body)
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.content, "good answer")
        self.assertEqual(assistant.openai_response_id, "resp_2")
        self.assertEqual(mock_stream.call_count, 2)
        self.assertEqual(mock_complete.call_count, 3)
        first_evaluation_prompt = mock_complete.call_args_list[1].args[1]
        self.assertIn("Goal set before planning:", first_evaluation_prompt)
        self.assertIn("Evaluation criteria set before planning:", first_evaluation_prompt)
        self.assertIn("ユーザーの発言に直接対応", first_evaluation_prompt)
        self.assertEqual(mock_complete.call_args_list[1].kwargs["max_output_tokens"], 80)
        self.assertEqual(mock_complete.call_args_list[1].kwargs["reasoning_effort"], "minimal")

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_final_evaluation_receives_rag_context_when_rag_was_used(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = True
            final_evaluation_max_retries = 1
            final_evaluation_max_output_tokens = 80
            final_evaluation_reasoning_effort = "minimal"
            final_evaluation = {"llm_max_retries": 0}

            def tool_enabled(self, name, default=False):
                return name == "rag"

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "alpha.txt").write_text("alpha is a sealed book in the deep sea library.", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_config.return_value = Config()
            mock_stream.return_value = iter([("delta", "alpha is a sealed book.")])
            mock_complete.return_value = '{"adequate": true, "reason": "uses retrieved context"}'

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "alphaについて要約して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        evaluation_prompt = mock_complete.call_args.args[1]
        self.assertIn("User question and available context:", evaluation_prompt)
        self.assertIn("RAG context from allowed local files:", evaluation_prompt)
        self.assertIn("alpha.txt", evaluation_prompt)
        self.assertIn("alpha is a sealed book", evaluation_prompt)
        self.assertIn("Candidate answer:", evaluation_prompt)

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_final_evaluation_empty_response_keeps_current_answer(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = True
            final_evaluation_max_retries = 3
            final_evaluation_max_output_tokens = 80
            final_evaluation_reasoning_effort = "minimal"
            final_evaluation = {"llm_max_retries": 1}

            def tool_enabled(self, name, default=False):
                return name == "rag"

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "alpha.txt").write_text("alpha is a sealed book in the deep sea library.", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_config.return_value = Config()
            mock_stream.return_value = iter([("delta", "current answer"), ("response_id", "resp_1")])
            mock_complete.side_effect = [None, ""]

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "alphaについて要約して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            body = b"".join(stream.streaming_content).decode()

        self.assertIn("current answer", body)
        self.assertIn("final evaluation could not be completed; keeping current answer", body)
        self.assertNotIn("Revised local file context after failed final evaluation", body)
        self.assertEqual(mock_stream.call_count, 1)
        self.assertEqual(mock_complete.call_count, 2)
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.content, "current answer")

    @patch("agent.applications.llm_helpers.complete_response")
    def test_final_evaluation_does_not_pass_incomplete_json(self, mock_complete):
        class Config:
            final_evaluation_max_output_tokens = 80
            final_evaluation_reasoning_effort = "none"
            final_evaluation = {"llm_max_retries": 0}

        mock_complete.return_value = '{"adequate": true, "reason": "ユーザーの質問に対して、test'

        result = _evaluate_final_answer(
            Config(),
            "User question",
            "Answer the user's request.",
            ["The answer directly addresses the request."],
            "Candidate answer",
        )

        self.assertFalse(result["adequate"])
        self.assertTrue(result["evaluation_failed"])
        self.assertIn("JSON", result["reason"])

    def test_failed_final_evaluation_uses_different_plan(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name == "rag" or default

        plan = AgentPlan(
            goal="Answer the user's request: hello",
            evaluation_criteria=["The answer directly addresses the user's request."],
            summary="Direct answer; no tool use needed.",
            steps=[AgentPlanStep("final", "Answer directly.")],
        )
        revised = _avoid_failed_plan(plan, Config(), [plan.summary])

        self.assertNotEqual(revised.summary, plan.summary)
        self.assertEqual(revised.goal, plan.goal)
        self.assertEqual(revised.evaluation_criteria, plan.evaluation_criteria)
        self.assertEqual([step.tool for step in revised.steps], ["rag", "final"])

    def test_failed_sandbox_plan_retries_without_sandbox(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return True

        plan = AgentPlan(
            goal="Answer the user's request: calculate from files",
            evaluation_criteria=["The answer includes the computed result."],
            summary="rag -> sandbox -> final",
            steps=[
                AgentPlanStep("rag", "Search context."),
                AgentPlanStep("sandbox", "Run calculation."),
            ],
        )
        revised = _avoid_failed_plan(plan, Config(), [plan.summary])

        self.assertNotEqual(revised.summary, plan.summary)
        self.assertEqual(revised.goal, plan.goal)
        self.assertEqual(revised.evaluation_criteria, plan.evaluation_criteria)
        self.assertEqual([step.tool for step in revised.steps], ["rag"])

    def test_failed_local_file_plan_keeps_local_context_in_final_fallback(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name in {"rag", "file_batch"} or default

        plan = AgentPlan(
            goal="Answer the user's request: merge files",
            evaluation_criteria=["The answer uses allowed local files."],
            summary="rag -> file_batch -> sandbox -> final",
            steps=[
                AgentPlanStep("rag", "Search context."),
                AgentPlanStep("file_batch", "Process files."),
                AgentPlanStep("sandbox", "Create artifact."),
            ],
        )
        failed = [
            plan.summary,
            "rag -> file_batch -> final (without sandbox)",
            "rag -> file_batch -> sandbox -> final (with added rag)",
        ]

        revised = _avoid_failed_plan(plan, Config(), failed)

        self.assertNotEqual(revised.summary, plan.summary)
        self.assertEqual(revised.goal, plan.goal)
        self.assertEqual(revised.evaluation_criteria, plan.evaluation_criteria)
        self.assertEqual([step.tool for step in revised.steps], ["file_batch", "final"])
        self.assertIn("local file context", revised.summary)

    @patch("agent.views.run_sandbox")
    @patch("agent.views.load_runtime_config")
    def test_streaming_uses_sandbox_tool_when_selected(self, mock_config, mock_run_sandbox):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        mock_run_sandbox.return_value = SandboxResult(True, "4")

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "2 + 2を正確に計算して"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        b"".join(stream.streaming_content)

        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("Sandbox実行結果: 成功", assistant.content)
        self.assertIn("4", assistant.content)
        mock_run_sandbox.assert_called_once()
        run = AgentRun.objects.get(assistant_message=assistant)
        self.assertEqual(run.status, "complete")
        self.assertEqual(run.final_message, assistant.content)
        task = AgentTaskRecord.objects.get(run=run)
        self.assertEqual(task.sequence, 1)
        self.assertEqual(task.tool, "sandbox")
        self.assertEqual(task.status, "ok")
        self.assertIn("4", task.result)

    @patch("agent.views.run_sandbox")
    @patch("agent.views.load_runtime_config")
    def test_sandbox_artifact_is_saved_by_host_broker(self, mock_config, mock_run_sandbox):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            output_path = root / "result.txt"
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])
            mock_run_sandbox.return_value = SandboxResult(True, "4")

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"2 + 2を正確に計算して、結果を{output_path}に保存してください"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "4\n")
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("Sandbox成果物の保存結果", assistant.content)
        self.assertIn("OK:", assistant.content)

    @patch("agent.views.run_sandbox")
    @patch("agent.views.load_runtime_config")
    def test_sandbox_artifact_json_is_saved_by_host_broker(self, mock_config, mock_run_sandbox):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            output_path = root / "artifact.txt"
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])
            mock_run_sandbox.return_value = SandboxResult(
                True,
                "\n".join(
                    [
                        "artifact ready",
                        "```json",
                        json.dumps({"maigent_artifacts": [{"path": str(output_path), "content": "artifact\n", "append": False}]}),
                        "```",
                    ]
                ),
            )

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"Pythonで成果物を作成して{output_path}に保存してください"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "artifact\n")

    @patch("agent.views.run_sandbox")
    @patch("agent.applications.sandbox.generate_sandbox_code")
    @patch("agent.views.load_runtime_config")
    def test_sandbox_image_artifact_base64_is_saved_and_linked(self, mock_config, mock_generate_sandbox_code, mock_run_sandbox):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        mock_generate_sandbox_code.return_value = "print('chart ready')"
        png_bytes = b"\x89PNG\r\n\x1a\nimage-bytes"
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])
            mock_run_sandbox.return_value = SandboxResult(
                True,
                "\n".join(
                    [
                        "chart ready",
                        "```json",
                        json.dumps(
                            {
                                "maigent_artifacts": [
                                    {
                                        "path": "chart.png",
                                        "content_base64": base64.b64encode(png_bytes).decode("ascii"),
                                        "mime_type": "image/png",
                                        "append": False,
                                    }
                                ]
                            }
                        ),
                        "```",
                    ]
                ),
            )

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": "Pythonで画像を作成して保存してください"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

            self.assertEqual((root / "chart.png").read_bytes(), png_bytes)
            assistant = Message.objects.get(id=payload["assistant_id"])
            self.assertIn(f"![chart.png](/projects/{self.project.id}/artifacts/chart.png)", assistant.content)

            image_response = self.client.get(reverse("artifact_image", args=[self.project.id, "chart.png"]))
            self.assertEqual(image_response.status_code, 200)
            self.assertEqual(image_response["Content-Type"], "image/png")

    @patch("agent.views.run_sandbox")
    @patch("agent.applications.sandbox.generate_sandbox_code")
    @patch("agent.views.load_runtime_config")
    def test_typed_sandbox_image_artifact_is_saved_without_raw_json_in_message(
        self,
        mock_config,
        mock_generate_sandbox_code,
        mock_run_sandbox,
    ):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        mock_generate_sandbox_code.return_value = "print('typed result')"
        png_bytes = b"\x89PNG\r\n\x1a\ntyped-image"
        artifact = {
            "path": "typed-chart.png",
            "content_base64": base64.b64encode(png_bytes).decode("ascii"),
            "mime_type": "image/png",
            "append": False,
        }
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])
            mock_run_sandbox.return_value = SandboxResult(True, "chart ready", (artifact,), json.dumps({"ignored": True}))

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": "Pythonで画像を作成して保存してください"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

            self.assertEqual((root / "typed-chart.png").read_bytes(), png_bytes)

        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("chart ready", assistant.content)
        self.assertIn(f"![typed-chart.png](/projects/{self.project.id}/artifacts/typed-chart.png)", assistant.content)
        self.assertNotIn("content_base64", assistant.content)
        self.assertNotIn("maigent_artifacts", assistant.content)

    def test_typed_sandbox_output_parser_separates_stdout_and_artifacts(self):
        payload = {
            "maigent_sandbox_result": {
                "stdout": "chart ready",
                "artifacts": [
                    {
                        "path": "chart.png",
                        "content_base64": base64.b64encode(b"png").decode("ascii"),
                        "mime_type": "image/png",
                    }
                ],
            }
        }

        stdout, artifacts = _parse_typed_sandbox_output(json.dumps(payload))

        self.assertEqual(stdout, "chart ready")
        self.assertEqual(artifacts[0]["path"], "chart.png")

    def test_typed_sandbox_output_parser_accepts_top_level_stdout_artifacts(self):
        payload = {
            "stdout": "分析結果\n平均値: 79.50点",
            "artifacts": [],
        }

        stdout, artifacts = _parse_typed_sandbox_output(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(stdout, "分析結果\n平均値: 79.50点")
        self.assertEqual(artifacts, [])

    def test_typed_sandbox_output_parser_extracts_embedded_payload(self):
        payload = {
            "maigent_sandbox_result": {
                "stdout": "chart ready",
                "artifacts": [
                    {
                        "path": "chart.png",
                        "content_base64": base64.b64encode(b"png").decode("ascii"),
                        "mime_type": "image/png",
                    }
                ],
            }
        }

        stdout, artifacts = _parse_typed_sandbox_output("合計人数: 30\n" + json.dumps(payload))

        self.assertEqual(stdout, "合計人数: 30")
        self.assertEqual(artifacts[0]["path"], "chart.png")

    def test_typed_sandbox_output_parser_extracts_python_dict_payload(self):
        payload = {
            "maigent_sandbox_result": {
                "stdout": "chart ready",
                "artifacts": [
                    {
                        "path": "chart.png",
                        "content_base64": base64.b64encode(b"png").decode("ascii"),
                        "mime_type": "image/png",
                    }
                ],
            }
        }

        stdout, artifacts = _parse_typed_sandbox_output(f"合計人数: 30\n{payload}")

        self.assertEqual(stdout, "合計人数: 30")
        self.assertEqual(artifacts[0]["path"], "chart.png")
        self.assertEqual(artifacts[0]["content_base64"], base64.b64encode(b"png").decode("ascii"))

    @patch("agent.views.run_sandbox")
    @patch("agent.applications.sandbox.generate_sandbox_code")
    @patch("agent.views.load_runtime_config")
    def test_sandbox_image_artifact_link_handles_spaces(self, mock_config, mock_generate_sandbox_code, mock_run_sandbox):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        mock_generate_sandbox_code.return_value = "print('chart ready')"
        png_bytes = b"\x89PNG\r\n\x1a\nimage-bytes"
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])
            mock_run_sandbox.return_value = SandboxResult(
                True,
                "\n".join(
                    [
                        "chart ready",
                        "```json",
                        json.dumps(
                            {
                                "maigent_artifacts": [
                                    {
                                        "path": "my chart.png",
                                        "content_base64": base64.b64encode(png_bytes).decode("ascii"),
                                        "mime_type": "image/png",
                                        "append": False,
                                    }
                                ]
                            }
                        ),
                        "```",
                    ]
                ),
            )

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": "Pythonで画像成果物を作成してmy chart.pngに保存してください\n```python\nprint('chart ready')\n```"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

            assistant = Message.objects.get(id=payload["assistant_id"])
            self.assertIn(f"![my chart.png](/projects/{self.project.id}/artifacts/my%20chart.png)", assistant.content)
            image_response = self.client.get(reverse("artifact_image", args=[self.project.id, "my chart.png"]))
            self.assertEqual(image_response.status_code, 200)
            self.assertEqual(image_response["Cache-Control"], "no-store")

    def test_artifact_image_route_rejects_non_images_and_output_escape(self):
        with TemporaryDirectory() as root_dir, TemporaryDirectory() as other_dir:
            root = Path(root_dir)
            other = Path(other_dir)
            self.project.output_path = str(root)
            self.project.save(update_fields=["output_path"])
            (root / "note.txt").write_text("not an image", encoding="utf-8")
            (other / "secret.png").write_bytes(b"secret")

            non_image = self.client.get(reverse("artifact_image", args=[self.project.id, "note.txt"]))
            escaped = self.client.get(reverse("artifact_image", args=[self.project.id, f"../{other.name}/secret.png"]))

            self.assertEqual(non_image.status_code, 404)
            self.assertEqual(escaped.status_code, 404)

    @patch("agent.views.run_sandbox")
    @patch("agent.views.load_runtime_config")
    def test_sandbox_artifact_json_rejects_path_outside_output_folder(self, mock_config, mock_run_sandbox):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        with TemporaryDirectory() as output_dir, TemporaryDirectory() as other_dir:
            self.project.output_path = output_dir
            self.project.save(update_fields=["output_path"])
            outside_path = Path(other_dir) / "artifact.txt"
            mock_run_sandbox.return_value = SandboxResult(
                True,
                json.dumps({"maigent_artifacts": [{"path": str(outside_path), "content": "artifact\n", "append": False}]}),
            )

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": f"Pythonで成果物を作成して{outside_path}に保存してください"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

            self.assertFalse(outside_path.exists())
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("Sandbox成果物の保存結果", assistant.content)
        self.assertIn("NG:", assistant.content)

    def test_implicit_sandbox_artifact_ignores_rag_source_paths(self):
        with TemporaryDirectory() as output_dir, TemporaryDirectory() as source_dir:
            output = Path(output_dir)
            source = Path(source_dir)
            source_path = source / "test.csv"
            source_path.write_text("score\n1\n", encoding="utf-8")
            self.project.output_path = str(output)
            self.project.save(update_fields=["output_path"])
            input_text = "\n".join(
                [
                    "test.csvのテストの点のヒストグラムを作り、pngファイルを読み書き可能フォルダに保存してください。",
                    "",
                    f"Auto-selected file: {source_path}",
                    "```text",
                    "score",
                    "1",
                    "```",
                ]
            )

            message = _persist_sandbox_artifacts(self.thread, input_text, "histogram text")

            self.assertIn("OK:", message)
            self.assertTrue((output / "maigent-histogram.txt").exists())
            self.assertEqual((output / "maigent-histogram.txt").read_text(encoding="utf-8"), "histogram text\n")
            self.assertEqual(source_path.read_text(encoding="utf-8"), "score\n1\n")

    @patch("agent.tooling.subprocess.run")
    def test_sandbox_generates_program_for_natural_language_sum(self, mock_run):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "6\n"
        mock_run.return_value.stderr = ""

        result = run_sandbox("1, 2, 3 の合計を正確に計算して", Config())

        self.assertTrue(result.ok)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[-1], "python /work/script.py")

    @patch("agent.tooling.subprocess.run")
    def test_sandbox_does_not_install_libraries_on_each_run_by_default(self, mock_run):
        class Config:
            sandbox_allowed_libraries = ["numpy", "pandas"]
            sandbox_install_libraries_on_run = False
            sandbox_image = "maigent-sandbox:py311"
            sandbox_timeout_seconds = 20

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "6\n"
        mock_run.return_value.stderr = ""

        result = run_sandbox("1, 2, 3 の合計を正確に計算して", Config())

        self.assertTrue(result.ok)
        command = mock_run.call_args.args[0]
        self.assertIn("--network", command)
        self.assertNotIn("pip install", command[-1])

    @patch("agent.tooling.subprocess.run")
    def test_sandbox_applies_configured_resource_limits(self, mock_run):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20
            sandbox_memory_limit_mb = 256
            sandbox_pids_limit = 64
            sandbox_cpus = 0.5

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "6\n"
        mock_run.return_value.stderr = ""

        result = run_sandbox("1, 2, 3 の合計を正確に計算して", Config())

        self.assertTrue(result.ok)
        command = mock_run.call_args.args[0]
        self.assertIn("--memory", command)
        self.assertEqual(command[command.index("--memory") + 1], "256m")
        self.assertIn("--pids-limit", command)
        self.assertEqual(command[command.index("--pids-limit") + 1], "64")
        self.assertIn("--cpus", command)
        self.assertEqual(command[command.index("--cpus") + 1], "0.5")

    @patch("agent.tooling.subprocess.run")
    def test_sandbox_defaults_resource_limits_when_config_omits_them(self, mock_run):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "6\n"
        mock_run.return_value.stderr = ""

        result = run_sandbox("1, 2, 3 の合計を正確に計算して", Config())

        self.assertTrue(result.ok)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[command.index("--memory") + 1], "512m")
        self.assertEqual(command[command.index("--pids-limit") + 1], "128")
        self.assertEqual(command[command.index("--cpus") + 1], "1.0")

    @patch("agent.tooling.subprocess.run")
    def test_sandbox_generates_program_from_rag_numeric_context(self, mock_run):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "20\n"
        mock_run.return_value.stderr = ""
        message = "\n".join(
            [
                "Agent plan: rag -> sandbox -> final",
                "RAG context from allowed local files:",
                "--- numbers.txt ---",
                "5",
                "15",
                "",
                "numbersファイルの平均を正確に計算して",
            ]
        )

        result = run_sandbox(message, Config())

        self.assertTrue(result.ok)
        self.assertEqual(result.output, "20")
        mock_run.assert_called_once()

    def test_sandbox_generates_csv_program_for_score_column(self):
        message = "\n".join(
            [
                "test.csvファイルの内容を確認して、テストの点の平均点と標準偏差を実際にプログラムを実行して回答してください。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "id,name,テストの点",
                "1001,Alice,70",
                "1002,Bob,80",
                "1003,Carol,90",
                "```",
            ]
        )

        code = _python_for_tabular_task(message)
        namespace = {"__name__": "__main__"}
        with patch("builtins.print") as mock_print:
            exec(code, namespace)

        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("column: テストの点", printed)
        self.assertIn("mean: 80", printed)
        self.assertIn("population_std:", printed)
        self.assertNotIn("column: id", printed)

    def test_sandbox_tabular_program_uses_host_dataset_when_available(self):
        message = "\n".join(
            [
                "test.csvの全てのテストの点を足し合わせてください。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "名前,テストの種類,テストの点",
                "佐藤太郎,国語,82",
                "```",
                "",
                "Host-provided sandbox dataset API:",
                "Available datasets:",
                "- id: rag_1; name: test.csv; kind: csv; rows: 3; columns: 名前, テストの種類, テストの点; truncated: no",
            ]
        )

        code = _python_for_tabular_task(message)

        self.assertIn("df = load_dataset('rag_1')", code)
        self.assertNotIn("csv_text =", code)

    def test_sandbox_csv_histogram_image_requires_llm_generated_code(self):
        message = "\n".join(
            [
                "test.csvのテストの点のヒストグラムを作り画像を表示してください。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "id,name,テストの点",
                "1001,Alice,70",
                "1002,Bob,80",
                "1003,Carol,90",
                "```",
            ]
        )

        self.assertTrue(requires_llm_sandbox_program(message))
        self.assertFalse(can_build_sandbox_program(message))

    def test_sandbox_code_policy_rejects_local_file_reads_and_path_writes(self):
        self.assertIn("local file reads", _sandbox_code_policy_violation("import pandas as pd\npd.read_csv('test.csv')"))
        self.assertIn(
            "local file reads",
            _sandbox_code_policy_violation("import pandas as pd\nfile_path = 'test.csv'\ndf = pd.read_csv(file_path)"),
        )
        self.assertIn(
            "local filesystem checks",
            _sandbox_code_policy_violation("import os\nfile_path = 'test.csv'\nos.path.exists(file_path)"),
        )
        self.assertIn("local file access", _sandbox_code_policy_violation("from pathlib import Path\nPath('test.csv').read_text()"))
        self.assertIn("local file access", _sandbox_code_policy_violation("open('test.csv')"))
        self.assertIn(
            "sandbox file writes",
            _sandbox_code_policy_violation("import matplotlib.pyplot as plt\noutput_path = 'histogram.png'\nplt.savefig(output_path)"),
        )
        self.assertIn(
            "legacy artifact payload format",
            _sandbox_code_policy_violation("print({'maigent_artifacts': []})"),
        )
        self.assertIn(
            "embedded tabular data copies",
            _sandbox_code_policy_violation(
                "csv_data = '''名前，テストの種類，テストの点\n佐藤太郎，国語，82\n佐藤太郎，数学，76'''\n"
                "df = pd.read_csv(io.StringIO(csv_data))"
            ),
        )
        self.assertEqual(
            _sandbox_code_policy_violation("import io, pandas as pd\ncsv_text='a\\n1\\n'\ndf = pd.read_csv(io.StringIO(csv_text))"),
            "",
        )
        self.assertEqual(
            _sandbox_code_policy_violation("df = load_dataset('rag_1')\nprint(df['テストの点'].sum())"),
            "",
        )
        self.assertEqual(
            _sandbox_code_policy_violation("import io, matplotlib.pyplot as plt\nbuf = io.BytesIO()\nplt.savefig(buf, format='png')"),
            "",
        )

    def test_sandbox_prepends_host_dataset_loader(self):
        dataset = make_sandbox_dataset(
            "rag_1",
            "test.csv",
            "/allowed/test.csv",
            "csv",
            "名前,テストの種類,テストの点\n佐藤太郎,国語,82\n",
        )
        script = _prepend_sandbox_datasets("df = load_dataset('rag_1')\nprint(df['テストの点'].sum())", (dataset,))

        self.assertIn("def load_dataset(dataset_id):", script)
        self.assertIn("名前,テストの種類,テストの点", script)

    def test_sandbox_runtime_prelude_configures_japanese_matplotlib_font(self):
        script = _prepend_sandbox_runtime_prelude("print('ok')\n")

        self.assertIn("Noto Sans CJK JP", script)
        self.assertIn("matplotlib.rcParams['font.family']", script)
        self.assertIn("matplotlib.rcParams['axes.unicode_minus'] = False", script)

    @patch("agent.applications.sandbox.generate_sandbox_code")
    def test_sandbox_code_generation_retries_after_policy_rejection(self, mock_generate_sandbox_code):
        config = RuntimeConfig(values={"sandbox_code_generation": {"llm_max_retries": 1}}, sources=[])

        mock_generate_sandbox_code.side_effect = [
            "import pandas as pd\nfile_path = 'test.csv'\ndf = pd.read_csv(file_path)",
            "import io, pandas as pd\ncsv_text='score\\n80\\n'\ndf = pd.read_csv(io.StringIO(csv_text))\nprint(df['score'].sum())",
        ]

        code = _generate_sandbox_code_with_retries(config, "test.csvを集計してください")

        self.assertIn("io.StringIO", code)
        self.assertEqual(mock_generate_sandbox_code.call_count, 2)
        self.assertIn("Previous generated code was rejected", mock_generate_sandbox_code.call_args_list[1].args[1])

    @patch("agent.applications.sandbox.generate_sandbox_code")
    def test_disabled_sandbox_code_generation_skips_llm_call(self, mock_generate_sandbox_code):
        config = RuntimeConfig(values={"sandbox_code_generation": {"enabled": False}}, sources=[])

        code = _generate_sandbox_code_with_retries(config, "Pythonコードを生成してください")

        self.assertEqual(code, "")
        mock_generate_sandbox_code.assert_not_called()

    def test_sandbox_artifact_payload_is_hidden_from_display_output(self):
        payload = json.dumps(
            {
                "maigent_artifacts": [
                    {
                        "path": "chart.png",
                        "content_base64": base64.b64encode(b"png").decode("ascii"),
                        "mime_type": "image/png",
                    }
                ]
            }
        )

        output = _strip_sandbox_artifact_payloads(f"chart ready\n```json\n{payload}\n```\ndone")

        self.assertEqual(output, "chart ready\n\ndone")
        self.assertNotIn("content_base64", output)

    def test_marked_sandbox_artifact_payload_is_hidden_from_display_output(self):
        payload = json.dumps(
            {
                "maigent_artifacts": [
                    {
                        "path": "chart.png",
                        "content_base64": base64.b64encode(b"png").decode("ascii"),
                        "mime_type": "image/png",
                    }
                ]
            },
            indent=2,
        )

        output = _strip_sandbox_artifact_payloads(f"chart ready\n<MAIGENT_ARTIFACT>\n{payload}\ndone")

        self.assertEqual(output, "chart ready\n\ndone")
        self.assertNotIn("content_base64", output)
        self.assertNotIn("<MAIGENT_ARTIFACT>", output)

    def test_typed_sandbox_artifact_payload_is_hidden_from_display_output(self):
        payload = json.dumps(
            {
                "maigent_sandbox_result": {
                    "stdout": "chart ready",
                    "artifacts": [
                        {
                            "path": "chart.png",
                            "content_base64": base64.b64encode(b"png").decode("ascii"),
                            "mime_type": "image/png",
                        }
                    ],
                }
            }
        )

        output = _strip_sandbox_artifact_payloads(f"```json\n{payload}\n```")

        self.assertEqual(output, "")

    @patch("agent.tooling.subprocess.run")
    def test_sandbox_prefers_csv_program_over_first_number(self, mock_run):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "column: テストの点\nmean: 80\n"
        mock_run.return_value.stderr = ""
        message = "\n".join(
            [
                "test.csvファイルの内容を確認して、テストの点の平均点と標準偏差を実際にプログラムを実行して回答してください。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "id,name,テストの点",
                "1001,Alice,70",
                "1002,Bob,80",
                "1003,Carol,90",
                "```",
            ]
        )

        result = run_sandbox(message, Config())

        self.assertTrue(result.ok)
        command = mock_run.call_args.args[0]
        self.assertIn("python /work/script.py", command[-1])
        self.assertIn("column: テストの点", result.output)

    def test_sandbox_csv_program_outputs_sum_for_add_up_request(self):
        message = "\n".join(
            [
                "テストの点を全て足し合わせてください。ファイルはtest.csvです。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "名前,テストの種類,テストの点",
                "佐藤太郎,国語,82",
                "佐藤太郎,数学,76",
                "佐藤太郎,英語,88",
                "```",
            ]
        )

        code = _python_for_tabular_task(message)
        namespace = {"__name__": "__main__"}
        with patch("builtins.print") as mock_print:
            exec(code, namespace)

        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("column: テストの点", printed)
        self.assertIn("sum: 246", printed)

    def test_sandbox_csv_program_outputs_mean_and_variance(self):
        message = "\n".join(
            [
                "test.csvの全てのテストの点の平均と分散を求めてください。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "名前,テストの点",
                "A,70",
                "B,80",
                "C,90",
                "```",
            ]
        )

        code = _python_for_tabular_task(message)
        namespace = {"__name__": "__main__"}
        with patch("builtins.print") as mock_print:
            exec(code, namespace)

        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("column: テストの点", printed)
        self.assertIn("mean: 80", printed)
        self.assertIn("population_variance:", printed)
        self.assertIn("sample_variance:", printed)

    def test_sandbox_csv_histogram_text_requires_llm_generated_code(self):
        message = "\n".join(
            [
                "test.csvのテストの点のヒストグラムを作ってください。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "名前,テストの点",
                "A,70",
                "B,80",
                "C,90",
                "```",
            ]
        )

        self.assertTrue(requires_llm_sandbox_program(message))
        self.assertFalse(can_build_sandbox_program(message))

    def test_grouped_csv_task_requires_llm_generated_program(self):
        message = "\n".join(
            [
                "test.csvのテストの点を名前ごとに全ての科目を合計してください。",
                "",
                "RAG context from allowed local files:",
                "Auto-selected file: /tmp/test.csv",
                "```text",
                "名前,テストの種類,テストの点",
                "佐藤太郎,国語,82",
                "佐藤太郎,数学,76",
                "```",
            ]
        )

        self.assertTrue(requires_llm_sandbox_program(message))
        self.assertFalse(can_build_sandbox_program(message))

    @patch("agent.tooling.subprocess.run")
    def test_sandbox_executes_explicit_code_override_for_grouped_task(self, mock_run):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "grouped result\n"
        mock_run.return_value.stderr = ""
        message = "test.csvのテストの点を科目ごとの平均点と合計点を教えてください。"

        result = run_sandbox(message, Config(), code="print('grouped result')")

        self.assertTrue(result.ok)
        self.assertEqual(result.output, "grouped result")
        mock_run.assert_called_once()

    @patch("agent.views.stream_response")
    @patch("agent.views.run_sandbox")
    @patch("agent.views.load_runtime_config")
    def test_sandbox_no_code_falls_back_to_llm(self, mock_config, mock_run_sandbox, mock_stream):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_config.return_value = Config()
        mock_run_sandbox.return_value = SandboxResult(False, "sandboxで実行できるPythonコードまたは計算式を特定できませんでした。")
        mock_stream.return_value = iter([("delta", "通常回答")])

        create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "正確に説明して"})
        payload = create.json()
        stream = self.client.get(payload["stream_url"])
        b"".join(stream.streaming_content)

        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertEqual(assistant.content, "通常回答")
        mock_run_sandbox.assert_not_called()
        mock_stream.assert_called_once()

    def test_precise_explanation_does_not_select_sandbox_without_program(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        plan = build_agent_plan("正確に説明してください", Config())

        self.assertEqual(plan.goal, "Answer the user's request: 正確に説明してください")
        self.assertIn(
            "回答は、ユーザーの発言に直接対応している。具体的な作業依頼であれば要求に応え、雑談・近況報告・意思表示など会話的な発言であれば、タスクの成果物ではなく自然な受け答えで応じている。",
            plan.evaluation_criteria,
        )
        self.assertEqual([step.tool for step in plan.steps], ["final"])
        self.assertFalse(can_build_sandbox_program("正確に説明してください"))

    def test_evaluation_criteria_are_loaded_from_prompt_file(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return True

        plan = build_agent_plan("test.csvを要約して一覧にしてください", Config())

        self.assertIn(evaluation_criterion("rag"), plan.evaluation_criteria)
        self.assertIn(evaluation_criterion("summary"), plan.evaluation_criteria)
        self.assertIn(evaluation_criterion("list"), plan.evaluation_criteria)

    def test_agent_plan_uses_file_batch_for_folder_wide_summary(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name in {"file_batch", "rag"}

        plan = build_agent_plan("フォルダの中のすべてのファイル名と一言要約を表にしてください", Config())

        self.assertEqual([step.tool for step in plan.steps], ["file_batch"])

    def test_tool_selection_prefers_sandbox_for_calculation_when_enabled(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name == "sandbox" or (name == "rag" and default)

        decision = select_tool("123 * 456 を正確に計算して", Config())

        self.assertEqual(decision.name, "sandbox")

    def test_tool_selection_prefers_sandbox_for_image_when_enabled(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name == "sandbox" or (name == "rag" and default)

        decision = select_tool("画像を作成して保存してください", Config())

        self.assertEqual(decision.name, "sandbox")

    def test_agent_plan_can_include_rag_then_sandbox(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name in {"rag", "sandbox"}

        plan = build_agent_plan("ファイルの数を正確に計算して", Config())

        self.assertEqual([step.tool for step in plan.steps], ["rag", "sandbox"])

    def test_agent_plan_uses_rag_before_sandbox_for_named_csv(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return name in {"rag", "sandbox"}

        plan = build_agent_plan("test.csvの全てのテストの点を足し合わせてください。", Config())

        self.assertEqual([step.tool for step in plan.steps], ["rag", "sandbox"])

    @patch("agent.applications.llm_helpers.complete_response")
    def test_dynamic_replanner_can_replace_remaining_queue(self, mock_complete):
        class Config:
            dynamic_replanner = {
                "enabled": True,
                "reasoning_effort": True,
                "max_output_tokens": 128,
            }

            def tool_enabled(self, name, default=False):
                return name in {"dynamic_replanner", "sandbox"}

        state = AgentState.from_plan(
            AgentPlan(
                goal="Answer the user's request: calculate",
                evaluation_criteria=["The answer includes the computed result."],
                summary="rag -> sandbox -> final",
                steps=[AgentPlanStep("sandbox", "Run calculation.")],
            ),
            "calculate",
        )
        state.task_history.append(
            TaskExecutionRecord(
                task=AgentPlanStep("rag", "Search context."),
                ok=True,
                input_before="calculate",
                input_after="calculate with context",
                result="completed",
            )
        )
        mock_complete.return_value = json.dumps(
            {
                "action": "replace",
                "reason": "Need debugging before sandbox.",
                "tasks": [{"tool": "sandbox", "purpose": "Run a narrower debug calculation."}],
            }
        )

        decision = _replan_after_step(Config(), state)

        self.assertEqual(decision.action, "replace")
        self.assertEqual([step.tool for step in decision.plan_queue], ["sandbox"])
        self.assertIn("debug", decision.plan_queue[0].purpose)
        self.assertEqual(mock_complete.call_args.kwargs["max_output_tokens"], 128)
        self.assertEqual(mock_complete.call_args.kwargs["reasoning_effort"], "medium")

    @patch("agent.applications.llm_helpers.complete_response")
    def test_dynamic_finalizer_can_add_validation_tasks(self, mock_complete):
        class Config:
            dynamic_finalizer = {
                "enabled": True,
                "reasoning_effort": "minimal",
                "max_output_tokens": 80,
            }

            def tool_enabled(self, name, default=False):
                return name in {"dynamic_finalizer", "sandbox"}

        state = AgentState.from_plan(
            AgentPlan(
                goal="Answer the user's request: verify",
                evaluation_criteria=["The answer is verified."],
                summary="sandbox -> final",
                steps=[],
            ),
            "verify",
        )
        state.final_message = "temporary artifact"
        mock_complete.return_value = json.dumps(
            {
                "action": "add_tasks",
                "reason": "Needs one more validation.",
                "tasks": [{"tool": "sandbox", "purpose": "Validate the artifact."}],
            }
        )

        decision = _route_final_output(Config(), state)

        self.assertEqual(decision.action, "replace")
        self.assertEqual([step.tool for step in decision.plan_queue], ["sandbox"])
        self.assertEqual(mock_complete.call_args.kwargs["max_output_tokens"], 80)
        self.assertEqual(mock_complete.call_args.kwargs["reasoning_effort"], "minimal")

    @patch("agent.views.run_sandbox")
    @patch("agent.applications.sandbox.generate_sandbox_code")
    @patch("agent.applications.llm_helpers.complete_response")
    def test_dynamic_finalizer_added_sandbox_task_is_executed(self, mock_complete, mock_generate_code, mock_run_sandbox):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20
            dynamic_finalizer = {
                "enabled": True,
                "reasoning_effort": "minimal",
                "max_output_tokens": 80,
            }

            def tool_enabled(self, name, default=False):
                return name in {"dynamic_finalizer", "sandbox", "web_search"}

        mock_complete.side_effect = [
            json.dumps(
                {
                    "action": "add_tasks",
                    "reason": "Validate before final output.",
                    "tasks": [{"tool": "sandbox", "purpose": "Validate the artifact."}],
                }
            ),
            json.dumps({"action": "save", "reason": "Validated."}),
        ]
        mock_generate_code.return_value = "print('validated')"
        mock_run_sandbox.return_value = SandboxResult(True, "validated")

        plan = AgentPlan(
            goal="Answer the user's request: verify",
            evaluation_criteria=["The answer is verified."],
            summary="web_search -> final",
            steps=[AgentPlanStep("web_search", "Produce a temporary artifact.")],
        )
        runner = _execute_agent_plan(self.thread, "verify", Config(), plan)
        while True:
            try:
                next(runner)
            except StopIteration:
                break

        mock_generate_code.assert_called_once()
        mock_run_sandbox.assert_called_once()
        self.assertEqual(mock_complete.call_args_list[0].kwargs["max_output_tokens"], 80)
        self.assertEqual(mock_complete.call_args_list[0].kwargs["reasoning_effort"], "minimal")

    @patch("agent.views.run_sandbox")
    @patch("agent.applications.sandbox.generate_sandbox_code")
    def test_generated_sandbox_code_is_passed_directly_to_runner(self, mock_generate_code, mock_run_sandbox):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_generate_code.return_value = "print('computed')"
        mock_run_sandbox.return_value = SandboxResult(True, "computed")
        input_text = "\n".join(
            [
                "test.csvのテストの点を科目ごとの平均点と合計点を教えてください。",
                "RAG context from allowed local files:",
                "```text",
                "名前,テストの種類,テストの点",
                "佐藤太郎,国語,82",
                "佐藤太郎,数学,76",
                "```",
            ]
        )
        plan = AgentPlan(
            goal="Answer the user's request: grouped stats",
            evaluation_criteria=["The answer includes computed grouped results."],
            summary="sandbox -> final",
            steps=[AgentPlanStep("sandbox", "Compute grouped statistics.")],
        )

        runner = _execute_agent_plan(self.thread, input_text, Config(), plan)
        result = None
        while True:
            try:
                next(runner)
            except StopIteration as exc:
                result = exc.value
                break

        self.assertTrue(result["ok"])
        self.assertIn("Sandbox実行結果: 成功", result["final_message"])
        self.assertEqual(mock_run_sandbox.call_args.kwargs["code"], "print('computed')")

    @patch("agent.applications.sandbox.generate_sandbox_code")
    def test_generated_sandbox_code_retries_after_runtime_error(self, mock_generate_code):
        class Config:
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20
            sandbox_code_generation = {"llm_max_retries": 1}

            def tool_enabled(self, name, default=False):
                return name == "sandbox"

        mock_generate_code.side_effect = [
            "json",
            "import json\nprint(json.dumps({'maigent_sandbox_result': {'stdout': 'ok', 'artifacts': []}}))",
        ]
        # run_sandbox is imported independently into both agent.views and
        # agent.applications.sandbox (the retry-after-failure path lives there),
        # so both call sites must share one Mock to see a consistent call sequence.
        mock_run_sandbox = Mock(
            side_effect=[
                SandboxResult(False, "Traceback (most recent call last):\nNameError: name 'json' is not defined"),
                SandboxResult(True, "ok"),
            ]
        )
        input_text = "テスト結果のヒストグラムを作成してください。"
        plan = AgentPlan(
            goal="Create a histogram.",
            evaluation_criteria=["The sandbox succeeds."],
            summary="sandbox -> final",
            steps=[AgentPlanStep("sandbox", "Create histogram.")],
        )

        with patch("agent.views.run_sandbox", mock_run_sandbox), patch("agent.applications.sandbox.run_sandbox", mock_run_sandbox):
            runner = _execute_agent_plan(self.thread, input_text, Config(), plan)
            result = None
            while True:
                try:
                    next(runner)
                except StopIteration as exc:
                    result = exc.value
                    break

        self.assertTrue(result["ok"])
        self.assertIn("Sandbox実行結果: 成功", result["final_message"])
        self.assertEqual(mock_generate_code.call_count, 2)
        self.assertEqual(mock_run_sandbox.call_count, 2)
        self.assertIn("Previous generated sandbox code failed during execution", mock_generate_code.call_args_list[1].args[1])
        self.assertIn("NameError: name 'json' is not defined", mock_generate_code.call_args_list[1].args[1])
        self.assertIn("import json", mock_run_sandbox.call_args.kwargs["code"])

    @patch("agent.applications.llm_helpers.complete_response")
    def test_llm_tool_selector_can_choose_rag_and_sandbox(self, mock_complete):
        class Config:
            tool_selector = {
                "enabled": True,
                "reasoning_effort": "low",
                "max_output_tokens": 96,
            }

            def tool_enabled(self, name, default=False):
                return name in {"tool_selector", "rag", "sandbox"} or default

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "numbers.txt").write_text("1\n2\n", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_complete.return_value = json.dumps(
                {
                    "steps": [
                        {"tool": "rag", "purpose": "Find the numbers file."},
                        {"tool": "sandbox", "purpose": "Compute the exact sum."},
                    ],
                    "rag_query": "numbers",
                    "reason": "Needs file context and exact computation.",
                }
            )

            from .views import _build_agent_plan_with_llm_tool_selection

            plan = _build_agent_plan_with_llm_tool_selection(self.thread, "numbersファイルの合計を計算して", Config())

        self.assertIsNotNone(plan)
        self.assertEqual([step.tool for step in plan.steps], ["rag", "sandbox", "final"])
        self.assertEqual(plan.summary, "rag -> sandbox -> final (LLM-selected tools)")
        self.assertEqual(plan.rag_query, "numbers")
        kwargs = mock_complete.call_args.kwargs
        self.assertEqual(kwargs["max_output_tokens"], 96)
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["temperature"], 0)

    @patch("agent.applications.llm_helpers.complete_response")
    def test_llm_tool_selector_corrects_direct_answer_to_rag_for_local_context(self, mock_complete):
        class Config:
            tool_selector = {
                "enabled": True,
                "reasoning_effort": "none",
                "max_output_tokens": 96,
            }

            def tool_enabled(self, name, default=False):
                return name in {"tool_selector", "rag"} or default

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "text_4.txt").write_text("古書修復師の灰島ノエは京都の月返寺で写本を修復する。", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_complete.return_value = json.dumps(
                {
                    "steps": [{"tool": "final", "purpose": "Answer directly."}],
                    "rag_query": "",
                    "reason": "Direct answer is enough.",
                }
            )

            from .views import _build_agent_plan_with_llm_tool_selection

            plan = _build_agent_plan_with_llm_tool_selection(self.thread, "古書修復師の灰島ノエについて教えて", Config())

        self.assertIsNotNone(plan)
        self.assertEqual([step.tool for step in plan.steps], ["rag", "final"])
        self.assertEqual(plan.summary, "rag -> final (LLM-selected tools)")
        self.assertEqual(plan.rag_query, "古書修復師 灰島ノエ")
        self.assertIn("RAG", plan.evaluation_criteria[-1])

    def test_should_force_rag_for_local_context_ignores_generic_investigate_verbs(self):
        from .applications.planning import _should_force_rag_for_local_context

        # Generic "look into/search/check" verbs alone no longer force rag;
        # they are not specific to local files and previously over-triggered
        # via the broad _should_search() marker list.
        self.assertFalse(_should_force_rag_for_local_context("資料を検索して確認してください"))
        self.assertFalse(_should_force_rag_for_local_context("ちょっと調べてみてよ"))

    def test_should_force_rag_for_local_context_defers_to_explicit_web_instruction(self):
        from .applications.planning import _should_force_rag_for_local_context

        # Even though "について" is present, an explicit request to use the web
        # must not be silently overridden with a forced rag step.
        self.assertFalse(_should_force_rag_for_local_context("今治西校の校歌についてwebを調べてみてよ"))
        self.assertFalse(_should_force_rag_for_local_context("この件についてネットで検索して"))

    def test_should_force_rag_for_local_context_still_catches_named_entity_questions(self):
        from .applications.planning import _should_force_rag_for_local_context

        # The original, intended case (a possibly project-specific proper noun
        # with no explicit external-search instruction) still forces rag.
        self.assertTrue(_should_force_rag_for_local_context("古書修復師の灰島ノエについて教えて"))

    @patch("agent.applications.llm_helpers.complete_response")
    def test_llm_tool_selector_does_not_force_rag_when_web_explicitly_requested(self, mock_complete):
        class Config:
            tool_selector = {
                "enabled": True,
                "reasoning_effort": "none",
                "max_output_tokens": 96,
            }

            def tool_enabled(self, name, default=False):
                return name in {"tool_selector", "rag", "web_search"} or default

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "unrelated.txt").write_text("quarterly budget notes", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_complete.return_value = json.dumps(
                {
                    "steps": [{"tool": "web_search", "purpose": "Search the web as explicitly requested."}],
                    "rag_query": "",
                    "reason": "User explicitly asked for a web search.",
                }
            )

            from .views import _build_agent_plan_with_llm_tool_selection

            plan = _build_agent_plan_with_llm_tool_selection(self.thread, "今治西校の校歌についてwebを調べてみてよ", Config())

        self.assertIsNotNone(plan)
        self.assertEqual([step.tool for step in plan.steps], ["web_search", "final"])

    @patch("agent.applications.llm_helpers.complete_response")
    def test_llm_tool_selector_adds_file_batch_for_folder_wide_rag_plan(self, mock_complete):
        class Config:
            tool_selector = {
                "enabled": True,
                "reasoning_effort": "none",
                "max_output_tokens": 96,
            }

            def tool_enabled(self, name, default=False):
                return name in {"tool_selector", "rag", "file_batch"} or default

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "scores.csv").write_text("受験者名,科目名,得点\nA,数学,80\n", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_complete.return_value = json.dumps(
                {
                    "steps": [{"tool": "rag", "purpose": "Search input files."}],
                    "rag_query": "input scores",
                    "reason": "Needs local files.",
                }
            )

            from .views import _build_agent_plan_with_llm_tool_selection

            plan = _build_agent_plan_with_llm_tool_selection(
                self.thread,
                "フォルダの中のすべてのファイルを読んで、テスト集計を統合してください",
                Config(),
            )

        self.assertIsNotNone(plan)
        self.assertEqual([step.tool for step in plan.steps], ["file_batch", "rag", "final"])
        self.assertEqual(plan.summary, "file_batch -> rag -> final (LLM-selected tools)")

    @patch("agent.applications.llm_helpers.complete_response")
    def test_llm_tool_selector_retries_empty_response(self, mock_complete):
        class Config:
            def tool_enabled(self, name, default=False):
                return name in {"tool_selector", "rag"} or default

        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "notes.txt").write_text("alpha", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")
            mock_complete.side_effect = [
                None,
                json.dumps(
                    {
                        "steps": [{"tool": "rag", "purpose": "Search notes."}],
                        "rag_query": "alpha",
                        "reason": "Retry recovered.",
                    }
                ),
            ]

            from .views import _build_agent_plan_with_llm_tool_selection

            plan = _build_agent_plan_with_llm_tool_selection(self.thread, "alphaについて教えて", Config())

        self.assertIsNotNone(plan)
        self.assertEqual([step.tool for step in plan.steps], ["rag", "final"])
        self.assertEqual(mock_complete.call_count, 2)

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_attaches_allowed_file_context_from_plain_message(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            file_path = root / "note.txt"
            file_path.write_text("allowed context", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": f"このファイルを読んで {file_path}"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("RAG context from allowed local files", input_text)
        self.assertIn("allowed context", input_text)

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_does_not_attach_unallowed_file_context(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        with TemporaryDirectory() as root_dir:
            file_path = Path(root_dir) / "secret.txt"
            file_path.write_text("secret context", encoding="utf-8")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": f"このファイルを読んで {file_path}"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("十分な情報が見つかりません", assistant.content)
        mock_stream.assert_not_called()

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_auto_attaches_relevant_allowed_file_without_path(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "fantasy_story.txt").write_text("dragon castle forest", encoding="utf-8")
            (root / "budget.csv").write_text("amount,total", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "fantasy storyを要約して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("Agent plan:", input_text)
        self.assertIn("RAG search query: fantasy story", input_text)
        self.assertIn("RAG context from allowed local files", input_text)
        self.assertIn("fantasy_story.txt", input_text)
        self.assertIn("dragon castle forest", input_text)
        self.assertNotIn("budget.csv", input_text)

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_named_file_rag_prefers_exact_basename_match(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "test.csv").write_text("名前,テストの点\nA,80\n", encoding="utf-8")
            (root / "construct_prompt.md").write_text("test csv columns scores unrelated instructions", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "test.csvのテストの点を教えて"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("test.csv", input_text)
        self.assertIn("A,80", input_text)
        self.assertNotIn("construct_prompt.md", input_text)

    def test_rag_auto_candidates_exclude_project_output_path(self):
        with TemporaryDirectory() as input_dir, TemporaryDirectory() as output_dir:
            input_root = Path(input_dir)
            output_root = Path(output_dir)
            (input_root / "scores.csv").write_text("受験者名,科目名,得点\nA,数学,80\n", encoding="utf-8")
            (output_root / "maigent-output.txt").write_text("以前の回答: アクセス権がありません", encoding="utf-8")
            self.project.output_path = str(output_root)
            self.project.save(update_fields=["output_path"])
            ProjectAccessPath.objects.create(project=self.project, path=str(input_root), mode="read")
            ProjectAccessPath.objects.create(project=self.project, path=str(output_root), mode="write")

            documents = _collect_candidate_documents(self.thread)

        paths = [path.name for path, _text in documents]
        self.assertIn("scores.csv", paths)
        self.assertNotIn("maigent-output.txt", paths)

    def test_japanese_rag_terms_split_long_request_phrases(self):
        terms = _search_terms("inputフォルダ内にテストの集計をしたファイルがいくつかあります。それを統合して一つのファイルに保存してください")

        self.assertIn("input", terms)
        self.assertIn("テスト", terms)
        self.assertIn("集計", terms)
        self.assertIn("統合", terms)
        self.assertIn("保存", terms)
        self.assertNotIn("フォルダ内にテストの集計をしたファイルがいくつかあります", terms)

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_uses_rag_top_k_for_bm25(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        AppSetting.objects.create(key="rag_top_k", value="1")
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "alpha_one.txt").write_text("alpha alpha", encoding="utf-8")
            (root / "alpha_two.txt").write_text("alpha", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "alphaについて要約して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("alpha_one.txt", input_text)
        self.assertNotIn("alpha_two.txt", input_text)
        self.assertIn("files=alpha_one.txt", input_text)

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_llm_decision_can_select_rag_for_ambiguous_named_question(self, mock_config, mock_stream, mock_complete):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        mock_complete.return_value = "\n".join(
            [
                "RAG_REQUIRED",
                "QUERY: 深海図書館 アビス リブラ 本 保存",
                "REASON: named local setting",
            ]
        )
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "text_10.txt").write_text(
                "深海図書館アビス・リブラでは、本は棚に並ばず、読者の肺活量に合わせて泳いでくる。",
                encoding="utf-8",
            )
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "深海図書館アビス・リブラでは本はどのように保存されていますか？"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("RAG search query: 深海図書館 アビス リブラ 本 保存", input_text)
        self.assertIn("RAG context from allowed local files", input_text)
        self.assertIn("読者の肺活量に合わせて泳いでくる", input_text)
        mock_complete.assert_called_once()

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_returns_no_info_when_search_needed_but_bm25_not_relevant(self, mock_config, mock_stream, mock_complete):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "should not call")])
        mock_complete.return_value = '{"relevant_indexes": [], "reason": "unrelated"}'
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "budget.txt").write_text("cost amount", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "astronomyについて要約して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            body = b"".join(stream.streaming_content).decode()

        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("十分な情報が見つかりません", assistant.content)
        mock_stream.assert_not_called()
        mock_complete.assert_called_once()

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_bm25_inadequate_uses_llm_judge_for_top_k_files(self, mock_config, mock_stream, mock_complete):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        mock_complete.return_value = '{"relevant_indexes": [1], "reason": "contains relevant lore"}'
        AppSetting.objects.create(key="rag_top_k", value="1")
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "a_notes.txt").write_text("books are sealed in glass", encoding="utf-8")
            (root / "budget.txt").write_text("cost amount", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "深海図書館の本について要約して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("RAG context from allowed local files", input_text)
        self.assertIn("a_notes.txt", input_text)
        self.assertIn("sealed in glass", input_text)
        self.assertNotIn("budget.txt", input_text)
        mock_complete.assert_called_once()

    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_streaming_auto_attaches_allowed_directory_listing_for_list_request(self, mock_config, mock_stream):
        mock_config.return_value.model = "test-model"
        mock_config.return_value.api_key = "test-key"
        mock_config.return_value.base_url = ""
        mock_config.return_value.sources = []
        mock_stream.return_value = iter([("delta", "ok")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "alpha.txt").write_text("a", encoding="utf-8")
            (root / "nested").mkdir()
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "ファイルの一覧をください"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        input_text = mock_stream.call_args.args[1]
        self.assertIn("RAG context from allowed local files", input_text)
        self.assertIn("[file] alpha.txt", input_text)
        self.assertIn("[dir] nested", input_text)

    @patch("agent.views.run_sandbox")
    @patch("agent.views.load_runtime_config")
    def test_plan_runs_rag_then_sandbox_when_both_are_needed(self, mock_config, mock_run_sandbox):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name in {"rag", "sandbox"}

        mock_config.return_value = Config()
        mock_run_sandbox.return_value = SandboxResult(True, "2")
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "numbers.txt").write_text("1\n1\n", encoding="utf-8")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "numbersファイルの合計を正確に計算して"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        sandbox_input = mock_run_sandbox.call_args.args[0]
        self.assertIn("RAG context from allowed local files", sandbox_input)
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("Sandbox実行結果: 成功", assistant.content)

    @patch("agent.views.stream_response")
    @patch("agent.views.run_sandbox")
    @patch("agent.views.load_runtime_config")
    def test_named_csv_runs_rag_then_sandbox_without_llm_code_response(self, mock_config, mock_run_sandbox, mock_stream):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name in {"rag", "sandbox"}

        mock_config.return_value = Config()
        mock_run_sandbox.return_value = SandboxResult(True, "column: テストの点\nsum: 246")
        mock_stream.return_value = iter([("delta", "should not call")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "test.csv").write_text(
                "名前,テストの種類,テストの点\n佐藤太郎,国語,82\n佐藤太郎,数学,76\n佐藤太郎,英語,88\n",
                encoding="utf-8",
            )
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": "test.csvの全てのテストの点を足し合わせてください。"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        sandbox_input = mock_run_sandbox.call_args.args[0]
        self.assertIn("RAG context from allowed local files", sandbox_input)
        self.assertIn("test.csv", sandbox_input)
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("Sandbox実行結果: 成功", assistant.content)
        self.assertIn("sum: 246", assistant.content)
        mock_stream.assert_not_called()

    @patch("agent.views.stream_response")
    @patch("agent.views.run_sandbox")
    @patch("agent.applications.sandbox.generate_sandbox_code")
    @patch("agent.views.load_runtime_config")
    def test_grouped_named_csv_generates_code_then_runs_sandbox(
        self,
        mock_config,
        mock_generate_sandbox_code,
        mock_run_sandbox,
        mock_stream,
    ):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name in {"rag", "sandbox"}

        mock_config.return_value = Config()
        mock_generate_sandbox_code.return_value = "print('佐藤太郎: 246')"
        mock_run_sandbox.return_value = SandboxResult(True, "佐藤太郎: 246")
        mock_stream.return_value = iter([("delta", "should not call")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / "test.csv").write_text(
                "名前,テストの種類,テストの点\n佐藤太郎,国語,82\n佐藤太郎,数学,76\n佐藤太郎,英語,88\n",
                encoding="utf-8",
            )
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": "test.csvのテストの点を名前ごとに全ての科目を合計してください。"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        mock_generate_sandbox_code.assert_called_once()
        sandbox_input = mock_run_sandbox.call_args.args[0]
        self.assertIn("Generated sandbox program", sandbox_input)
        self.assertIn("print('佐藤太郎: 246')", sandbox_input)
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("Sandbox実行結果: 成功", assistant.content)
        self.assertIn("佐藤太郎: 246", assistant.content)
        mock_stream.assert_not_called()

    @patch("agent.views.stream_response")
    @patch("agent.views.run_sandbox")
    @patch("agent.applications.sandbox.generate_sandbox_code")
    @patch("agent.views.load_runtime_config")
    def test_llm_sandbox_code_gets_host_dataset_api_for_named_csv(
        self,
        mock_config,
        mock_generate_sandbox_code,
        mock_run_sandbox,
        mock_stream,
    ):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            sandbox_allowed_libraries = []
            sandbox_image = "python:3.11-slim"
            sandbox_timeout_seconds = 20

            def tool_enabled(self, name, default=False):
                return name in {"rag", "sandbox"}

        mock_config.return_value = Config()
        mock_generate_sandbox_code.return_value = "df = load_dataset('rag_1')\nprint(df['テストの点'].sum())"
        mock_run_sandbox.return_value = SandboxResult(True, "246")
        mock_stream.return_value = iter([("delta", "should not call")])
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            source = root / "test.csv"
            source.write_text(
                "名前,テストの種類,テストの点\n佐藤太郎,国語,82\n佐藤太郎,数学,76\n佐藤太郎,英語,88\n",
                encoding="utf-8",
            )
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            create = self.client.post(
                reverse("send_message", args=[self.thread.id]),
                {"message": "test.csvのテストの点のヒストグラムを作り画像を表示してください。"},
            )
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            b"".join(stream.streaming_content)

        generation_prompt = mock_generate_sandbox_code.call_args.args[1]
        self.assertIn("Host-provided sandbox dataset API", generation_prompt)
        self.assertIn("id: rag_1", generation_prompt)
        self.assertIn("columns: 名前, テストの種類, テストの点", generation_prompt)
        datasets = mock_run_sandbox.call_args.kwargs["datasets"]
        self.assertEqual(datasets[0].id, "rag_1")
        self.assertEqual(datasets[0].text.splitlines()[0], "名前,テストの種類,テストの点")
        self.assertIn("load_dataset('rag_1')", mock_run_sandbox.call_args.kwargs["code"])
        assistant = Message.objects.get(id=payload["assistant_id"])
        self.assertIn("Sandbox実行結果: 成功", assistant.content)

    def test_approval_actions_never_execute_command(self):
        approval = ApprovalRequest.objects.create(thread=self.thread, command="rm -rf /tmp/example")

        response = self.client.post(reverse("approval_action", args=[approval.id]), {"action": "approve"})

        self.assertEqual(response.status_code, 302)
        approval.refresh_from_db()
        self.assertEqual(approval.status, "approved")

    def test_add_access_path_normalizes_and_saves_mode(self):
        with TemporaryDirectory() as root_dir:
            response = self.client.post(
                reverse("add_access_path", args=[self.project.id]),
                {"path": root_dir, "mode": "write", "note": "workspace"},
            )

        self.assertEqual(response.status_code, 302)
        access = ProjectAccessPath.objects.get(project=self.project)
        self.assertEqual(access.mode, "write")
        self.assertEqual(access.note, "workspace")
        self.assertEqual(access.path, normalize_access_path(root_dir))

    def test_update_output_path_normalizes_and_saves_folder(self):
        with TemporaryDirectory() as root_dir:
            response = self.client.post(
                reverse("update_output_path", args=[self.project.id]),
                {"output_path": root_dir},
            )

        self.assertEqual(response.status_code, 302)
        self.project.refresh_from_db()
        self.assertEqual(self.project.output_path, normalize_access_path(root_dir))

    def test_delete_access_path_removes_entry(self):
        access = ProjectAccessPath.objects.create(project=self.project, path="/tmp/example", mode="read")

        response = self.client.post(reverse("delete_access_path", args=[access.id]))

        self.assertEqual(response.status_code, 302)
        self.assertFalse(ProjectAccessPath.objects.filter(id=access.id).exists())

    def test_access_helper_distinguishes_read_and_write(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            child = root / "child.txt"
            child.write_text("x")
            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="read")

            self.assertTrue(is_path_allowed(self.project, str(child), write=False))
            self.assertFalse(is_path_allowed(self.project, str(child), write=True))

            ProjectAccessPath.objects.create(project=self.project, path=str(root), mode="write")
            self.assertTrue(is_path_allowed(self.project, str(child), write=True))

    def test_access_helper_rejects_symlink_escaping_allowed_root(self):
        with TemporaryDirectory() as allowed_dir, TemporaryDirectory() as outside_dir:
            allowed_root = Path(allowed_dir)
            outside_root = Path(outside_dir)
            secret = outside_root / "secret.txt"
            secret.write_text("outside secret")
            escape_link = allowed_root / "escape"
            escape_link.symlink_to(outside_root, target_is_directory=True)
            ProjectAccessPath.objects.create(project=self.project, path=str(allowed_root), mode="read")

            self.assertFalse(is_path_allowed(self.project, str(escape_link / "secret.txt"), write=False))
            self.assertFalse(is_path_allowed(self.project, str(outside_root), write=False))
            self.assertTrue(is_path_allowed(self.project, str(allowed_root / "inside.txt"), write=False))


class AgentsMdAndSkillTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="SkillProj", path="")
        self.thread = Thread.objects.create(project=self.project, title="Main")

    def _set_project_path(self, root_dir: str) -> None:
        self.project.path = root_dir
        self.project.save(update_fields=["path"])

    def test_load_agents_md_includes_project_layer(self):
        with TemporaryDirectory() as root_dir:
            maigent_dir = Path(root_dir) / ".maigent"
            maigent_dir.mkdir()
            (maigent_dir / "AGENTS.md").write_text("Follow the project style guide.", encoding="utf-8")

            content = load_agents_md(root_dir)

        self.assertIn("Follow the project style guide.", content)

    def test_load_agents_md_returns_empty_when_absent(self):
        with TemporaryDirectory() as root_dir:
            content = load_agents_md(root_dir)

        self.assertNotIn("Follow the project style guide.", content)

    def test_build_instructions_includes_project_agents_md(self):
        with TemporaryDirectory() as root_dir:
            self._set_project_path(root_dir)
            maigent_dir = Path(root_dir) / ".maigent"
            maigent_dir.mkdir()
            (maigent_dir / "AGENTS.md").write_text("常に丁寧語で回答してください。", encoding="utf-8")

            instructions = _build_instructions(self.thread)

        self.assertIn("Project AGENTS.md instructions:", instructions)
        self.assertIn("常に丁寧語で回答してください。", instructions)

    def test_load_skills_parses_frontmatter_and_body(self):
        with TemporaryDirectory() as root_dir:
            skill_dir = Path(root_dir) / ".maigent" / "skills" / "weekly-report"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: weekly-report\ndescription: Summarize weekly sales figures.\n---\n\nStep 1: gather data.\nStep 2: summarize.",
                encoding="utf-8",
            )

            skills = load_skills(root_dir)

        found = [skill for skill in skills if skill.name == "weekly-report"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].description, "Summarize weekly sales figures.")
        self.assertIn("Step 1: gather data.", found[0].body)

    def test_load_skills_falls_back_to_directory_name_without_frontmatter(self):
        with TemporaryDirectory() as root_dir:
            skill_dir = Path(root_dir) / ".maigent" / "skills" / "my-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("Just do the thing.", encoding="utf-8")

            skills = load_skills(root_dir)

        found = [skill for skill in skills if skill.name == "my-skill"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].description, "")
        self.assertEqual(found[0].body, "Just do the thing.")

    def test_load_skills_ignores_directories_without_skill_md(self):
        with TemporaryDirectory() as root_dir:
            (Path(root_dir) / ".maigent" / "skills" / "empty-dir").mkdir(parents=True)

            skills = load_skills(root_dir)

        # Only checks that this test's own (empty) project layer contributed
        # nothing; the app-level .maigent/skills/ layer may legitimately have
        # its own real skills, which load_skills() also returns.
        self.assertFalse(any(skill.path.startswith(root_dir) for skill in skills))

    def test_repo_sample_skill_is_not_loaded(self):
        # .maigent/skill/sample/SKILL.md (singular "skill") is a documentation
        # template, deliberately outside .maigent/skills/ (plural) so it never
        # becomes an active tool-selector candidate for real projects.
        skills = load_skills("")

        self.assertNotIn("sample-skill", {skill.name for skill in skills})

    def test_repo_sample_agents_md_is_not_loaded(self):
        # .maigent/AGENTS.sample.md is a documentation template (same convention
        # as config.yaml.sample); load_agents_md() only reads the exact filename
        # "AGENTS.md", so this sample never affects real instructions.
        content = load_agents_md("")

        self.assertNotIn("常に丁寧語で回答してください", content)

    def test_available_tool_specs_includes_discovered_skill(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return False

        with TemporaryDirectory() as root_dir:
            self._set_project_path(root_dir)
            skill_dir = Path(root_dir) / ".maigent" / "skills" / "weekly-report"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: weekly-report\ndescription: Summarize weekly sales figures.\n---\n\nBody", encoding="utf-8"
            )

            specs = _available_tool_specs(self.thread, Config())

        spec_by_name = {spec["name"]: spec for spec in specs}
        self.assertIn("skill:weekly-report", spec_by_name)
        self.assertEqual(spec_by_name["skill:weekly-report"]["description"], "Summarize weekly sales figures.")

    def test_allowed_plan_tools_includes_discovered_skill_names(self):
        class Config:
            def tool_enabled(self, name, default=False):
                return False

        with TemporaryDirectory() as root_dir:
            self._set_project_path(root_dir)
            skill_dir = Path(root_dir) / ".maigent" / "skills" / "weekly-report"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: weekly-report\n---\nBody", encoding="utf-8")

            allowed_with_thread = _allowed_plan_tools(Config(), self.thread)
            allowed_without_thread = _allowed_plan_tools(Config())

        self.assertIn("skill:weekly-report", allowed_with_thread)
        self.assertIn("final", allowed_with_thread)
        self.assertNotIn("skill:weekly-report", allowed_without_thread)

    @patch("agent.applications.llm_helpers.complete_response")
    @patch("agent.views.stream_response")
    @patch("agent.views.load_runtime_config")
    def test_skill_selected_by_tool_selector_is_applied(self, mock_config, mock_stream, mock_complete):
        class Config:
            model = "test-model"
            api_key = "test-key"
            base_url = ""
            sources = []
            final_evaluation_enabled = False

            def tool_enabled(self, name, default=False):
                return name == "tool_selector"

        def fake_complete(config, prompt, instructions, **kwargs):
            if "ツールを選択" in instructions:
                return '{"steps": [{"tool": "skill:weekly-report", "purpose": "apply weekly report skill"}], "reason": "matches skill"}'
            return '{"needs_clarification": false, "reason": "clear", "questions": []}'

        with TemporaryDirectory() as root_dir:
            self._set_project_path(root_dir)
            skill_dir = Path(root_dir) / ".maigent" / "skills" / "weekly-report"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: weekly-report\ndescription: Summarize weekly sales figures in a fixed format.\n---\n\n"
                "Always start the answer with 'Weekly Report:' and list three bullet points.",
                encoding="utf-8",
            )
            mock_config.return_value = Config()
            mock_complete.side_effect = fake_complete
            mock_stream.return_value = iter([("delta", "Weekly Report: ...")])

            create = self.client.post(reverse("send_message", args=[self.thread.id]), {"message": "今週の売上をまとめて"})
            payload = create.json()
            stream = self.client.get(payload["stream_url"])
            body = b"".join(stream.streaming_content).decode()

        self.assertIn("Weekly Report: ...", body)
        answer_generation_input = mock_stream.call_args.args[1]
        self.assertIn("Always start the answer with 'Weekly Report:'", answer_generation_input)

    def test_execute_agent_task_reports_missing_skill(self):
        from .views import _execute_agent_task

        with TemporaryDirectory() as root_dir:
            self._set_project_path(root_dir)

            runner = _execute_agent_task(
                self.thread,
                "hello",
                object(),
                "input text",
                [],
                AgentPlanStep("skill:missing", "apply missing skill"),
            )
            outcome = None
            while True:
                try:
                    next(runner)
                except StopIteration as exc:
                    outcome = exc.value
                    break

        self.assertFalse(outcome["ok"])
        self.assertIn("見つかりませんでした", outcome["final_message"])


@override_settings(STATICFILES_DIRS=[])
class AdminTests(TestCase):
    def setUp(self):
        self.admin_user = get_user_model().objects.create_superuser(
            username="admin-test",
            email="admin@example.com",
            password="test-password",
        )
        self.client.force_login(self.admin_user)

    def test_all_agent_models_are_registered(self):
        expected_models = {
            Project,
            Thread,
            ProjectAccessPath,
            Message,
            AppSetting,
            FeatureFlag,
            Automation,
            ApprovalRequest,
            AgentRun,
            AgentWorkerRun,
            AgentTaskRecord,
        }

        self.assertTrue(expected_models.issubset(admin.site._registry))

    def test_all_agent_model_changelists_are_accessible(self):
        models = [
            Project,
            Thread,
            ProjectAccessPath,
            Message,
            AppSetting,
            FeatureFlag,
            Automation,
            ApprovalRequest,
            AgentRun,
            AgentWorkerRun,
            AgentTaskRecord,
        ]

        for model in models:
            with self.subTest(model=model.__name__):
                url = reverse(f"admin:{model._meta.app_label}_{model._meta.model_name}_changelist")
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)

    def test_app_setting_can_be_edited_in_admin(self):
        setting = AppSetting.objects.create(key="admin_test", value="before")
        url = reverse("admin:agent_appsetting_change", args=[setting.pk])

        response = self.client.post(
            url,
            {"key": "admin_test", "value": "after", "_save": "Save"},
        )

        self.assertEqual(response.status_code, 302)
        setting.refresh_from_db()
        self.assertEqual(setting.value, "after")


@override_settings(STATICFILES_DIRS=[])
class DirectoryBrowseTests(TestCase):
    def test_browse_directories_lists_visible_child_folders(self):
        with TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").mkdir()
            (root / ".hidden").mkdir()
            (root / "notes.txt").write_text("not a directory")

            response = self.client.get(reverse("browse_directories"), {"path": str(root)})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["current"], str(root.resolve()))
            self.assertEqual(len(payload["directories"]), 1)
            self.assertEqual(payload["directories"][0]["name"], "repo")
            self.assertTrue(payload["directories"][0]["is_repo"])

    def test_browse_directories_rejects_missing_path(self):
        response = self.client.get(reverse("browse_directories"), {"path": "/definitely/missing/path"})

        self.assertEqual(response.status_code, 404)


class OpenAIClientTests(TestCase):
    @patch("agent.openai_client._complete_response")
    def test_generate_sandbox_code_passes_section_options(self, mock_complete_response):
        config = RuntimeConfig(
            values={
                "model": "model",
                "api_key": "key",
                "sandbox_code_generation": {
                    "enabled": True,
                    "max_output_tokens": 32768,
                    "reasoning_effort": "xhigh",
                },
            },
            sources=[],
        )
        mock_complete_response.return_value = "```python\nprint('ok')\n```"

        from .openai_client import generate_sandbox_code

        result = generate_sandbox_code(config, "generate code")

        self.assertEqual(result, "print('ok')")
        self.assertEqual(mock_complete_response.call_args.kwargs["max_output_tokens"], 32768)
        self.assertEqual(mock_complete_response.call_args.kwargs["reasoning_effort"], "xhigh")

    def test_auto_mode_falls_back_to_chat_completions(self):
        class Config:
            model = "model"
            api_key = "key"
            base_url = ""
            api_mode = "auto"

        class Responses:
            def stream(self, **kwargs):
                raise RuntimeError("responses unsupported")

        class ChatCompletions:
            def create(self, **kwargs):
                delta = type("Delta", (), {"content": "ok"})()
                choice = type("Choice", (), {"delta": delta})()
                yield type("Chunk", (), {"id": "chat_1", "choices": [choice]})()

        class Chat:
            completions = ChatCompletions()

        class Client:
            responses = Responses()
            chat = Chat()

        with patch("agent.openai_client.OpenAI", return_value=Client()):
            events = list(stream_response(Config(), "hello", "system"))

        self.assertEqual(events, [("delta", "ok"), ("response_id", "chat_1")])

    def test_auto_complete_response_falls_back_to_chat_when_responses_is_empty(self):
        class Config:
            model = "model"
            api_key = "key"
            base_url = ""
            api_mode = "auto"

        class Responses:
            def create(self, **kwargs):
                return type("Response", (), {"output_text": "", "output": []})()

        class ChatCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = type("Message", (), {"content": "chat ok"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        class Chat:
            completions = ChatCompletions()

        class Client:
            responses = Responses()
            chat = Chat()

        with patch("agent.openai_client.OpenAI", return_value=Client()):
            from .openai_client import complete_response

            result = complete_response(Config(), "hello", "system", max_output_tokens=64, temperature=0)

        self.assertEqual(result, "chat ok")
        self.assertEqual(Client.chat.completions.kwargs["max_tokens"], 64)
        self.assertEqual(Client.chat.completions.kwargs["temperature"], 0)

    def test_chat_completion_empty_content_logs_response_structure(self):
        class Config:
            model = "model"
            api_key = "key"
            base_url = ""
            api_mode = "chat"

        class ChatCompletions:
            def create(self, **kwargs):
                message = type("Message", (), {"role": "assistant", "content": ""})()
                choice = type("Choice", (), {"index": 0, "finish_reason": "stop", "message": message})()
                return type(
                    "Response",
                    (),
                    {"id": "chat_1", "model": "model", "object": "chat.completion", "usage": {"total_tokens": 12}, "choices": [choice]},
                )()

        class Chat:
            completions = ChatCompletions()

        class Client:
            chat = Chat()

        with patch("agent.openai_client.OpenAI", return_value=Client()):
            from .openai_client import complete_response

            with self.assertLogs("agent.openai_client", level="WARNING") as logs:
                result = complete_response(Config(), "hello", "system")

        self.assertEqual(result, "")
        self.assertIn("CHAT_COMPLETION_EMPTY_RESPONSE reason=empty_message_content", "\n".join(logs.output))
        self.assertIn("'messages_len': 2", "\n".join(logs.output))
        self.assertIn("'role': 'system'", "\n".join(logs.output))
        self.assertIn("'content_preview': 'system'", "\n".join(logs.output))
        self.assertIn("'role': 'user'", "\n".join(logs.output))
        self.assertIn("'content_preview': 'hello'", "\n".join(logs.output))
        self.assertIn("'finish_reason': 'stop'", "\n".join(logs.output))
        self.assertIn("'content_len': 0", "\n".join(logs.output))

    def test_ollama_uses_openai_compatible_client_with_default_base_url_and_dummy_key(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "ollama": {
                        "enabled": True,
                        "model": "llama3.1",
                    }
                }
            },
            sources=[],
        )

        class ChatCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = type("Message", (), {"content": "local ok"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        class Chat:
            completions = ChatCompletions()

        class Client:
            chat = Chat()

        with patch("agent.openai_client.OpenAI", return_value=Client()) as mock_openai:
            from .openai_client import complete_response

            result = complete_response(config, "hello")

        self.assertEqual(result, "local ok")
        mock_openai.assert_called_once_with(api_key="ollama", base_url="http://localhost:11434/v1")

    def test_openai_compatible_uses_dummy_key_and_merges_chat_request_options(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "openai_compatible": {
                        "enabled": True,
                        "model": "Qwen/Qwen3.8-27B",
                        "base_url": "http://localhost:8000/v1",
                        "request": {
                            "model": "ignored-model",
                            "messages": [{"role": "user", "content": "ignored"}],
                            "input": "ignored-input",
                            "instructions": "ignored-instructions",
                            "stream": True,
                            "max_tokens": 16,
                            "max_completion_tokens": 32,
                            "temperature": 0.7,
                            "reasoning_effort": "xhigh",
                            "stream_options": {"include_usage": True},
                            "extra_body": {
                                "chat_template_kwargs": {
                                    "enable_thinking": True,
                                    "preserve_thinking": True,
                                }
                            },
                        },
                    }
                }
            },
            sources=[],
        )

        class ChatCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = type("Message", (), {"content": "qwen ok"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        class Chat:
            completions = ChatCompletions()

        class Client:
            chat = Chat()

        with patch("agent.openai_client.OpenAI", return_value=Client()) as mock_openai:
            from .openai_client import complete_response

            result = complete_response(
                config,
                "hello",
                "system",
                max_output_tokens=64,
                reasoning_effort="low",
                temperature=0,
            )

        self.assertEqual(result, "qwen ok")
        mock_openai.assert_called_once_with(api_key="openai_compatible", base_url="http://localhost:8000/v1")
        request = Client.chat.completions.kwargs
        self.assertEqual(request["model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(request["messages"], [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}])
        self.assertFalse(request["stream"])
        self.assertNotIn("input", request)
        self.assertNotIn("instructions", request)
        self.assertNotIn("stream_options", request)
        self.assertEqual(request["max_tokens"], 64)
        self.assertNotIn("max_completion_tokens", request)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
        )

    def test_chat_reasoning_effort_none_removes_provider_default(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "openai_compatible": {
                        "enabled": True,
                        "model": "model",
                        "base_url": "http://localhost:8000/v1",
                        "request": {"reasoning_effort": "xhigh"},
                    }
                }
            },
            sources=[],
        )

        class ChatCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = type("Message", (), {"content": "ok"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        class Chat:
            completions = ChatCompletions()

        class Client:
            chat = Chat()

        with patch("agent.openai_client.OpenAI", return_value=Client()):
            from .openai_client import complete_response

            complete_response(config, "hello", reasoning_effort="none")

        self.assertNotIn("reasoning_effort", Client.chat.completions.kwargs)

    def test_streaming_chat_keeps_stream_options_and_reserved_fields(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "openai_compatible": {
                        "enabled": True,
                        "model": "served-model",
                        "base_url": "http://localhost:8000/v1",
                        "request": {
                            "model": "ignored-model",
                            "messages": [],
                            "stream": False,
                            "reasoning_effort": "none",
                            "stream_options": {"include_usage": True},
                        },
                    }
                }
            },
            sources=[],
        )

        class ChatCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                delta = type("Delta", (), {"content": "ok"})()
                choice = type("Choice", (), {"delta": delta})()
                return iter([type("Chunk", (), {"id": "chat_1", "choices": [choice]})()])

        class Chat:
            completions = ChatCompletions()

        class Client:
            chat = Chat()

        with patch("agent.openai_client.OpenAI", return_value=Client()):
            events = list(stream_response(config, "hello", "system"))

        self.assertEqual(events, [("delta", "ok"), ("response_id", "chat_1")])
        request = Client.chat.completions.kwargs
        self.assertEqual(request["model"], "served-model")
        self.assertEqual(request["messages"], [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}])
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertNotIn("reasoning_effort", request)

    def test_openai_compatible_requires_base_url(self):
        config = RuntimeConfig(
            values={"providers": {"openai_compatible": {"enabled": True, "model": "model"}}},
            sources=[],
        )

        with self.assertRaisesRegex(ValueError, "base_url"):
            list(stream_response(config, "hello"))

    def test_request_options_are_shared_by_openai_compatible_providers(self):
        provider_configs = {
            "openai": {"api_key": "key"},
            "openai_compatible": {"base_url": "http://localhost:8000/v1"},
            "ollama": {},
            "lmstudio": {},
            "openrouter": {"api_key": "key"},
        }

        for provider, provider_values in provider_configs.items():
            with self.subTest(provider=provider):
                config = RuntimeConfig(
                    values={
                        "providers": {
                            provider: {
                                "enabled": True,
                                "model": "model",
                                "request": {"extra_body": {"top_k": 20}},
                                **provider_values,
                            }
                        }
                    },
                    sources=[],
                )

                class ChatCompletions:
                    def create(self, **kwargs):
                        self.kwargs = kwargs
                        message = type("Message", (), {"content": "ok"})()
                        choice = type("Choice", (), {"message": message})()
                        return type("Response", (), {"choices": [choice]})()

                class Chat:
                    completions = ChatCompletions()

                class Client:
                    chat = Chat()

                with patch("agent.openai_client.OpenAI", return_value=Client()):
                    from .openai_client import complete_response

                    complete_response(config, "hello")

                self.assertEqual(Client.chat.completions.kwargs["extra_body"], {"top_k": 20})

    def test_azure_provider_uses_azure_client(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "azure": {
                        "enabled": True,
                        "model": "deployment",
                        "api_key": "azure-key",
                        "azure_endpoint": "https://example.openai.azure.com",
                        "api_version": "2024-02-15-preview",
                    }
                }
            },
            sources=[],
        )

        class ChatCompletions:
            def create(self, **kwargs):
                message = type("Message", (), {"content": "azure ok"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        class Chat:
            completions = ChatCompletions()

        class Client:
            chat = Chat()

        with patch("agent.openai_client.AzureOpenAI", return_value=Client()) as mock_azure:
            from .openai_client import complete_response

            result = complete_response(config, "hello")

        self.assertEqual(result, "azure ok")
        mock_azure.assert_called_once_with(
            api_key="azure-key",
            azure_endpoint="https://example.openai.azure.com",
            api_version="2024-02-15-preview",
        )

    def test_bedrock_provider_uses_converse(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "bedrock": {
                        "enabled": True,
                        "model": "anthropic.test-model",
                        "region": "ap-northeast-1",
                    }
                }
            },
            sources=[],
        )

        class BedrockClient:
            def converse(self, **kwargs):
                self.kwargs = kwargs
                return {"output": {"message": {"content": [{"text": "bedrock ok"}]}}}

        with patch("agent.openai_client._bedrock_client", return_value=BedrockClient()):
            from .openai_client import complete_response

            result = complete_response(config, "hello", "system")

        self.assertEqual(result, "bedrock ok")

    def test_complete_response_passes_compact_response_options(self):
        config = RuntimeConfig(values={"model": "gpt-test", "api_key": "key", "api_mode": "responses"}, sources=[])

        class Responses:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_text": "ok"})()

        class Client:
            responses = Responses()

        with patch("agent.openai_client.OpenAI", return_value=Client()):
            from .openai_client import complete_response

            result = complete_response(config, "hello", max_output_tokens=64, reasoning_effort="none", temperature=0)

        self.assertEqual(result, "ok")
        self.assertEqual(Client.responses.kwargs["max_output_tokens"], 64)
        self.assertEqual(Client.responses.kwargs["reasoning"], {"effort": "none"})
        self.assertEqual(Client.responses.kwargs["temperature"], 0)

    def test_responses_api_merges_request_options_and_ignores_reserved_fields(self):
        config = RuntimeConfig(
            values={
                "providers": {
                    "openai": {
                        "enabled": True,
                        "model": "gpt-test",
                        "api_key": "key",
                        "api_mode": "responses",
                        "request": {
                            "model": "ignored-model",
                            "input": "ignored-input",
                            "instructions": "ignored-instructions",
                            "stream": True,
                            "stream_options": {"include_usage": True},
                            "extra_body": {"custom_option": True},
                        },
                    }
                }
            },
            sources=[],
        )

        class Responses:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_text": "ok"})()

        class Client:
            responses = Responses()

        with patch("agent.openai_client.OpenAI", return_value=Client()):
            from .openai_client import complete_response

            result = complete_response(config, "hello", "system")

        self.assertEqual(result, "ok")
        self.assertEqual(Client.responses.kwargs["model"], "gpt-test")
        self.assertEqual(Client.responses.kwargs["input"], "hello")
        self.assertEqual(Client.responses.kwargs["instructions"], "system")
        self.assertNotIn("stream", Client.responses.kwargs)
        self.assertNotIn("stream_options", Client.responses.kwargs)
        self.assertEqual(Client.responses.kwargs["extra_body"], {"custom_option": True})


class WebSearchTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Repo", is_current=True)
        self.thread = Thread.objects.create(project=self.project, title="Main")

    def test_search_web_reports_not_configured_without_api_key(self):
        from .applications.web_search import search_web

        class Config:
            web_search_api_key = ""
            web_search_max_results = 5
            web_search_timeout_seconds = 10

        result = search_web(Config(), "latest django release")

        self.assertFalse(result.ok)
        self.assertIn("未設定", result.message)

    @patch("agent.applications.web_search.urllib.request.urlopen")
    def test_search_web_returns_parsed_results_when_configured(self, mock_urlopen):
        from .applications.web_search import search_web

        class Config:
            web_search_api_key = "tvly-test-key"
            web_search_max_results = 5
            web_search_timeout_seconds = 10

        response_body = json.dumps(
            {
                "results": [
                    {"title": "Django 5.1 released", "url": "https://example.com/django-5-1", "content": "Release notes."},
                ]
            }
        ).encode("utf-8")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return response_body

        mock_urlopen.return_value = Response()

        result = search_web(Config(), "django latest release")

        self.assertTrue(result.ok)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].title, "Django 5.1 released")
        self.assertEqual(result.results[0].url, "https://example.com/django-5-1")

    @patch("agent.applications.web_search.urllib.request.urlopen")
    def test_search_web_reports_no_results(self, mock_urlopen):
        from .applications.web_search import search_web

        class Config:
            web_search_api_key = "tvly-test-key"
            web_search_max_results = 5
            web_search_timeout_seconds = 10

        response_body = json.dumps({"results": []}).encode("utf-8")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return response_body

        mock_urlopen.return_value = Response()

        result = search_web(Config(), "an extremely obscure query")

        self.assertFalse(result.ok)
        self.assertIn("見つかりませんでした", result.message)

    @patch("agent.views.search_web")
    def test_execute_agent_task_reports_web_search_not_configured(self, mock_search_web):
        from .applications.web_search import WebSearchResult
        from .tooling import AgentPlanStep
        from .views import _execute_agent_task

        mock_search_web.return_value = WebSearchResult(ok=False, query="latest news", message="web_searchのAPIキーが未設定です。")

        runner = _execute_agent_task(
            self.thread,
            "最新のニュースを調べて",
            object(),
            "answer base",
            [],
            AgentPlanStep("web_search", "Collect current information."),
        )
        result = None
        while True:
            try:
                next(runner)
            except StopIteration as exc:
                result = exc.value
                break

        self.assertFalse(result["ok"])
        self.assertIn("未設定", result["final_message"])

    @patch("agent.views.search_web")
    def test_execute_agent_task_attaches_web_search_context(self, mock_search_web):
        from .applications.web_search import WebSearchItem, WebSearchResult
        from .tooling import AgentPlanStep
        from .views import _execute_agent_task

        mock_search_web.return_value = WebSearchResult(
            ok=True,
            query="latest django release",
            results=(WebSearchItem(title="Django 5.1 released", url="https://example.com", snippet="Release notes."),),
        )

        runner = _execute_agent_task(
            self.thread,
            "Djangoの最新リリースを調べて",
            object(),
            "answer base",
            [],
            AgentPlanStep("web_search", "Collect current information."),
        )
        result = None
        while True:
            try:
                next(runner)
            except StopIteration as exc:
                result = exc.value
                break

        self.assertTrue(result["ok"])
        self.assertIn("Web search results", result["input_text"])
        self.assertIn("Django 5.1 released", result["input_text"])
        self.assertIn("https://example.com", result["input_text"])

    @patch("agent.views.search_web")
    def test_execute_agent_task_uses_cleaned_query_for_web_search(self, mock_search_web):
        from .applications.web_search import WebSearchItem, WebSearchResult
        from .tooling import AgentPlanStep
        from .views import _execute_agent_task

        mock_search_web.return_value = WebSearchResult(
            ok=True,
            query="今治西高校 校歌",
            results=(WebSearchItem(title="今治西高校の校歌", url="https://example.com", snippet="..."),),
        )

        runner = _execute_agent_task(
            self.thread,
            "じゃ、ちょっと今治西校の校歌についてwebを調べてみてよ",
            object(),
            "answer base",
            [],
            AgentPlanStep("web_search", "Search the web."),
        )
        while True:
            try:
                next(runner)
            except StopIteration:
                break

        called_query = mock_search_web.call_args.args[1]
        self.assertNotEqual(called_query, "じゃ、ちょっと今治西校の校歌についてwebを調べてみてよ")
        self.assertIn("今治西", called_query)

    def test_execute_agent_task_rag_no_context_continues_when_more_steps_queued(self):
        from .tooling import AgentPlanStep
        from .views import _execute_agent_task

        runner = _execute_agent_task(
            self.thread,
            "今治西校の校歌について教えて",
            object(),
            "answer base",
            [],
            AgentPlanStep("rag", "Search local files."),
            "今治西高校 校歌",
            None,
            True,
        )
        result = None
        while True:
            try:
                next(runner)
            except StopIteration as exc:
                result = exc.value
                break

        self.assertTrue(result["ok"])
        self.assertEqual(result["final_message"], "")

    def test_execute_agent_task_rag_no_context_stops_when_no_more_steps(self):
        from .tooling import AgentPlanStep
        from .views import _execute_agent_task

        runner = _execute_agent_task(
            self.thread,
            "今治西校の校歌について教えて",
            object(),
            "answer base",
            [],
            AgentPlanStep("rag", "Search local files."),
            "今治西高校 校歌",
        )
        result = None
        while True:
            try:
                next(runner)
            except StopIteration as exc:
                result = exc.value
                break

        self.assertTrue(result["ok"])
        self.assertIn("見つかりませんでした", result["final_message"])

    @patch("agent.applications.web_search.urllib.request.urlopen")
    def test_search_web_retries_on_transient_network_error_then_succeeds(self, mock_urlopen):
        from .applications.web_search import search_web

        class Config:
            web_search_api_key = "tvly-test-key"
            web_search_max_results = 5
            web_search_timeout_seconds = 10
            web_search_max_retries = 1

        response_body = json.dumps(
            {"results": [{"title": "Django 5.1 released", "url": "https://example.com", "content": "Release notes."}]}
        ).encode("utf-8")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return response_body

        mock_urlopen.side_effect = [OSError("temporary failure"), Response()]

        result = search_web(Config(), "django latest release")

        self.assertTrue(result.ok)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("agent.applications.web_search.urllib.request.urlopen")
    def test_search_web_does_not_retry_on_invalid_api_key(self, mock_urlopen):
        from urllib.error import HTTPError

        from .applications.web_search import search_web

        class Config:
            web_search_api_key = "tvly-invalid-key"
            web_search_max_results = 5
            web_search_timeout_seconds = 10
            web_search_max_retries = 2

        mock_urlopen.side_effect = HTTPError("https://api.tavily.com/search", 401, "Unauthorized", {}, None)

        result = search_web(Config(), "django latest release")

        self.assertFalse(result.ok)
        self.assertIn("拒否されました", result.message)
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("agent.applications.web_search.urllib.request.urlopen")
    def test_search_web_exhausts_retries_on_repeated_server_error(self, mock_urlopen):
        from urllib.error import HTTPError

        from .applications.web_search import search_web

        class Config:
            web_search_api_key = "tvly-test-key"
            web_search_max_results = 5
            web_search_timeout_seconds = 10
            web_search_max_retries = 2

        mock_urlopen.side_effect = HTTPError("https://api.tavily.com/search", 503, "Service Unavailable", {}, None)

        result = search_web(Config(), "django latest release")

        self.assertFalse(result.ok)
        self.assertIn("status=503", result.message)
        self.assertEqual(mock_urlopen.call_count, 3)


class LoggingRedactionTests(TestCase):
    def _filtered_message(self, message, *args):
        import logging

        from .logging import RedactSecretsFilter

        record = logging.LogRecord("agent", logging.DEBUG, __file__, 1, message, args, None)
        RedactSecretsFilter().filter(record)
        return record.getMessage()

    def test_redacts_openai_style_api_key(self):
        message = self._filtered_message("llm_start config=%s", "api_key=sk-abcdefghijklmnop")

        self.assertNotIn("sk-abcdefghijklmnop", message)
        self.assertIn("********", message)

    def test_redacts_aws_access_key(self):
        message = self._filtered_message("bedrock_auth key=%s", "AKIAABCDEFGHIJKLMNOP")

        self.assertNotIn("AKIAABCDEFGHIJKLMNOP", message)
        self.assertIn("********", message)

    def test_redacts_generic_key_value_secret(self):
        message = self._filtered_message("config_loaded values=%s", "api_key: hunter2secretvalue")

        self.assertNotIn("hunter2secretvalue", message)
        self.assertIn("********", message)

    def test_leaves_non_secret_messages_unchanged(self):
        message = self._filtered_message("agent_step_result thread_id=%s tool=%s", 1, "rag")

        self.assertEqual(message, "agent_step_result thread_id=1 tool=rag")
