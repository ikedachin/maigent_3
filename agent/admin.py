from django.contrib import admin

from .models import (
    AgentRun,
    AgentTaskRecord,
    AgentWorkerRun,
    AppSetting,
    ApprovalRequest,
    Automation,
    FeatureFlag,
    Message,
    Project,
    ProjectAccessPath,
    Thread,
)


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "path", "output_path", "is_current", "updated_at")
    list_filter = ("is_current",)
    search_fields = ("name", "path", "output_path", "description")
    ordering = ("name",)


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ("title", "project", "memory_enabled", "updated_at")
    list_filter = ("memory_enabled", "project")
    search_fields = ("title", "summary", "project__name")
    list_select_related = ("project",)
    ordering = ("-updated_at",)


@admin.register(ProjectAccessPath)
class ProjectAccessPathAdmin(admin.ModelAdmin):
    list_display = ("project", "mode", "path", "note", "created_at")
    list_filter = ("mode", "project")
    search_fields = ("path", "note", "project__name")
    list_select_related = ("project",)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "role", "status", "content_preview", "created_at")
    list_filter = ("role", "status", "created_at")
    search_fields = ("content", "openai_response_id", "thread__title", "thread__project__name")
    list_select_related = ("thread", "thread__project")
    ordering = ("-created_at",)

    @admin.display(description="Content")
    def content_preview(self, obj):
        compact = " ".join(obj.content.split())
        return compact[:80]


@admin.register(AppSetting)
class AppSettingAdmin(admin.ModelAdmin):
    list_display = ("key", "value_preview", "updated_at")
    search_fields = ("key", "value")
    ordering = ("key",)

    @admin.display(description="Value")
    def value_preview(self, obj):
        compact = " ".join(obj.value.split())
        return compact[:100]


@admin.register(FeatureFlag)
class FeatureFlagAdmin(admin.ModelAdmin):
    list_display = ("name", "enabled", "description", "updated_at")
    list_filter = ("enabled",)
    search_fields = ("name", "description")
    list_editable = ("enabled",)
    ordering = ("name",)


@admin.register(Automation)
class AutomationAdmin(admin.ModelAdmin):
    list_display = ("name", "schedule", "status", "updated_at")
    list_filter = ("status",)
    search_fields = ("name", "schedule", "description")
    ordering = ("name",)


@admin.register(ApprovalRequest)
class ApprovalRequestAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "status", "command_preview", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("command", "rationale", "thread__title", "thread__project__name")
    list_select_related = ("thread", "thread__project")
    ordering = ("-created_at",)

    @admin.display(description="Command")
    def command_preview(self, obj):
        compact = " ".join(obj.command.split())
        return compact[:80]


@admin.register(AgentRun)
class AgentRunAdmin(admin.ModelAdmin):
    list_display = ("id", "thread", "attempt", "status", "goal_preview", "created_at", "updated_at")
    list_filter = ("status", "attempt", "created_at")
    search_fields = ("goal", "initial_plan_summary", "final_message", "error", "thread__title")
    list_select_related = ("thread", "user_message", "assistant_message")
    ordering = ("-created_at",)

    @admin.display(description="Goal")
    def goal_preview(self, obj):
        compact = " ".join(obj.goal.split())
        return compact[:100]


@admin.register(AgentWorkerRun)
class AgentWorkerRunAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "name", "role", "status", "started_at", "finished_at")
    list_filter = ("status", "role")
    search_fields = ("name", "role", "purpose", "result", "error", "run__goal")
    list_select_related = ("run",)
    ordering = ("-created_at",)


@admin.register(AgentTaskRecord)
class AgentTaskRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "sequence", "tool", "status", "worker", "created_at")
    list_filter = ("status", "tool", "created_at")
    search_fields = ("purpose", "input_before", "input_after", "result", "error", "run__goal", "worker__name")
    list_select_related = ("run", "worker")
    ordering = ("-created_at", "sequence")
