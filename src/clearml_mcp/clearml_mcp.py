"""ClearML MCP Server implementation."""

import re
from pathlib import Path
from typing import Any, cast

from clearml import Model, Task
from fastmcp import FastMCP

mcp = FastMCP("clearml-mcp")


def initialize_clearml_connection() -> None:
    """Initialize and validate ClearML connection."""
    try:
        projects = Task.get_projects()
        if not projects:
            raise ValueError("No ClearML projects accessible - check your clearml.conf")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ClearML connection: {e!s}")


# Fields fetched in the single bulk ``query_tasks`` call below. Requesting these
# via ``additional_return_fields`` makes ClearML hydrate them server-side in one
# paginated call (``tasks.get_all_ex`` with ``only_fields``), avoiding a per-task
# ``Task.get_task`` round-trip (the N+1 problem fixed here, see issue #6).
_TASK_FIELDS = ("id", "name", "status", "type", "comment", "created", "project", "tags")


def _literal_ci_regex(query: str) -> str:
    """Build a case-insensitive regex that matches ``query`` as a literal substring.

    ClearML matches ``task_name`` and ``_any_`` patterns as regular expressions
    server-side, so user input must be escaped to match literally (e.g. ``a.b``
    matches the literal string, ``[`` does not error the query) and prefixed with
    ``(?i)`` to preserve the case-insensitive behaviour callers expect.
    """
    return f"(?i){re.escape(query)}"


def _project_id_to_name(task_dicts: list[dict[str, Any]]) -> dict[str, str]:
    """Build a project-id -> project-name map for the projects referenced by tasks.

    ``query_tasks`` returns a project *id* per task. We resolve names in a single
    ``get_projects`` call rather than one lookup per task.
    """
    project_ids = {d.get("project") for d in task_dicts if d.get("project")}
    if not project_ids:
        return {}
    return {
        proj.id: proj.name
        for proj in Task.get_projects()
        if getattr(proj, "id", None) in project_ids
    }


def _query_task_dicts(
    project_name: str | None = None,
    task_name: str | None = None,
    tags: list[str] | None = None,
    status: str | None = None,
    any_pattern: str | None = None,
    any_fields: tuple[str, ...] = ("name", "comment", "tags"),
) -> list[dict[str, Any]]:
    """Fetch fully hydrated task records in a single bulk backend call.

    ``Task.query_tasks`` returns bare ID strings by default; with
    ``additional_return_fields`` it returns a list of dicts with the requested
    fields fetched in one bulk call. ``task_name``, ``status`` and ``any_pattern``
    are all matched server-side so we never download every task just to filter
    client-side. ``any_pattern`` is a regex matched against any of ``any_fields``.
    """
    task_filter: dict[str, Any] = {}
    if status:
        task_filter["status"] = [status]
    if any_pattern:
        task_filter["_any_"] = {"fields": list(any_fields), "pattern": any_pattern}
    # ``additional_return_fields`` guarantees a list of dicts (one per task).
    raw = cast(
        "list[dict[str, Any]]",
        Task.query_tasks(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            additional_return_fields=list(_TASK_FIELDS),
            task_filter=task_filter or None,
        ),
    )
    project_names = _project_id_to_name(raw)
    return [
        {
            "id": d.get("id"),
            "name": d.get("name"),
            "status": d.get("status"),
            "type": d.get("type"),
            "comment": d.get("comment") or "",
            "created": str(d.get("created")) if d.get("created") is not None else None,
            "project": project_names.get(d.get("project"), d.get("project")),
            "tags": list(d.get("tags") or []),
        }
        for d in raw
    ]


@mcp.tool()
async def get_task_info(task_id: str) -> dict[str, Any]:
    """Get ClearML task details, parameters, and status."""
    try:
        task = Task.get_task(task_id=task_id)
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status,
            "project": task.get_project_name(),
            "created": str(task.data.created),
            "last_update": str(task.data.last_update),
            "tags": list(task.data.tags) if task.data.tags else [],
            "type": task.task_type,
            "comment": task.comment if hasattr(task, "comment") else None,
        }
    except Exception as e:
        return {"error": f"Failed to get task info: {e!s}"}


@mcp.tool()
async def list_tasks(
    project_name: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """List ClearML tasks with filters."""
    try:
        tasks = _query_task_dicts(project_name=project_name, status=status, tags=tags)
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "status": t["status"],
                "project": t["project"],
                "created": t["created"],
                "tags": t["tags"],
            }
            for t in tasks
        ]
    except Exception as e:
        return [{"error": f"Failed to list tasks: {e!s}"}]


@mcp.tool()
async def get_task_parameters(task_id: str) -> dict[str, Any]:
    """Get task hyperparameters and configuration."""
    try:
        task = Task.get_task(task_id=task_id)
        return task.get_parameters_as_dict()
    except Exception as e:
        return {"error": f"Failed to get task parameters: {e!s}"}


@mcp.tool()
async def get_task_metrics(task_id: str) -> dict[str, Any]:
    """Get task training metrics and scalars."""
    try:
        task = Task.get_task(task_id=task_id)
        scalars = task.get_reported_scalars()

        metrics = {}
        for metric, variants in scalars.items():
            metrics[metric] = {}
            for variant, data in variants.items():
                if data and "y" in data:
                    metrics[metric][variant] = {
                        "last_value": data["y"][-1] if data["y"] else None,
                        "min_value": min(data["y"]) if data["y"] else None,
                        "max_value": max(data["y"]) if data["y"] else None,
                        "iterations": len(data["y"]),
                    }
        return metrics
    except Exception as e:
        return {"error": f"Failed to get task metrics: {e!s}"}


@mcp.tool()
async def get_task_script(task_id: str, output_path: str | None = None) -> dict[str, Any]:
    """Get the repo, branch and commit a task ran from, plus its uncommitted diff.

    The diff is the content of the ClearML UI's "UNCOMMITTED CHANGES" panel, and it
    is what makes a run reproducible: repository and commit alone do not describe
    the working tree the task actually executed. Diffs routinely run to tens of KB,
    so pass ``output_path`` to write the diff to that file and get back its size
    instead of the inline text.
    """
    try:
        task = Task.get_task(task_id=task_id)
        script = task.data.script
        diff = script.diff or ""
        result: dict[str, Any] = {
            "repository": script.repository,
            "branch": script.branch,
            "commit": script.version_num,
            "entry_point": script.entry_point,
            "working_dir": script.working_dir,
        }
        if output_path:
            Path(output_path).write_text(diff, encoding="utf-8")
            result["output_path"] = output_path
            result["diff_size_bytes"] = len(diff)
        else:
            result["diff"] = diff
        return result
    except Exception as e:
        return {"error": f"Failed to get task script: {e!s}"}


@mcp.tool()
async def get_task_console_logs(task_id: str, number_of_reports: int = 100) -> dict[str, Any]:
    """Get the most recent console log lines reported by a task."""
    try:
        task = Task.get_task(task_id=task_id)
        logs = task.get_reported_console_output(number_of_reports=number_of_reports)
        return {"logs": logs, "count": len(logs)}
    except Exception as e:
        return {"error": f"Failed to get console logs: {e!s}"}


@mcp.tool()
async def get_task_artifacts(task_id: str) -> dict[str, Any]:
    """Get task artifacts and outputs."""
    try:
        task = Task.get_task(task_id=task_id)
        artifacts = task.artifacts

        artifact_dict = {}
        for key, artifact in artifacts.items():
            artifact_dict[key] = {
                "type": artifact.type,
                "mode": artifact.mode,
                "uri": artifact.uri,
                "content_type": artifact.content_type,
                "timestamp": str(artifact.timestamp) if hasattr(artifact, "timestamp") else None,
            }
        return artifact_dict
    except Exception as e:
        return {"error": f"Failed to get task artifacts: {e!s}"}


@mcp.tool()
async def get_model_info(task_id: str) -> dict[str, Any]:
    """Get model metadata and configuration."""
    try:
        task = Task.get_task(task_id=task_id)
        models = task.models

        model_info = {"input": [], "output": []}

        if models.get("input"):
            for model in models["input"]:
                model_info["input"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                    },
                )

        if models.get("output"):
            for model in models["output"]:
                model_info["output"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                    },
                )

        return model_info
    except Exception as e:
        return {"error": f"Failed to get model info: {e!s}"}


@mcp.tool()
async def list_models(project_name: str | None = None) -> list[dict[str, Any]]:
    """List available models with filtering."""
    try:
        models = Model.query_models(project_name=project_name)
        return [
            {
                "id": model.id,
                "name": model.name,
                "project": model.project,
                "framework": model.framework,
                "created": str(model.created),
                "tags": list(model.tags) if model.tags else [],
                "task_id": model.task,
            }
            for model in models
        ]
    except Exception as e:
        return [{"error": f"Failed to list models: {e!s}"}]


@mcp.tool()
async def get_model_artifacts(task_id: str) -> dict[str, Any]:
    """Get model files and download URLs."""
    try:
        task = Task.get_task(task_id=task_id)
        models = task.models

        artifacts = {"input_models": [], "output_models": []}

        if models.get("input"):
            for model in models["input"]:
                artifacts["input_models"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                        "uri": model.uri,
                    },
                )

        if models.get("output"):
            for model in models["output"]:
                artifacts["output_models"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                        "uri": model.uri,
                    },
                )

        return artifacts
    except Exception as e:
        return {"error": f"Failed to get model artifacts: {e!s}"}


@mcp.tool()
async def find_project_by_pattern(pattern: str) -> list[dict[str, Any]]:
    """Find ClearML projects by name pattern (case-insensitive)."""
    try:
        all_projects = Task.get_projects()
        matching_projects = []

        pattern_lower = pattern.lower()
        for proj in all_projects:
            if pattern_lower in proj.name.lower():
                matching_projects.append(
                    {
                        "id": getattr(proj, "id", None),
                        "name": proj.name,
                    }
                )

        return matching_projects
    except Exception as e:
        return [{"error": f"Failed to find projects by pattern: {e!s}"}]


@mcp.tool()
async def find_experiment_in_project(
    project_name: str, experiment_pattern: str
) -> list[dict[str, Any]]:
    """Find experiments in a specific project by name pattern."""
    try:
        # Match the pattern as a literal, case-insensitive substring server-side
        # (ClearML treats task_name as a regex), so only matching tasks come back.
        tasks = _query_task_dicts(
            project_name=project_name, task_name=_literal_ci_regex(experiment_pattern)
        )
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "status": t["status"],
                "project": t["project"],
                "created": t["created"],
            }
            for t in tasks
        ]
    except Exception as e:
        return [{"error": f"Failed to find experiments: {e!s}"}]


@mcp.tool()
async def list_projects() -> list[dict[str, Any]]:
    """List available ClearML projects."""
    try:
        projects = Task.get_projects()
        return [
            {
                "id": proj.id if hasattr(proj, "id") else None,
                "name": proj.name,
            }
            for proj in projects
        ]
    except Exception as e:
        return [{"error": f"Failed to list projects: {e!s}"}]


@mcp.tool()
async def get_project_stats(project_name: str) -> dict[str, Any]:
    """Get project statistics and task counts."""
    try:
        tasks = _query_task_dicts(project_name=project_name)

        status_counts: dict[str, int] = {}
        for task in tasks:
            status = task["status"]
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "project_name": project_name,
            "total_tasks": len(tasks),
            "status_breakdown": status_counts,
            "task_types": sorted({task["type"] for task in tasks if task["type"]}),
        }
    except Exception as e:
        return {"error": f"Failed to get project stats: {e!s}"}


@mcp.tool()
async def compare_tasks(task_ids: list[str], metrics: list[str] | None = None) -> dict[str, Any]:
    """Compare multiple tasks by metrics."""
    try:
        comparison = {}

        for task_id in task_ids:
            task = Task.get_task(task_id=task_id)
            scalars = task.get_reported_scalars()

            task_metrics = {"name": task.name, "status": task.status, "metrics": {}}

            if metrics:
                for metric in metrics:
                    if metric in scalars:
                        task_metrics["metrics"][metric] = {}
                        for variant, data in scalars[metric].items():
                            if data and "y" in data and data["y"]:
                                task_metrics["metrics"][metric][variant] = {
                                    "last_value": data["y"][-1],
                                    "min_value": min(data["y"]),
                                    "max_value": max(data["y"]),
                                }
            else:
                for metric, variants in scalars.items():
                    task_metrics["metrics"][metric] = {}
                    for variant, data in variants.items():
                        if data and "y" in data and data["y"]:
                            task_metrics["metrics"][metric][variant] = {
                                "last_value": data["y"][-1],
                                "min_value": min(data["y"]),
                                "max_value": max(data["y"]),
                            }

            comparison[task_id] = task_metrics

        return comparison
    except Exception as e:
        return {"error": f"Failed to compare tasks: {e!s}"}


@mcp.tool()
async def search_tasks(query: str, project_name: str | None = None) -> list[dict[str, Any]]:
    """Search tasks by name, tags, or description."""
    try:
        # Match the query as a literal substring (case-insensitive) against
        # name/comment/tags server-side, so we never hydrate non-matching tasks.
        tasks = _query_task_dicts(project_name=project_name, any_pattern=_literal_ci_regex(query))
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "status": t["status"],
                "project": t["project"],
                "created": t["created"],
                "tags": t["tags"],
                "comment": t["comment"],
            }
            for t in tasks
        ]
    except Exception as e:
        return [{"error": f"Failed to search tasks: {e!s}"}]


def main() -> None:
    """Entry point for uvx clearml-mcp."""
    initialize_clearml_connection()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
