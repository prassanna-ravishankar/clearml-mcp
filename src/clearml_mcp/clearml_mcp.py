"""ClearML MCP Server implementation."""

from typing import Any

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
        # Task.query_tasks returns task IDs (strings), not task objects
        task_ids = Task.query_tasks(
            project_name=project_name,
            task_filter={"status": [status]} if status else None,
            tags=tags,
        )

        # Convert task IDs to full task objects
        tasks = []
        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)
                tasks.append(
                    {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status,
                        "project": task.get_project_name(),
                        "created": str(task.data.created),
                        "tags": list(task.data.tags) if task.data.tags else [],
                    }
                )
            except Exception as e:
                # If we can't get a specific task, include the error but continue
                tasks.append({"id": task_id, "error": f"Failed to get task details: {e!s}"})

        return tasks
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
        # Get task IDs for the project
        task_ids = Task.query_tasks(project_name=project_name)

        matching_experiments = []
        pattern_lower = experiment_pattern.lower()

        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)
                if pattern_lower in task.name.lower():
                    matching_experiments.append(
                        {
                            "id": task.id,
                            "name": task.name,
                            "status": task.status,
                            "project": task.get_project_name(),
                            "created": str(task.data.created),
                        }
                    )
            except Exception:
                # Skip tasks we can't access - could be permissions or API issues
                pass

        return matching_experiments
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
        tasks = Task.query_tasks(project_name=project_name)

        status_counts = {}
        for task in tasks:
            status = task.status
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "project_name": project_name,
            "total_tasks": len(tasks),
            "status_breakdown": status_counts,
            "task_types": list(set(task.type for task in tasks if hasattr(task, "type"))),
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
        # Task.query_tasks returns task IDs (strings), not task objects
        task_ids = Task.query_tasks(project_name=project_name)

        matching_tasks = []
        query_lower = query.lower()

        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)

                # Check if the task matches the search query
                task_name = task.name.lower()
                task_comment = getattr(task, "comment", "") or ""
                task_tags = list(task.data.tags) if task.data.tags else []

                if (
                    query_lower in task_name
                    or (task_comment and query_lower in task_comment.lower())
                    or any(query_lower in tag.lower() for tag in task_tags)
                ):
                    matching_tasks.append(
                        {
                            "id": task.id,
                            "name": task.name,
                            "status": task.status,
                            "project": task.get_project_name(),
                            "created": str(task.data.created),
                            "tags": task_tags,
                            "comment": task_comment,
                        }
                    )
            except Exception as e:
                # If we can't get a specific task, skip it but log the error
                matching_tasks.append(
                    {"id": task_id, "error": f"Failed to get task details: {e!s}"}
                )

        return matching_tasks
    except Exception as e:
        return [{"error": f"Failed to search tasks: {e!s}"}]


def main() -> None:
    """Entry point for uvx clearml-mcp."""
    initialize_clearml_connection()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
