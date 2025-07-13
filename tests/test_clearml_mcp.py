"""Behavioral tests for ClearML MCP server."""

from unittest.mock import Mock, patch

import pytest

from clearml_mcp import clearml_mcp


class TestClearMLConnection:
    """Test ClearML connection initialization behavior."""

    @patch("clearml_mcp.clearml_mcp.Task")
    def test_successful_connection_requires_accessible_projects(self, mock_task):
        """Connection succeeds when projects are accessible."""
        mock_task.get_projects.return_value = [Mock(name="project1")]

        # Should not raise
        clearml_mcp.initialize_clearml_connection()

    @patch("clearml_mcp.clearml_mcp.Task")
    def test_connection_fails_when_no_projects_accessible(self, mock_task):
        """Connection fails when no projects are accessible."""
        mock_task.get_projects.return_value = []

        with pytest.raises(RuntimeError, match="No ClearML projects accessible"):
            clearml_mcp.initialize_clearml_connection()

    @patch("clearml_mcp.clearml_mcp.Task")
    def test_connection_fails_on_api_error(self, mock_task):
        """Connection fails gracefully on API errors."""
        mock_task.get_projects.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Failed to initialize ClearML connection"):
            clearml_mcp.initialize_clearml_connection()


class TestTaskInfo:
    """Test task information retrieval behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_complete_task_information(self, mock_task):
        """get_task_info returns all expected task fields."""
        # Arrange: Create a realistic task mock
        task = Mock()
        task.id = "task_123"
        task.name = "Training Experiment"
        task.status = "completed"
        task.get_project_name.return_value = "ML Project"
        task.task_type = "training"
        task.comment = "Test experiment"

        task.data = Mock()
        task.data.created = "2024-01-01T00:00:00Z"
        task.data.last_update = "2024-01-02T00:00:00Z"
        task.data.tags = ["experiment", "v1"]

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.get_task_info.fn("task_123")

        # Assert: Verify complete information structure
        expected_fields = {
            "id",
            "name",
            "status",
            "project",
            "created",
            "last_update",
            "tags",
            "type",
            "comment",
        }
        assert set(result.keys()) == expected_fields
        assert result["id"] == "task_123"
        assert result["name"] == "Training Experiment"
        assert result["status"] == "completed"
        assert result["project"] == "ML Project"
        assert result["tags"] == ["experiment", "v1"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_task_without_optional_fields(self, mock_task):
        """get_task_info handles tasks missing optional fields."""
        task = Mock()
        task.id = "task_456"
        task.name = "Minimal Task"
        task.status = "running"
        task.get_project_name.return_value = "Test Project"
        task.task_type = "inference"

        # Properly mock missing comment attribute
        def mock_hasattr(obj, attr):
            if attr == "comment":
                return False
            return True

        task.data = Mock()
        task.data.created = "2024-01-01T00:00:00Z"
        task.data.last_update = "2024-01-01T00:00:00Z"
        task.data.tags = None  # No tags

        mock_task.get_task.return_value = task

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = await clearml_mcp.get_task_info.fn("task_456")

        assert result["tags"] == []
        assert result["comment"] is None

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_error_for_invalid_task_id(self, mock_task):
        """get_task_info returns error for non-existent tasks."""
        mock_task.get_task.side_effect = Exception("Task not found")

        result = await clearml_mcp.get_task_info.fn("invalid_id")

        assert "error" in result
        assert "Failed to get task info" in result["error"]


class TestTaskListing:
    """Test task listing behavior with various filters."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_lists_all_tasks_without_filters(self, mock_task):
        """list_tasks returns all tasks when no filters applied."""
        # Create multiple task mocks
        tasks = []
        for i in range(3):
            task = Mock()
            task.id = f"task_{i}"
            task.name = f"Task {i}"
            task.status = "completed" if i % 2 == 0 else "running"
            task.project = "Test Project"
            task.created = f"2024-01-0{i + 1}T00:00:00Z"
            task.tags = [f"tag_{i}"]
            tasks.append(task)

        mock_task.query_tasks.return_value = tasks

        result = await clearml_mcp.list_tasks.fn()

        assert len(result) == 3
        assert all("id" in task for task in result)
        assert all("name" in task for task in result)
        assert result[0]["id"] == "task_0"
        assert result[1]["status"] == "running"  # Task 1 (i=1) should be running

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_filters_tasks_by_project_and_status(self, mock_task):
        """list_tasks applies project and status filters correctly."""
        task = Mock()
        task.id = "filtered_task"
        task.name = "Filtered Task"
        task.status = "completed"
        task.project = "Specific Project"
        task.created = "2024-01-01T00:00:00Z"
        task.tags = []

        mock_task.query_tasks.return_value = [task]

        result = await clearml_mcp.list_tasks.fn(
            project_name="Specific Project", status="completed"
        )

        # Verify filter was applied correctly
        mock_task.query_tasks.assert_called_with(
            project_name="Specific Project", task_filter={"status": ["completed"]}, tags=None
        )
        assert len(result) == 1
        assert result[0]["id"] == "filtered_task"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_empty_task_list(self, mock_task):
        """list_tasks handles empty results gracefully."""
        mock_task.query_tasks.return_value = []

        result = await clearml_mcp.list_tasks.fn()

        assert result == []

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_error_on_query_failure(self, mock_task):
        """list_tasks returns error when query fails."""
        mock_task.query_tasks.side_effect = Exception("Query failed")

        result = await clearml_mcp.list_tasks.fn()

        assert len(result) == 1
        assert "error" in result[0]


class TestTaskParameters:
    """Test task parameter retrieval behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_task_parameters_structure(self, mock_task):
        """get_task_parameters returns the parameter dictionary."""
        task = Mock()
        expected_params = {
            "General": {"learning_rate": 0.001, "batch_size": 32, "epochs": 100},
            "Model": {"hidden_layers": 3, "dropout": 0.2},
        }
        task.get_parameters_as_dict.return_value = expected_params
        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_parameters.fn("test_task")

        assert result == expected_params
        assert "General" in result
        assert "Model" in result
        assert result["General"]["learning_rate"] == 0.001

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_task_without_parameters(self, mock_task):
        """get_task_parameters handles tasks with no parameters."""
        task = Mock()
        task.get_parameters_as_dict.return_value = {}
        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_parameters.fn("test_task")

        assert result == {}

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_error_on_parameter_retrieval_failure(self, mock_task):
        """get_task_parameters returns error when parameter retrieval fails."""
        mock_task.get_task.side_effect = Exception("Parameter access failed")

        result = await clearml_mcp.get_task_parameters.fn("invalid_task")

        assert "error" in result
        assert "Failed to get task parameters" in result["error"]


class TestTaskMetrics:
    """Test task metrics retrieval behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_processes_metrics_with_complete_data(self, mock_task):
        """get_task_metrics processes scalar metrics correctly."""
        task = Mock()
        scalars_data = {
            "loss": {
                "train": {"x": [1, 2, 3, 4], "y": [1.0, 0.8, 0.6, 0.4]},
                "validation": {"x": [1, 2, 3, 4], "y": [1.2, 1.0, 0.8, 0.7]},
            },
            "accuracy": {"train": {"x": [1, 2, 3, 4], "y": [0.6, 0.7, 0.8, 0.9]}},
        }
        task.get_reported_scalars.return_value = scalars_data
        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_metrics.fn("test_task")

        # Verify metric structure and calculations
        assert "loss" in result
        assert "accuracy" in result
        assert "train" in result["loss"]
        assert "validation" in result["loss"]

        # Verify calculations for train loss
        train_loss = result["loss"]["train"]
        assert train_loss["last_value"] == 0.4
        assert train_loss["min_value"] == 0.4
        assert train_loss["max_value"] == 1.0
        assert train_loss["iterations"] == 4

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_metrics_with_empty_data(self, mock_task):
        """get_task_metrics handles metrics with empty or missing data."""
        task = Mock()
        scalars_data = {
            "loss": {
                "train": {"x": [], "y": []},  # Empty data
                "validation": {"x": [1, 2]},  # Missing y data entirely
            },
            "accuracy": {
                "train": {}  # Missing data entirely
            },
        }
        task.get_reported_scalars.return_value = scalars_data
        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_metrics.fn("test_task")

        # Should handle gracefully without crashing
        assert isinstance(result, dict)
        # Should not crash and should skip metrics without proper data

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_task_without_metrics(self, mock_task):
        """get_task_metrics handles tasks with no metrics."""
        task = Mock()
        task.get_reported_scalars.return_value = {}
        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_metrics.fn("test_task")

        assert result == {}


class TestTaskArtifacts:
    """Test task artifact retrieval behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_artifact_information(self, mock_task):
        """get_task_artifacts returns complete artifact information."""
        task = Mock()

        # Mock artifacts
        artifact1 = Mock()
        artifact1.type = "model"
        artifact1.mode = "output"
        artifact1.uri = "s3://bucket/model.pkl"
        artifact1.content_type = "application/octet-stream"
        artifact1.timestamp = "2024-01-01T00:00:00Z"

        artifact2 = Mock()
        artifact2.type = "data"
        artifact2.mode = "input"
        artifact2.uri = "file://data/train.csv"
        artifact2.content_type = "text/csv"

        # Properly mock missing timestamp attribute
        def mock_hasattr(obj, attr):
            if obj is artifact2 and attr == "timestamp":
                return False
            return True

        task.artifacts = {"model": artifact1, "dataset": artifact2}
        mock_task.get_task.return_value = task

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = await clearml_mcp.get_task_artifacts.fn("test_task")

        assert "model" in result
        assert "dataset" in result

        # Verify model artifact
        model_artifact = result["model"]
        assert model_artifact["type"] == "model"
        assert model_artifact["uri"] == "s3://bucket/model.pkl"
        assert model_artifact["timestamp"] == "2024-01-01T00:00:00Z"

        # Verify dataset artifact (without timestamp)
        dataset_artifact = result["dataset"]
        assert dataset_artifact["type"] == "data"
        assert dataset_artifact["timestamp"] is None


class TestProjectOperations:
    """Test project-related operations behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_lists_available_projects(self, mock_task):
        """list_projects returns available project information."""
        projects = []
        for i in range(3):
            proj = Mock()
            proj.id = f"proj_{i}"
            proj.name = f"Project {i}"
            projects.append(proj)

        mock_task.get_projects.return_value = projects

        result = await clearml_mcp.list_projects.fn()

        assert len(result) == 3
        assert all("id" in proj for proj in result)
        assert all("name" in proj for proj in result)
        assert result[0]["name"] == "Project 0"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_projects_without_id_attribute(self, mock_task):
        """list_projects handles projects missing id attribute."""
        proj = Mock()
        proj.name = "Simple Project"

        # Properly mock missing id attribute
        def mock_hasattr(obj, attr):
            if obj is proj and attr == "id":
                return False
            return True

        mock_task.get_projects.return_value = [proj]

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = await clearml_mcp.list_projects.fn()

        assert len(result) == 1
        assert result[0]["id"] is None
        assert result[0]["name"] == "Simple Project"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_calculates_project_statistics(self, mock_task):
        """get_project_stats calculates accurate project statistics."""
        # Create tasks with different statuses and types
        tasks = []
        statuses = ["completed", "running", "completed", "failed", "running"]
        types = ["training", "inference", "training", "training", "inference"]

        for i, (status, task_type) in enumerate(zip(statuses, types, strict=False)):
            task = Mock()
            task.status = status
            task.type = task_type
            tasks.append(task)

        mock_task.query_tasks.return_value = tasks

        result = await clearml_mcp.get_project_stats.fn("Test Project")

        # Verify statistics calculation
        assert result["project_name"] == "Test Project"
        assert result["total_tasks"] == 5

        # Verify status breakdown
        status_breakdown = result["status_breakdown"]
        assert status_breakdown["completed"] == 2
        assert status_breakdown["running"] == 2
        assert status_breakdown["failed"] == 1

        # Verify task types
        task_types = result["task_types"]
        assert set(task_types) == {"training", "inference"}


class TestTaskComparison:
    """Test task comparison behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_compares_multiple_tasks_with_specific_metrics(self, mock_task):
        """compare_tasks compares specified metrics across tasks."""

        def mock_get_task(task_id):
            task = Mock()
            task.name = f"Task {task_id}"
            task.status = "completed"

            if task_id == "task1":
                task.get_reported_scalars.return_value = {
                    "loss": {
                        "train": {"x": [1, 2], "y": [1.0, 0.5]},
                    },
                    "accuracy": {
                        "train": {"x": [1, 2], "y": [0.8, 0.9]},
                    },
                }
            else:  # task2
                task.get_reported_scalars.return_value = {
                    "loss": {
                        "train": {"x": [1, 2], "y": [1.2, 0.7]},
                    }
                }
            return task

        mock_task.get_task.side_effect = mock_get_task

        result = await clearml_mcp.compare_tasks.fn(["task1", "task2"], metrics=["loss"])

        # Verify comparison structure
        assert "task1" in result
        assert "task2" in result

        # Verify task1 comparison
        task1_data = result["task1"]
        assert task1_data["name"] == "Task task1"
        assert "loss" in task1_data["metrics"]
        assert task1_data["metrics"]["loss"]["train"]["last_value"] == 0.5

        # Verify task2 comparison
        task2_data = result["task2"]
        assert task2_data["metrics"]["loss"]["train"]["last_value"] == 0.7

        # Verify accuracy not included (not requested)
        assert "accuracy" not in task1_data["metrics"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_compares_all_metrics_when_none_specified(self, mock_task):
        """compare_tasks includes all metrics when none specified."""
        task = Mock()
        task.name = "Full Task"
        task.status = "completed"
        task.get_reported_scalars.return_value = {
            "loss": {"train": {"x": [1], "y": [0.5]}},
            "accuracy": {"train": {"x": [1], "y": [0.9]}},
            "f1_score": {"train": {"x": [1], "y": [0.85]}},
        }
        mock_task.get_task.return_value = task

        result = await clearml_mcp.compare_tasks.fn(["task1"])

        task_metrics = result["task1"]["metrics"]
        assert "loss" in task_metrics
        assert "accuracy" in task_metrics
        assert "f1_score" in task_metrics


class TestTaskSearch:
    """Test task search behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_searches_by_task_name(self, mock_task):
        """search_tasks finds tasks by name matching."""
        tasks = []
        names = ["Neural Network Training", "Data Processing", "Model Inference"]

        for i, name in enumerate(names):
            task = Mock()
            task.id = f"task_{i}"
            task.name = name
            task.status = "completed"
            task.project = "Test Project"
            task.created = "2024-01-01T00:00:00Z"
            task.tags = []
            task.comment = None
            tasks.append(task)

        mock_task.query_tasks.return_value = tasks

        result = await clearml_mcp.search_tasks.fn("neural")

        assert len(result) == 1
        assert result[0]["name"] == "Neural Network Training"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_searches_by_tags_and_comments(self, mock_task):
        """search_tasks finds tasks by tags and comments."""
        task1 = Mock()
        task1.id = "task1"
        task1.name = "Task 1"
        task1.comment = "Experiment with new architecture"
        task1.tags = ["deep-learning", "cnn"]
        task1.status = "completed"
        task1.project = "Project"
        task1.created = "2024-01-01T00:00:00Z"

        task2 = Mock()
        task2.id = "task2"
        task2.name = "Task 2"
        task2.comment = "Basic training run"
        task2.tags = ["baseline"]
        task2.status = "completed"
        task2.project = "Project"
        task2.created = "2024-01-01T00:00:00Z"

        mock_task.query_tasks.return_value = [task1, task2]

        # Search by tag
        result = await clearml_mcp.search_tasks.fn("cnn")
        assert len(result) == 1
        assert result[0]["id"] == "task1"

        # Search by comment
        mock_task.query_tasks.return_value = [task1, task2]
        result = await clearml_mcp.search_tasks.fn("architecture")
        assert len(result) == 1
        assert result[0]["id"] == "task1"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_empty_list_when_no_matches(self, mock_task):
        """search_tasks returns empty list when no matches found."""
        task = Mock()
        task.name = "Unrelated Task"
        task.comment = None
        task.tags = []

        mock_task.query_tasks.return_value = [task]

        result = await clearml_mcp.search_tasks.fn("nonexistent")

        assert result == []


class TestMainEntryPoint:
    """Test main entry point behavior."""

    @patch("clearml_mcp.clearml_mcp.mcp")
    @patch("clearml_mcp.clearml_mcp.initialize_clearml_connection")
    def test_main_initializes_connection_and_runs_mcp(self, mock_init, mock_mcp):
        """main() initializes connection and runs MCP server."""
        clearml_mcp.main()

        mock_init.assert_called_once()
        mock_mcp.run.assert_called_once_with(transport="stdio")
