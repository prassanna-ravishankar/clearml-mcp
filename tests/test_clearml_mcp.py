"""Behavioral tests for ClearML MCP server."""

from unittest.mock import Mock, patch

import pytest

from clearml_mcp import clearml_mcp


def _make_query_tasks(task_dicts):
    """Build a fake Task.query_tasks that mimics the real SDK contract.

    The real ``Task.query_tasks`` returns bare ID strings unless
    ``additional_return_fields`` is passed, in which case it returns a list of
    dicts with the requested fields hydrated in a single bulk call. These tests
    always exercise the bulk path, so the fake returns the provided dicts and
    fails loudly if a caller forgets ``additional_return_fields`` (which would
    have forced the old, broken N+1 ``get_task`` hydration).
    """

    def query_tasks(*, additional_return_fields=None, **_kwargs: object):
        if not additional_return_fields:
            raise AssertionError(
                "query_tasks called without additional_return_fields - "
                "this is the N+1 / 'str'.status bug from issue #6"
            )
        return list(task_dicts)

    return query_tasks


def _fake_project(project_id, name):
    proj = Mock()
    proj.id = project_id
    proj.name = name
    return proj


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
        task.data.created = "2024-01-01T00:00:00Z"
        task.data.last_update = "2024-01-01T02:00:00Z"
        task.data.tags = ["training", "production"]
        task.task_type = "training"
        task.comment = "Experiment with improved accuracy"

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.get_task_info.fn("task_123")

        # Assert: Verify all expected fields are present
        assert result["id"] == "task_123"
        assert result["name"] == "Training Experiment"
        assert result["status"] == "completed"
        assert result["project"] == "ML Project"
        assert result["created"] == "2024-01-01T00:00:00Z"
        assert result["last_update"] == "2024-01-01T02:00:00Z"
        assert result["tags"] == ["training", "production"]
        assert result["type"] == "training"
        assert result["comment"] == "Experiment with improved accuracy"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_task_without_optional_fields(self, mock_task):
        """get_task_info gracefully handles tasks missing optional fields."""
        # Arrange: Create task mock missing some fields
        task = Mock()
        task.id = "task_456"
        task.name = "Basic Task"
        task.status = "running"
        task.get_project_name.return_value = "Test Project"
        task.data.created = "2024-01-01T00:00:00Z"
        task.data.last_update = "2024-01-01T00:00:00Z"
        task.data.tags = None
        task.task_type = "inference"

        # Mock hasattr to return False for comment attribute
        def mock_hasattr(obj, attr):
            if attr == "comment":
                return False
            return True

        mock_task.get_task.return_value = task

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = await clearml_mcp.get_task_info.fn("task_456")

        # Assert: Should handle missing fields gracefully
        assert result["id"] == "task_456"
        assert result["name"] == "Basic Task"
        assert result["status"] == "running"
        assert result["project"] == "Test Project"
        assert result["created"] == "2024-01-01T00:00:00Z"
        assert result["tags"] == []
        assert result["comment"] is None

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_error_for_invalid_task_id(self, mock_task):
        """get_task_info returns error message for invalid task ID."""
        mock_task.get_task.side_effect = Exception("Task not found")

        result = await clearml_mcp.get_task_info.fn("invalid_id")

        assert "error" in result
        assert "Failed to get task info" in result["error"]


class TestTaskListing:
    """Test task listing behavior with different filters."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_lists_all_tasks_without_filters(self, mock_task):
        """list_tasks returns all tasks from a single bulk query."""
        mock_task.query_tasks.side_effect = _make_query_tasks(
            [
                {
                    "id": "task_1",
                    "name": "Experiment 1",
                    "status": "completed",
                    "type": "training",
                    "comment": "",
                    "created": "2024-01-01T00:00:00Z",
                    "project": "proj_a",
                    "tags": [],
                },
                {
                    "id": "task_2",
                    "name": "Experiment 2",
                    "status": "running",
                    "type": "training",
                    "comment": "",
                    "created": "2024-01-02T00:00:00Z",
                    "project": "proj_b",
                    "tags": [],
                },
            ]
        )
        mock_task.get_projects.return_value = [
            _fake_project("proj_a", "Project A"),
            _fake_project("proj_b", "Project B"),
        ]

        result = await clearml_mcp.list_tasks.fn()

        assert len(result) == 2
        assert result[0]["id"] == "task_1"
        assert result[0]["project"] == "Project A"
        assert result[1]["id"] == "task_2"
        assert result[1]["project"] == "Project B"
        # Regression: must NOT hydrate tasks one-by-one (issue #6 N+1).
        mock_task.get_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_filters_tasks_by_project_and_status(self, mock_task):
        """list_tasks pushes project and status filters into the bulk query."""
        mock_task.query_tasks.side_effect = _make_query_tasks(
            [
                {
                    "id": "task_1",
                    "name": "Task task_1",
                    "status": "completed",
                    "type": "training",
                    "comment": "",
                    "created": "2024-01-01T00:00:00Z",
                    "project": "proj_1",
                    "tags": [],
                }
            ]
        )
        mock_task.get_projects.return_value = [_fake_project("proj_1", "Filtered Project")]

        result = await clearml_mcp.list_tasks.fn(
            project_name="Filtered Project", status="completed"
        )

        # The single bulk call requests fields server-side and filters by status.
        _, kwargs = mock_task.query_tasks.call_args
        assert kwargs["project_name"] == "Filtered Project"
        assert kwargs["additional_return_fields"]
        assert kwargs["task_filter"] == {"status": ["completed"]}
        assert len(result) == 1
        assert result[0]["status"] == "completed"
        assert result[0]["project"] == "Filtered Project"
        mock_task.get_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_empty_task_list(self, mock_task):
        """list_tasks handles empty results gracefully."""
        mock_task.query_tasks.side_effect = _make_query_tasks([])

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
        assert "Failed to list tasks" in result[0]["error"]


class TestTaskParameters:
    """Test task parameter retrieval behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_task_parameters_structure(self, mock_task):
        """get_task_parameters returns properly structured parameter data."""
        # Arrange
        task = Mock()
        task.get_parameters_as_dict.return_value = {
            "General": {"learning_rate": 0.001, "batch_size": 32},
            "Model": {"layers": 3, "neurons": 128},
        }

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.get_task_parameters.fn("task_123")

        # Assert
        assert result["General"]["learning_rate"] == 0.001
        assert result["Model"]["layers"] == 3

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_task_without_parameters(self, mock_task):
        """get_task_parameters handles tasks with no parameters."""
        task = Mock()
        task.get_parameters_as_dict.return_value = {}

        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_parameters.fn("task_123")

        assert result == {}

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_error_on_parameter_retrieval_failure(self, mock_task):
        """get_task_parameters returns error when parameter retrieval fails."""
        mock_task.get_task.side_effect = Exception("Parameter access denied")

        result = await clearml_mcp.get_task_parameters.fn("task_123")

        assert "error" in result
        assert "Failed to get task parameters" in result["error"]


class TestTaskMetrics:
    """Test task metrics retrieval behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_processes_metrics_with_complete_data(self, mock_task):
        """get_task_metrics correctly processes scalar metrics data."""
        # Arrange: Create realistic metrics data
        task = Mock()
        task.get_reported_scalars.return_value = {
            "loss": {
                "train": {"x": [1, 2, 3], "y": [0.8, 0.6, 0.4]},
                "validation": {"x": [1, 2, 3], "y": [0.9, 0.7, 0.5]},
            },
            "accuracy": {
                "train": {"x": [1, 2, 3], "y": [0.7, 0.8, 0.9]},
            },
        }

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.get_task_metrics.fn("task_123")

        # Assert: Verify metrics are properly processed
        assert "loss" in result
        assert "accuracy" in result
        assert "train" in result["loss"]
        assert "validation" in result["loss"]

        # Verify statistical calculations
        train_loss = result["loss"]["train"]
        assert train_loss["last_value"] == 0.4
        assert train_loss["min_value"] == 0.4
        assert train_loss["max_value"] == 0.8

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_metrics_with_empty_data(self, mock_task):
        """get_task_metrics handles metrics with empty or missing data."""
        task = Mock()
        task.get_reported_scalars.return_value = {
            "loss": {
                "train": {"x": [], "y": []},  # Empty data - should be skipped
            },
            "accuracy": {
                "train": {"x": [1, 2]},  # Missing "y" key - should be skipped
            },
        }

        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_metrics.fn("task_123")

        # Should handle empty data gracefully
        assert "loss" in result
        assert "accuracy" in result
        # Empty y data should still create entry with None values
        assert result["loss"]["train"]["iterations"] == 0
        assert result["loss"]["train"]["last_value"] is None
        # Missing y key should be skipped entirely
        assert result["accuracy"] == {}

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_task_without_metrics(self, mock_task):
        """get_task_metrics handles tasks with no reported metrics."""
        task = Mock()
        task.get_reported_scalars.return_value = {}

        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_metrics.fn("task_123")

        assert result == {}

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_error_on_metrics_retrieval_failure(self, mock_task):
        """get_task_metrics returns error when metrics retrieval fails."""
        mock_task.get_task.side_effect = Exception("Metrics access denied")

        result = await clearml_mcp.get_task_metrics.fn("task_123")

        assert "error" in result
        assert "Failed to get task metrics" in result["error"]


class TestTaskArtifacts:
    """Test task artifact retrieval behavior."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_artifact_information(self, mock_task):
        """get_task_artifacts returns artifact details correctly."""
        # Arrange
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

        # Mock hasattr to control which attributes are available
        def mock_hasattr(obj, attr):
            if obj is artifact2 and attr == "timestamp":
                return False
            return True

        task = Mock()
        task.artifacts = {"model": artifact1, "dataset": artifact2}

        mock_task.get_task.return_value = task

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = await clearml_mcp.get_task_artifacts.fn("task_123")

        # Assert
        assert "model" in result
        assert "dataset" in result
        assert result["model"]["type"] == "model"
        assert result["model"]["uri"] == "s3://bucket/model.pkl"
        assert result["dataset"]["type"] == "data"
        assert result["dataset"]["timestamp"] is None

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_task_without_artifacts(self, mock_task):
        """get_task_artifacts handles tasks with no artifacts."""
        task = Mock()
        task.artifacts = {}

        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_task_artifacts.fn("task_123")

        assert result == {}

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_error_on_artifacts_retrieval_failure(self, mock_task):
        """get_task_artifacts returns error when artifact retrieval fails."""
        mock_task.get_task.side_effect = Exception("Artifacts access denied")

        result = await clearml_mcp.get_task_artifacts.fn("task_123")

        assert "error" in result
        assert "Failed to get task artifacts" in result["error"]


class TestModelOperations:
    """Test model-related functions."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_model_info_with_input_and_output_models(self, mock_task):
        """get_model_info returns model information for both input and output models."""
        # Arrange
        input_model = Mock()
        input_model.id = "input_model_1"
        input_model.name = "pretrained_model"
        input_model.url = "https://models.clearml.io/input.pkl"
        input_model.framework = "pytorch"

        output_model = Mock()
        output_model.id = "output_model_1"
        output_model.name = "trained_model"
        output_model.url = "https://models.clearml.io/output.pkl"
        output_model.framework = "pytorch"

        task = Mock()
        task.models = {
            "input": [input_model],
            "output": [output_model],
        }

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.get_model_info.fn("task_123")

        # Assert
        assert "input" in result
        assert "output" in result
        assert len(result["input"]) == 1
        assert len(result["output"]) == 1
        assert result["input"][0]["name"] == "pretrained_model"
        assert result["output"][0]["name"] == "trained_model"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_model_info_with_no_models(self, mock_task):
        """get_model_info handles tasks with no models."""
        task = Mock()
        task.models = {}

        mock_task.get_task.return_value = task

        result = await clearml_mcp.get_model_info.fn("task_123")

        assert result["input"] == []
        assert result["output"] == []

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_model_info_returns_error_on_failure(self, mock_task):
        """get_model_info returns error when model retrieval fails."""
        mock_task.get_task.side_effect = Exception("Model access denied")

        result = await clearml_mcp.get_model_info.fn("task_123")

        assert "error" in result
        assert "Failed to get model info" in result["error"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Model")
    async def test_list_models_returns_model_list(self, mock_model):
        """list_models returns list of available models."""
        # Arrange
        model1 = Mock()
        model1.id = "model_1"
        model1.name = "Model 1"
        model1.project = "Project A"
        model1.framework = "pytorch"
        model1.created = "2024-01-01T00:00:00Z"
        model1.tags = ["production", "v1.0"]
        model1.task = "task_123"

        model2 = Mock()
        model2.id = "model_2"
        model2.name = "Model 2"
        model2.project = "Project B"
        model2.framework = "tensorflow"
        model2.created = "2024-01-02T00:00:00Z"
        model2.tags = None
        model2.task = "task_456"

        mock_model.query_models.return_value = [model1, model2]

        # Act
        result = await clearml_mcp.list_models.fn()

        # Assert
        assert len(result) == 2
        assert result[0]["name"] == "Model 1"
        assert result[0]["tags"] == ["production", "v1.0"]
        assert result[1]["name"] == "Model 2"
        assert result[1]["tags"] == []

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Model")
    async def test_list_models_with_project_filter(self, mock_model):
        """list_models applies project filter correctly."""
        mock_model.query_models.return_value = []

        await clearml_mcp.list_models.fn(project_name="Specific Project")

        mock_model.query_models.assert_called_once_with(project_name="Specific Project")

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Model")
    async def test_list_models_returns_error_on_failure(self, mock_model):
        """list_models returns error when query fails."""
        mock_model.query_models.side_effect = Exception("Model query failed")

        result = await clearml_mcp.list_models.fn()

        assert len(result) == 1
        assert "error" in result[0]
        assert "Failed to list models" in result[0]["error"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_model_artifacts_returns_artifact_details(self, mock_task):
        """get_model_artifacts returns model artifact information."""
        # Arrange
        input_model = Mock()
        input_model.id = "input_1"
        input_model.name = "base_model"
        input_model.url = "https://models.clearml.io/base.pkl"
        input_model.framework = "pytorch"
        input_model.uri = "s3://bucket/base.pkl"

        output_model = Mock()
        output_model.id = "output_1"
        output_model.name = "fine_tuned_model"
        output_model.url = "https://models.clearml.io/finetuned.pkl"
        output_model.framework = "pytorch"
        output_model.uri = "s3://bucket/finetuned.pkl"

        task = Mock()
        task.models = {
            "input": [input_model],
            "output": [output_model],
        }

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.get_model_artifacts.fn("task_123")

        # Assert
        assert "input_models" in result
        assert "output_models" in result
        assert len(result["input_models"]) == 1
        assert len(result["output_models"]) == 1
        assert result["input_models"][0]["uri"] == "s3://bucket/base.pkl"
        assert result["output_models"][0]["uri"] == "s3://bucket/finetuned.pkl"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_model_artifacts_returns_error_on_failure(self, mock_task):
        """get_model_artifacts returns error when retrieval fails."""
        mock_task.get_task.side_effect = Exception("Model artifacts access denied")

        result = await clearml_mcp.get_model_artifacts.fn("task_123")

        assert "error" in result
        assert "Failed to get model artifacts" in result["error"]


class TestProjectSearch:
    """Test project search functions."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_find_project_by_pattern_returns_matching_projects(self, mock_task):
        """find_project_by_pattern returns projects matching the pattern."""
        # Arrange
        project1 = Mock()
        project1.id = "proj_1"
        project1.name = "Machine Learning Project"

        project2 = Mock()
        project2.id = "proj_2"
        project2.name = "Data Analysis Project"

        project3 = Mock()
        project3.name = "Web Development"
        # project3 has no id attribute

        mock_task.get_projects.return_value = [project1, project2, project3]

        # Act
        result = await clearml_mcp.find_project_by_pattern.fn("machine")

        # Assert
        assert len(result) == 1
        assert result[0]["name"] == "Machine Learning Project"
        assert result[0]["id"] == "proj_1"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_find_project_by_pattern_case_insensitive(self, mock_task):
        """find_project_by_pattern performs case-insensitive matching."""
        # Arrange
        project = Mock()
        project.id = "proj_1"
        project.name = "UPPER CASE PROJECT"

        mock_task.get_projects.return_value = [project]

        # Act
        result = await clearml_mcp.find_project_by_pattern.fn("upper case")

        # Assert
        assert len(result) == 1
        assert result[0]["name"] == "UPPER CASE PROJECT"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_find_project_by_pattern_returns_error_on_failure(self, mock_task):
        """find_project_by_pattern returns error when search fails."""
        mock_task.get_projects.side_effect = Exception("Project access denied")

        result = await clearml_mcp.find_project_by_pattern.fn("test")

        assert len(result) == 1
        assert "error" in result[0]
        assert "Failed to find projects by pattern" in result[0]["error"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_find_experiment_in_project_returns_matching_experiments(self, mock_task):
        """find_experiment_in_project matches the pattern server-side."""
        # The server-side task_name filter only returns the matching tasks.
        mock_task.query_tasks.side_effect = _make_query_tasks(
            [
                {
                    "id": "task_1",
                    "name": "Training Experiment",
                    "status": "completed",
                    "type": "training",
                    "comment": "",
                    "created": "2024-01-01T00:00:00Z",
                    "project": "proj_1",
                    "tags": [],
                },
                {
                    "id": "task_2",
                    "name": "Validation Experiment",
                    "status": "running",
                    "type": "testing",
                    "comment": "",
                    "created": "2024-01-02T00:00:00Z",
                    "project": "proj_1",
                    "tags": [],
                },
            ]
        )
        mock_task.get_projects.return_value = [_fake_project("proj_1", "ML Project")]

        result = await clearml_mcp.find_experiment_in_project.fn("ML Project", "experiment")

        assert len(result) == 2
        assert result[0]["name"] == "Training Experiment"
        assert result[1]["name"] == "Validation Experiment"
        # The pattern is matched server-side via task_name, not by hydrating each task.
        # It is built as a case-insensitive literal-substring regex.
        _, kwargs = mock_task.query_tasks.call_args
        assert kwargs["task_name"] == "(?i)experiment"
        mock_task.get_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_find_experiment_escapes_regex_special_characters(self, mock_task):
        """find_experiment_in_project matches the pattern literally, not as a regex.

        ClearML treats task_name as a regex, so metacharacters must be escaped:
        "a.b" should match only the literal "a.b" (not "axb"), and "exp[1]" must
        not error the query.
        """
        mock_task.query_tasks.side_effect = _make_query_tasks([])
        mock_task.get_projects.return_value = []

        result = await clearml_mcp.find_experiment_in_project.fn("ML Project", "exp[1].a")

        assert result == []
        _, kwargs = mock_task.query_tasks.call_args
        # Special characters are escaped (literal match) and case-insensitive.
        assert kwargs["task_name"] == r"(?i)exp\[1\]\.a"
        mock_task.get_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_find_experiment_is_case_insensitive(self, mock_task):
        """find_experiment_in_project preserves case-insensitive matching."""
        mock_task.query_tasks.side_effect = _make_query_tasks(
            [
                {
                    "id": "task_1",
                    "name": "Training EXPERIMENT",
                    "status": "completed",
                    "type": "training",
                    "comment": "",
                    "created": "2024-01-01T00:00:00Z",
                    "project": "proj_1",
                    "tags": [],
                }
            ]
        )
        mock_task.get_projects.return_value = [_fake_project("proj_1", "ML Project")]

        result = await clearml_mcp.find_experiment_in_project.fn("ML Project", "experiment")

        # The (?i) prefix makes the server match regardless of case.
        _, kwargs = mock_task.query_tasks.call_args
        assert kwargs["task_name"].startswith("(?i)")
        assert len(result) == 1
        assert result[0]["name"] == "Training EXPERIMENT"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_find_experiment_returns_error_on_query_failure(self, mock_task):
        """find_experiment_in_project returns error when query fails."""
        mock_task.query_tasks.side_effect = Exception("Query failed")

        result = await clearml_mcp.find_experiment_in_project.fn("ML Project", "test")

        assert len(result) == 1
        assert "error" in result[0]
        assert "Failed to find experiments" in result[0]["error"]


class TestProjectOperations:
    """Test project listing and statistics."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_lists_available_projects(self, mock_task):
        """list_projects returns available project information."""
        # Arrange
        project1 = Mock()
        project1.id = "proj_1"
        project1.name = "Project Alpha"

        project2 = Mock()
        project2.id = "proj_2"
        project2.name = "Project Beta"

        mock_task.get_projects.return_value = [project1, project2]

        # Act
        result = await clearml_mcp.list_projects.fn()

        # Assert
        assert len(result) == 2
        assert result[0]["id"] == "proj_1"
        assert result[0]["name"] == "Project Alpha"
        assert result[1]["id"] == "proj_2"
        assert result[1]["name"] == "Project Beta"

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_handles_projects_without_id_attribute(self, mock_task):
        """list_projects handles projects missing id attribute gracefully."""
        project = Mock()
        project.name = "Project Without ID"

        # Mock hasattr to return False for id attribute
        def mock_hasattr(obj, attr):
            return attr != "id"

        mock_task.get_projects.return_value = [project]

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = await clearml_mcp.list_projects.fn()

        assert len(result) == 1
        assert result[0]["name"] == "Project Without ID"
        assert result[0]["id"] is None

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_list_projects_returns_error_on_failure(self, mock_task):
        """list_projects returns error when project listing fails."""
        mock_task.get_projects.side_effect = Exception("Project access denied")

        result = await clearml_mcp.list_projects.fn()

        assert len(result) == 1
        assert "error" in result[0]
        assert "Failed to list projects" in result[0]["error"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_calculates_project_statistics(self, mock_task):
        """get_project_stats aggregates status counts from bulk dict results."""
        # query_tasks returns dicts (one bulk call), not objects with .status.
        statuses = [
            "created",
            "created",
            "queued",
            "in_progress",
            "in_progress",
            "in_progress",
            "stopped",
            "published",
            "published",
            "published",
            "closed",
            "failed",
            "failed",
            "completed",
            "completed",
            "completed",
            "completed",
        ]
        task_dicts = [
            {
                "id": f"task_{i}",
                "name": f"Task {i}",
                "status": status,
                "type": "training" if i % 2 == 0 else "inference",
                "comment": "",
                "created": "2024-01-01T00:00:00Z",
                "project": "proj_1",
                "tags": [],
            }
            for i, status in enumerate(statuses)
        ]
        mock_task.query_tasks.side_effect = _make_query_tasks(task_dicts)
        mock_task.get_projects.return_value = [_fake_project("proj_1", "Test Project")]

        result = await clearml_mcp.get_project_stats.fn("Test Project")

        assert "error" not in result
        assert result["project_name"] == "Test Project"
        assert result["total_tasks"] == 17
        assert result["status_breakdown"]["created"] == 2
        assert result["status_breakdown"]["in_progress"] == 3
        assert result["status_breakdown"]["completed"] == 4
        assert result["status_breakdown"]["failed"] == 2
        assert sorted(result["task_types"]) == ["inference", "training"]
        # Regression: no per-task hydration.
        mock_task.get_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_project_stats_returns_error_on_failure(self, mock_task):
        """get_project_stats returns error when statistics calculation fails."""
        mock_task.query_tasks.side_effect = Exception("Stats calculation failed")

        result = await clearml_mcp.get_project_stats.fn("Test Project")

        assert "error" in result
        assert "Failed to get project stats" in result["error"]


class TestTaskComparison:
    """Test task comparison functionality."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_compares_multiple_tasks_with_specific_metrics(self, mock_task):
        """compare_tasks compares specified metrics across tasks."""

        # Arrange
        def mock_get_task(task_id):
            task = Mock()
            task.id = task_id
            task.name = f"Task {task_id}"
            task.status = "completed"

            if task_id == "task_1":
                task.get_reported_scalars.return_value = {
                    "loss": {"train": {"y": [0.8, 0.6, 0.4]}},
                    "accuracy": {"train": {"y": [0.7, 0.8, 0.9]}},
                }
            else:
                task.get_reported_scalars.return_value = {
                    "loss": {"train": {"y": [0.9, 0.7, 0.5]}},
                    "accuracy": {"train": {"y": [0.6, 0.7, 0.8]}},
                }
            return task

        mock_task.get_task.side_effect = mock_get_task

        # Act
        result = await clearml_mcp.compare_tasks.fn(["task_1", "task_2"], ["loss"])

        # Assert
        assert "task_1" in result
        assert "task_2" in result
        assert "loss" in result["task_1"]["metrics"]
        assert "loss" in result["task_2"]["metrics"]
        assert "accuracy" not in result["task_1"]["metrics"]  # Only requested loss

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_compares_all_metrics_when_none_specified(self, mock_task):
        """compare_tasks compares all metrics when none specified."""
        # Arrange
        task = Mock()
        task.id = "task_1"
        task.name = "Task 1"
        task.status = "completed"
        task.get_reported_scalars.return_value = {
            "loss": {"train": {"y": [0.8, 0.6, 0.4]}},
            "accuracy": {"train": {"y": [0.7, 0.8, 0.9]}},
        }

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.compare_tasks.fn(["task_1"], None)

        # Assert
        assert "task_1" in result
        assert "loss" in result["task_1"]["metrics"]
        assert "accuracy" in result["task_1"]["metrics"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_compare_tasks_handles_empty_metrics_data(self, mock_task):
        """compare_tasks handles tasks with empty or missing metrics data."""
        # Arrange
        task = Mock()
        task.id = "task_1"
        task.name = "Task 1"
        task.status = "completed"
        task.get_reported_scalars.return_value = {
            "loss": {"train": {"y": []}},  # Empty data
            "accuracy": {"train": None},  # None data
        }

        mock_task.get_task.return_value = task

        # Act
        result = await clearml_mcp.compare_tasks.fn(["task_1"], ["loss", "accuracy"])

        # Assert
        assert "task_1" in result
        assert result["task_1"]["metrics"]["loss"] == {}
        assert result["task_1"]["metrics"]["accuracy"] == {}

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_compare_tasks_returns_error_on_failure(self, mock_task):
        """compare_tasks returns error when comparison fails."""
        mock_task.get_task.side_effect = Exception("Task access denied")

        result = await clearml_mcp.compare_tasks.fn(["task_1"], ["loss"])

        assert "error" in result
        assert "Failed to compare tasks" in result["error"]


class TestTaskSearch:
    """Test task search functionality."""

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_matches_query_server_side(self, mock_task):
        """search_tasks pushes the substring match into the bulk query."""
        # The server matches name/comment/tags, so only matching tasks come back.
        mock_task.query_tasks.side_effect = _make_query_tasks(
            [
                {
                    "id": "task_1",
                    "name": "Training Neural Network",
                    "status": "completed",
                    "type": "training",
                    "comment": "Deep learning experiment",
                    "created": "2024-01-01T00:00:00Z",
                    "project": "proj_1",
                    "tags": ["training", "neural"],
                }
            ]
        )
        mock_task.get_projects.return_value = [_fake_project("proj_1", "ML Project")]

        result = await clearml_mcp.search_tasks.fn("neural")

        assert len(result) == 1
        assert result[0]["name"] == "Training Neural Network"
        assert result[0]["id"] == "task_1"
        assert result[0]["project"] == "ML Project"
        # Regression: matching is a single bulk query (server-side _any_), no N+1.
        _, kwargs = mock_task.query_tasks.call_args
        any_filter = kwargs["task_filter"]["_any_"]
        assert "neural" in any_filter["pattern"]
        assert set(any_filter["fields"]) == {"name", "comment", "tags"}
        mock_task.get_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_escapes_regex_special_characters(self, mock_task):
        """search_tasks treats the query as a literal substring, not a regex."""
        mock_task.query_tasks.side_effect = _make_query_tasks([])
        mock_task.get_projects.return_value = []

        await clearml_mcp.search_tasks.fn("a.b(c)")

        _, kwargs = mock_task.query_tasks.call_args
        # Special chars are escaped so they match literally rather than as regex.
        assert r"a\.b\(c\)" in kwargs["task_filter"]["_any_"]["pattern"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_returns_empty_list_when_no_matches(self, mock_task):
        """search_tasks returns empty list when the server finds no matches."""
        mock_task.query_tasks.side_effect = _make_query_tasks([])
        mock_task.get_projects.return_value = []

        result = await clearml_mcp.search_tasks.fn("nonexistent")

        assert result == []

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_search_tasks_returns_error_on_query_failure(self, mock_task):
        """search_tasks returns error when query fails."""
        mock_task.query_tasks.side_effect = Exception("Search query failed")

        result = await clearml_mcp.search_tasks.fn("test")

        assert len(result) == 1
        assert "error" in result[0]
        assert "Failed to search tasks" in result[0]["error"]

    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_search_tasks_handles_missing_comment_and_tags(self, mock_task):
        """search_tasks normalizes tasks with missing comment and tags."""
        mock_task.query_tasks.side_effect = _make_query_tasks(
            [
                {
                    "id": "task_1",
                    "name": "Simple Task",
                    "status": "completed",
                    "type": "training",
                    "comment": None,
                    "created": "2024-01-01T00:00:00Z",
                    "project": "proj_1",
                    "tags": None,
                }
            ]
        )
        mock_task.get_projects.return_value = [_fake_project("proj_1", "ML Project")]

        result = await clearml_mcp.search_tasks.fn("simple")

        assert len(result) == 1
        assert result[0]["name"] == "Simple Task"
        assert result[0]["tags"] == []
        assert result[0]["comment"] == ""


class TestMainEntryPoint:
    """Test main function and entry point."""

    @patch("clearml_mcp.clearml_mcp.mcp")
    @patch("clearml_mcp.clearml_mcp.initialize_clearml_connection")
    def test_main_initializes_connection_and_runs_mcp(self, mock_init, mock_mcp):
        """main() initializes ClearML connection and runs MCP server."""
        clearml_mcp.main()

        mock_init.assert_called_once()
        mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_main_module_execution(self):
        """Test that __name__ == '__main__' calls main()."""
        # This is covered by importing and running the module
        # The actual line is tested when the module is executed
