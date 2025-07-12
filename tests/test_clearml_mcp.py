"""Tests for ClearML MCP server."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from clearml_mcp import clearml_mcp


class TestClearMLConnection:
    """Test ClearML connection initialization."""
    
    @patch("clearml_mcp.clearml_mcp.Task")
    def test_initialize_connection_success(self, mock_task):
        """Test successful connection initialization."""
        mock_task.get_projects.return_value = [Mock(name="project1")]
        
        # Should not raise
        clearml_mcp.initialize_clearml_connection()
        mock_task.get_projects.assert_called_once()
    
    @patch("clearml_mcp.clearml_mcp.Task")
    def test_initialize_connection_no_projects(self, mock_task):
        """Test connection fails when no projects are accessible."""
        mock_task.get_projects.return_value = []
        
        with pytest.raises(RuntimeError, match="No ClearML projects accessible"):
            clearml_mcp.initialize_clearml_connection()
    
    @patch("clearml_mcp.clearml_mcp.Task")
    def test_initialize_connection_error(self, mock_task):
        """Test connection fails with API error."""
        mock_task.get_projects.side_effect = Exception("API Error")
        
        with pytest.raises(RuntimeError, match="Failed to initialize ClearML connection"):
            clearml_mcp.initialize_clearml_connection()


class TestTaskOperations:
    """Test task-related operations."""
    
    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_task_info_success(self, mock_task):
        """Test successful task info retrieval."""
        mock_task_instance = Mock()
        mock_task_instance.id = "test_id"
        mock_task_instance.name = "test_task"
        mock_task_instance.status = "completed"
        mock_task_instance.get_project_name.return_value = "test_project"
        mock_task_instance.task_type = "training"
        mock_task_instance.comment = "test comment"
        
        mock_data = Mock()
        mock_data.created = "2024-01-01"
        mock_data.last_update = "2024-01-02"
        mock_data.tags = ["tag1", "tag2"]
        mock_task_instance.data = mock_data
        
        mock_task.get_task.return_value = mock_task_instance
        
        result = await clearml_mcp.get_task_info.fn("test_id")
        
        assert result["id"] == "test_id"
        assert result["name"] == "test_task"
        assert result["status"] == "completed"
        assert result["project"] == "test_project"
        assert result["tags"] == ["tag1", "tag2"]
    
    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_task_info_error(self, mock_task):
        """Test task info retrieval with error."""
        mock_task.get_task.side_effect = Exception("Task not found")
        
        result = await clearml_mcp.get_task_info.fn("invalid_id")
        
        assert "error" in result
        assert "Failed to get task info" in result["error"]
    
    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_list_tasks_success(self, mock_task):
        """Test successful task listing."""
        mock_task1 = Mock()
        mock_task1.id = "task1"
        mock_task1.name = "Task 1"
        mock_task1.status = "completed"
        mock_task1.project = "project1"
        mock_task1.created = "2024-01-01"
        mock_task1.tags = ["tag1"]
        
        mock_task.query_tasks.return_value = [mock_task1]
        
        result = await clearml_mcp.list_tasks.fn(project_name="project1", status="completed")
        
        assert len(result) == 1
        assert result[0]["id"] == "task1"
        assert result[0]["name"] == "Task 1"
    
    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_task_parameters_success(self, mock_task):
        """Test successful parameter retrieval."""
        mock_task_instance = Mock()
        mock_task_instance.get_parameters_as_dict.return_value = {
            "General": {"learning_rate": 0.01, "batch_size": 32}
        }
        
        mock_task.get_task.return_value = mock_task_instance
        
        result = await clearml_mcp.get_task_parameters.fn("test_id")
        
        assert "General" in result
        assert result["General"]["learning_rate"] == 0.01
    
    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_get_task_metrics_success(self, mock_task):
        """Test successful metrics retrieval."""
        mock_task_instance = Mock()
        mock_task_instance.get_reported_scalars.return_value = {
            "loss": {
                "train": {"x": [1, 2, 3], "y": [0.5, 0.3, 0.1]},
                "val": {"x": [1, 2, 3], "y": [0.6, 0.4, 0.2]}
            }
        }
        
        mock_task.get_task.return_value = mock_task_instance
        
        result = await clearml_mcp.get_task_metrics.fn("test_id")
        
        assert "loss" in result
        assert "train" in result["loss"]
        assert result["loss"]["train"]["last_value"] == 0.1
        assert result["loss"]["train"]["min_value"] == 0.1
        assert result["loss"]["train"]["max_value"] == 0.5


class TestProjectOperations:
    """Test project-related operations."""
    
    @pytest.mark.asyncio
    @patch("clearml_mcp.clearml_mcp.Task")
    async def test_list_projects_success(self, mock_task):
        """Test successful project listing."""
        mock_project1 = Mock()
        mock_project1.name = "Project 1"
        mock_project1.id = "proj1"
        
        mock_project2 = Mock()
        mock_project2.name = "Project 2"
        
        mock_task.get_projects.return_value = [mock_project1, mock_project2]
        
        result = await clearml_mcp.list_projects.fn()
        
        assert len(result) == 2
        assert result[0]["name"] == "Project 1"
        assert result[0]["id"] == "proj1"
        assert result[1]["name"] == "Project 2"