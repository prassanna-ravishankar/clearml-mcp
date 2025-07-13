# ClearML MCP Server

This project implements a Model Context Protocol (MCP) server for ClearML, enabling AI agents to interact with ClearML experiments, models, and projects.

## Project Overview

The ClearML MCP server provides comprehensive tools for AI agents to:
- Discover and analyze ML experiments
- Compare model performance across tasks
- Retrieve training metrics and artifacts
- Search and filter projects and tasks
- Get comprehensive model context and lineage

## Available Tools

### Task Operations
- `get_task_info(task_id)` - Get ClearML task details, parameters, and status
- `list_tasks(project_name, status, tags)` - List ClearML tasks with filters
- `get_task_parameters(task_id)` - Get task hyperparameters and configuration
- `get_task_metrics(task_id)` - Get task training metrics and scalars
- `get_task_artifacts(task_id)` - Get task artifacts and outputs

### Model Operations
- `get_model_info(task_id)` - Get model metadata and configuration
- `list_models(project_name)` - List available models with filtering
- `get_model_artifacts(task_id)` - Get model files and download URLs

### Project Operations
- `list_projects()` - List available ClearML projects
- `get_project_stats(project_name)` - Get project statistics and task counts

### Analysis Tools
- `compare_tasks(task_ids, metrics)` - Compare multiple tasks by metrics
- `search_tasks(query, project_name)` - Search tasks by name, tags, or description

## Technical Details

### Dependencies
- fastmcp>=0.1.0
- clearml>=1.16.0
- pydantic>=2.0.0

### Architecture
- Built with FastMCP framework
- Uses stdio transport for Claude Desktop integration
- Leverages existing ~/clearml.conf for authentication
- Implements JSON-RPC 2.0 protocol

### Entry Point
- Main executable: `clearml-mcp` (via uvx)
- Main module: `src/clearml_mcp/clearml_mcp.py`

## Usage

### Prerequisites
Users must have a configured `~/clearml.conf` file with:
```
[api]
api_server = https://your-clearml-server.com
access_key = your-access-key
secret_key = your-secret-key
```

### Running the Server
```bash
# Using uvx (no installation needed)
uvx clearml-mcp

# Verify clearml.conf works first
clearml-task --help
```

### Claude Desktop Integration
Add to Claude Desktop configuration:
```json
{
  "mcpServers": {
    "clearml": {
      "command": "uvx",
      "args": ["clearml-mcp"]
    }
  }
}
```

## Development Commands

### Testing
```bash
# Run tests (to be implemented)
pytest tests/

# Type checking (to be implemented)
mypy src/clearml_mcp/
```

### Building
```bash
# Build package
hatch build

# Publish to PyPI
hatch publish
```

## Implementation Status

- [ ] Core MCP server with 12 tools
- [ ] PyPI packaging
- [ ] Claude Desktop integration
- [ ] Documentation
- [ ] Tests

## Security Considerations

- Validates task_id format before API calls
- Uses existing clearml.conf for authentication
- No credentials stored in code
- Local-only usage (no remote server endpoints)

## References

- [MCP Quickstart](https://modelcontextprotocol.io/quickstart/server)
- [ClearML SDK Documentation](https://clear.ml/docs/latest/docs/references/sdk/task)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)
