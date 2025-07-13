# ClearML MCP Server

![ClearML MCP](https://raw.githubusercontent.com/prassanna-ravishankar/clearml-mcp/main/docs/clearml.png)

[![PyPI version](https://badge.fury.io/py/clearml-mcp.svg)](https://badge.fury.io/py/clearml-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight **Model Context Protocol (MCP) server** that enables AI assistants to interact with [ClearML](https://clear.ml) experiments, models, and projects. Get comprehensive ML experiment context and analysis directly in your AI conversations.

## ✨ Features

- **🔍 Experiment Discovery**: Find and analyze ML experiments across projects
- **📊 Performance Analysis**: Compare model metrics and training progress
- **📈 Real-time Metrics**: Access training scalars, validation curves, and convergence analysis
- **🏷️ Smart Search**: Filter tasks by name, tags, status, and custom queries
- **📦 Artifact Management**: Retrieve model files, datasets, and experiment outputs
- **🌐 Cross-platform**: Works with all major AI assistants and code editors

## 📋 Requirements

- **uv** ([installation guide](https://docs.astral.sh/uv/getting-started/installation/)) for `uvx` command
- **ClearML account** with valid API credentials in `~/.clearml/clearml.conf`

## 🚀 Quick Start

### Prerequisites

You need a configured ClearML environment with your credentials in `~/.clearml/clearml.conf`:

```ini
[api]
api_server = https://api.clear.ml
web_server = https://app.clear.ml
files_server = https://files.clear.ml
credentials {
    "access_key": "your-access-key",
    "secret_key": "your-secret-key"
}
```

Get your credentials from [ClearML Settings](https://app.clear.ml/settings).

### Installation

```bash
# Install from PyPI
pip install clearml-mcp

# Or run directly with uvx (no installation needed)
uvx clearml-mcp
```

## 🔌 Integrations

<details>
<summary><strong>🤖 Claude Desktop</strong></summary>

Add to your Claude Desktop configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

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

Alternative with pip installation:
```json
{
  "mcpServers": {
    "clearml": {
      "command": "python",
      "args": ["-m", "clearml_mcp.clearml_mcp"]
    }
  }
}
```
</details>

<details>
<summary><strong>⚡ Cursor</strong></summary>

Add to your Cursor settings (`Ctrl/Cmd + ,` → Search "MCP"):

```json
{
  "mcp.servers": {
    "clearml": {
      "command": "uvx",
      "args": ["clearml-mcp"]
    }
  }
}
```

Or add to `.cursorrules` in your project:
```
When analyzing ML experiments or asking about model performance, use the clearml MCP server to access experiment data, metrics, and artifacts.
```
</details>

<details>
<summary><strong>🔥 Continue</strong></summary>

Add to your Continue configuration (`~/.continue/config.json`):

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
</details>

<details>
<summary><strong>🦾 Cody</strong></summary>

Add to your Cody settings:

```json
{
  "cody.experimental.mcp": {
    "servers": {
      "clearml": {
        "command": "uvx",
        "args": ["clearml-mcp"]
      }
    }
  }
}
```
</details>

<details>
<summary><strong>🧠 Other AI Assistants</strong></summary>

For any MCP-compatible AI assistant, use this configuration:

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

**Compatible with:**
- Zed Editor
- OpenHands
- Roo-Cline
- Any MCP-enabled application
</details>

## 🛠️ Available Tools

The ClearML MCP server provides **14 comprehensive tools** for ML experiment analysis:

### 📊 Task Operations
- `get_task_info` - Get detailed task information, parameters, and status
- `list_tasks` - List tasks with advanced filtering (project, status, tags, user)
- `get_task_parameters` - Retrieve hyperparameters and configuration
- `get_task_metrics` - Access training metrics, scalars, and plots
- `get_task_artifacts` - Get artifacts, model files, and outputs

### 🤖 Model Operations
- `get_model_info` - Get model metadata and configuration details
- `list_models` - Browse available models with filtering
- `get_model_artifacts` - Access model files and download URLs

### 📁 Project Operations
- `list_projects` - Discover available ClearML projects
- `get_project_stats` - Get project statistics and task summaries
- `find_project_by_pattern` - Find projects matching name patterns
- `find_experiment_in_project` - Find specific experiments within projects

### 🔍 Analysis Tools
- `compare_tasks` - Compare multiple tasks by specific metrics
- `search_tasks` - Advanced search by name, tags, comments, and more

## 💡 Usage Examples

### Demo

[![asciicast](https://asciinema.org/a/3eUmgiUJGRVYa9uEJXz5oPXFj.svg)](https://asciinema.org/a/3eUmgiUJGRVYa9uEJXz5oPXFj)

Once configured, you can ask your AI assistant questions like:

- *"Show me the latest experiments in the 'computer-vision' project"*
- *"Compare the accuracy metrics between tasks task-123 and task-456"*
- *"What are the hyperparameters for the best performing model?"*
- *"Find all failed experiments from last week"*
- *"Get the training curves for my latest BERT fine-tuning"*

## 🏗️ Development

### Setup

```bash
# Clone and setup with UV
git clone https://github.com/prassanna-ravishankar/clearml-mcp.git
cd clearml-mcp
uv sync

# Run locally
uv run python -m clearml_mcp.clearml_mcp
```

### Available Commands

```bash
# Run tests with coverage
uv run task coverage

# Lint and format
uv run task lint
uv run task format

# Type checking
uv run task type

# Run examples
uv run task consolidated-debug  # Full ML debugging demo
uv run task example-simple      # Basic integration
uv run task find-experiments    # Discover real experiments
```

### Testing with MCP Inspector

```bash
# Test the MCP server directly
npx @modelcontextprotocol/inspector uvx clearml-mcp
```

## 🚨 Troubleshooting

<details>
<summary><strong>Connection Issues</strong></summary>

**"No ClearML projects accessible"**
- Verify your `~/.clearml/clearml.conf` credentials
- Test with: `python -c "from clearml import Task; print(Task.get_projects())"`
- Check network access to your ClearML server

**Module not found errors**
- Try `bunx clearml-mcp` instead of `uvx clearml-mcp`
- Or use direct Python: `python -m clearml_mcp.clearml_mcp`
</details>

<details>
<summary><strong>Performance Issues</strong></summary>

**Large dataset queries**
- Use filters in `list_tasks` to limit results
- Specify `project_name` to narrow scope
- Use `task_status` filters (`completed`, `running`, `failed`)

**Slow metric retrieval**
- Request specific metrics instead of all metrics
- Use `compare_tasks` with metric names for focused analysis
</details>

## 🤝 Contributing

Contributions welcome! This project uses:

- **UV** for dependency management
- **Ruff** for linting and formatting
- **Pytest** for testing with 69% coverage
- **GitHub Actions** for CI/CD

See our [testing philosophy](.cursor/rules/testing-philosophy.mdc) and [linting approach](.cursor/rules/linting-philosophy.mdc) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- **PyPI**: [clearml-mcp](https://pypi.org/project/clearml-mcp/)
- **ClearML**: [clear.ml](https://clear.ml)
- **Model Context Protocol**: [MCP Specification](https://modelcontextprotocol.io/)

---

**Created by [Prass, The Nomadic Coder](https://github.com/prassanna-ravishankar)**
