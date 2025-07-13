## **ClearML MCP Server - Implementation Plan**

### **Goals**
- Build a lightweight MCP server for ClearML model context and operations
- Support Claude Desktop integration via `uvx clearml-mcp`
- Provide comprehensive ClearML tools for AI agents
- Keep it simple: local usage only

### **Phase 1: Core MCP Server**

#### **1.1 Project Structure** ✅
Already set up with modern Python packaging (`src/clearml_mcp/`, `pyproject.toml`)

#### **1.2 Dependencies & Entry Point**
```toml
dependencies = [
    "fastmcp>=0.1.0",
    "clearml>=1.16.0",
    "pydantic>=2.0.0"
]

[project.scripts]
clearml-mcp = "clearml_mcp.clearml_mcp:main"
```

#### **1.3 Core MCP Tools** (`src/clearml_mcp/clearml_mcp.py`)
```python
import os
from fastmcp import FastMCP
from clearml import Task

mcp = FastMCP("clearml-mcp")

# Task Operations
@mcp.tool()
async def get_task_info(task_id: str) -> dict:
    """Get ClearML task details, parameters, and status"""
    # Security: Validate task_id format and user access
    # Implementation using Task.get_task()

@mcp.tool()
async def list_tasks(project_name: str = None, status: str = None, tags: list = None) -> list:
    """List ClearML tasks with filters"""
    # Implementation using Task.query_tasks()

@mcp.tool()
async def get_task_parameters(task_id: str) -> dict:
    """Get task hyperparameters and configuration"""
    # Implementation using Task.get_parameters_as_dict()

@mcp.tool()
async def get_task_metrics(task_id: str) -> dict:
    """Get task training metrics and scalars"""
    # Implementation using Task.get_reported_scalars()

@mcp.tool()
async def get_task_artifacts(task_id: str) -> dict:
    """Get task artifacts and outputs"""
    # Implementation using Task.get_artifacts()

# Model Operations
@mcp.tool()
async def get_model_info(task_id: str) -> dict:
    """Get model metadata and configuration"""
    # Implementation using Task.get_models(), Task.get_model_config_dict()

@mcp.tool()
async def list_models(project_name: str = None) -> list:
    """List available models with filtering"""
    # Implementation using Task.query_tasks() with model filters

@mcp.tool()
async def get_model_artifacts(task_id: str) -> dict:
    """Get model files and download URLs"""
    # Implementation using Task.get_registered_artifacts()

# Project Operations
@mcp.tool()
async def list_projects() -> list:
    """List available ClearML projects"""
    # Implementation using Task.get_projects()

@mcp.tool()
async def get_project_stats(project_name: str) -> dict:
    """Get project statistics and task counts"""
    # Implementation using Task.query_tasks() grouped by project

# Comparison Tools
@mcp.tool()
async def compare_tasks(task_ids: list, metrics: list = None) -> dict:
    """Compare multiple tasks by metrics"""
    # Implementation using Task.get_reported_scalars() for multiple tasks

@mcp.tool()
async def search_tasks(query: str, project_name: str = None) -> list:
    """Search tasks by name, tags, or description"""
    # Implementation using Task.query_tasks() with search filters

def main():
    """Entry point for uvx clearml-mcp"""
    # Initialize ClearML connection with validation
    initialize_clearml_connection()
    mcp.run(transport='stdio')

def initialize_clearml_connection():
    """Initialize and validate ClearML connection"""
    try:
        # Uses existing ~/clearml.conf automatically
        # Validate connection by testing project access
        projects = Task.get_projects()
        if not projects:
            raise ValueError("No ClearML projects accessible - check your clearml.conf")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize ClearML connection: {str(e)}")

if __name__ == "__main__":
    main()
```

#### **1.4 Configuration**
- **Simple:** Use existing `~/clearml.conf` file
- **Automatic:** ClearML SDK discovers and uses user's existing configuration
- **Secure:** Credentials stay on user's machine

#### **1.5 MCP Tools Summary**
**12 Comprehensive Tools:**
- **Task Operations:** get_task_info, list_tasks, get_task_parameters, get_task_metrics, get_task_artifacts
- **Model Operations:** get_model_info, list_models, get_model_artifacts
- **Project Operations:** list_projects, get_project_stats
- **Analysis Tools:** compare_tasks, search_tasks

**Enables AI agents to:**
- Discover and analyze ML experiments
- Compare model performance across tasks
- Retrieve training metrics and artifacts
- Search and filter projects and tasks
- Get comprehensive model context and lineage

### **Phase 2: Distribution**

#### **2.1 PyPI Package**
- Package with `hatchling` (already configured)
- Publish as `clearml-mcp`
- Enable `uvx clearml-mcp` (no installation needed)

#### **2.2 Claude Desktop Integration**
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

**Prerequisites:** Users must have `~/clearml.conf` configured for their ClearML instance

### **Phase 3: Documentation**

#### **3.1 Setup Instructions**
- README with installation and setup instructions
- Prerequisite: existing clearml.conf configuration
- Claude Desktop integration steps

#### **3.2 Usage Examples**
```bash
# Local usage - Simple!
# Uses existing ~/clearml.conf automatically
uvx clearml-mcp

# Verify your clearml.conf is working first
clearml-task --help  # Should work if clearml.conf is configured
```

#### **3.3 Requirements**
```bash
# Users need existing clearml.conf with their deployment
# Example ~/clearml.conf:
# [api]
# api_server = https://your-clearml-server.com
# access_key = your-access-key
# secret_key = your-secret-key
#
# This is typically already configured by ClearML users
```

### **Implementation Priority**
1. **Core MCP tools** (12 comprehensive tools covering tasks, models, projects, and comparison)
2. **PyPI packaging** (enable `uvx clearml-mcp`)
3. **Claude Desktop integration** (local usage via uvx, requires ~/clearml.conf)
4. **Documentation** (setup guide and usage examples)

### **Success Metrics**
- Working MCP server with comprehensive ClearML tools (12 tools)
- Successful PyPI publication (`uvx clearml-mcp`)
- Claude Desktop integration works seamlessly
- Clear documentation and easy setup process
- Positive user feedback from ClearML community

**Total Scope:** ~500-700 lines of core code, providing comprehensive ClearML operations for AI agents including task management, model operations, project analytics, and comparison tools.

**Simple Philosophy:** One deployment method, one configuration approach, focused on delivering value to ClearML users who want AI agent access to their experiments.
