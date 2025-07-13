# ClearML MCP Examples

This directory contains examples showing how to use the ClearML MCP server with different AI frameworks and libraries.

## Smolagents Example

The smolagents example demonstrates how to create an intelligent agent that can analyze ClearML experiments using natural language queries.

### Prerequisites

1. **ClearML Setup**: Ensure you have ClearML configured with `~/.clearml.conf`:
   ```bash
   clearml-init
   ```

2. **ClearML MCP Server**: Install and test the server:
   ```bash
   uvx clearml-mcp
   ```

3. **Dependencies**: Install the required packages:
   ```bash
   # Option 1: Use uv dependency group (recommended)
   uv sync --group examples
   
   # Option 2: Install directly
   cd examples/smolagents_example
   pip install -r requirements.txt
   ```

### Quick Start

#### Simple Example
Run the basic example to test the integration:

```bash
# Using task runner (recommended)
uv run task example-simple

# Or directly
python examples/smolagents_example/simple_example.py
```

#### OpenAI-Compatible Example (Recommended)
Run the enhanced example using Gemini 2.0 Flash via OpenAI API:

```bash
# Using task runner (recommended)
uv run task example

# Or directly  
python examples/smolagents_example/openai_compatible_example.py
```

Features:
- **Rich UI**: Beautiful terminal interface with panels and progress bars
- **Gemini 2.0 Flash**: Latest model via OpenAI-compatible API
- **Enhanced Analysis**: More detailed experiment insights
- **Demo mode**: Runs 5 comprehensive analysis queries
- **Interactive mode**: Real-time Q&A with rich formatting

#### Legacy Gemini Example
Run the original Gemini API example:

```bash
python examples/smolagents_example/clearml_analysis_agent.py
```

### Example Queries

The agent can handle queries like:

- **Project Overview**: "List all available ClearML projects"
- **Experiment Analysis**: "Analyze experiment efe5f7a6c5f34a15b4bfbf1c33660e20"  
- **Performance Metrics**: "Show me the training metrics for experiment xyz"
- **Comparison**: "Compare experiments abc123 and def456"
- **Search**: "Find experiments related to computer vision"
- **Hyperparameters**: "What hyperparameters were used in experiment xyz?"

### Features Demonstrated

1. **Natural Language Interface**: Ask questions about experiments in plain English
2. **Comprehensive Analysis**: Get insights into training metrics, parameters, and performance
3. **Multi-tool Integration**: Combines ClearML data with Gemini's analytical capabilities
4. **Interactive Mode**: Real-time Q&A about your ML experiments
5. **Error Handling**: Graceful handling of invalid experiment IDs or connectivity issues

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Smolagents    │    │   ClearML MCP    │    │     ClearML     │
│     Agent       │◄──►│     Server       │◄──►│    Instance     │
│  (Gemini API)   │    │  (12 tools)      │    │  (Experiments)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

The agent uses:
- **Gemini** for natural language understanding and response generation
- **ClearML MCP Server** for structured access to experiment data
- **Smolagents** for intelligent tool orchestration and code generation

### Troubleshooting

If you encounter issues:

1. **ClearML Configuration**: Verify with `clearml-task --help`
2. **MCP Server**: Test with `uvx clearml-mcp` (should show server startup)
3. **Gemini API**: Ensure API key is valid and has quota
4. **Dependencies**: Check all packages are installed correctly

### Sample Output

```
🔬 ClearML Analysis Agent
Powered by Smolagents + Gemini + ClearML MCP

🚀 Starting ClearML Analysis Agent with Gemini...
============================================================
✅ Connected to ClearML MCP server
🛠️  Available tools: 12 ClearML tools

🔍 Analysis 1: Project Overview
📝 Query: List all available ClearML projects and give me a summary
--------------------------------------------------
✅ Analysis Complete!
📊 Result: Found 5 ClearML projects:
1. Computer Vision Pipeline - 23 experiments
2. NLP Research - 15 experiments  
3. Recommendation Systems - 8 experiments
...
```

### Next Steps

- Modify queries in `clearml_analysis_agent.py` for your specific use cases
- Add custom analysis functions for your experimental workflows
- Integrate with other MCP servers for multi-platform analysis
- Extend with additional AI models or agent frameworks