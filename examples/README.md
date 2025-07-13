# ClearML MCP Examples

This directory contains examples demonstrating how to use the ClearML MCP server with smolagents for ML experiment debugging and analysis.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync --group examples
   ```

2. **Run the main consolidated example:**
   ```bash
   uv run --group examples examples/consolidated_debugger.py
   ```

## Examples Overview

### 🎯 Main Example: `consolidated_debugger.py`
**Recommended starting point** - Complete debugging workflow with all features:

- **Real experiment discovery** - Find actual experiments in your ClearML instance
- **Scalar convergence analysis** - Analyze training/validation metrics
- **ML issue detection** - Identify overfitting, learning rate problems, instability
- **Expert recommendations** - Get actionable optimization advice

```bash
# Run complete analysis
uv run --group examples examples/consolidated_debugger.py

# Or use the task runner
uv run consolidated-debug
```

**Features:**
- ✅ Interactive mode selection
- ✅ Real experiment discovery
- ✅ Convergence pattern analysis
- ✅ Common ML issue detection
- ✅ Expert optimization recommendations
- ✅ Rate limiting graceful handling
- ✅ Non-interactive execution support

### 📊 Individual Examples (by complexity)

#### Simple Examples
- `01_simple_example.py` - Basic MCP + smolagents integration
- `02_openai_compatible_example.py` - Using Gemini via OpenAI API

#### Intermediate Examples
- `03_find_real_experiments.py` - Discover experiments in ClearML
- `04_quick_scalar_demo.py` - Quick scalar convergence analysis

## Usage Patterns

### Complete Debugging Workflow
```python
from consolidated_debugger import ClearMLDebugger

debugger = ClearMLDebugger()
debugger.run_complete_analysis()
```

### Scalar Analysis Only
```python
debugger = ClearMLDebugger()
debugger.analyze_scalar_patterns()
```

### Issue Detection Demo
```python
debugger = ClearMLDebugger()
debugger.demonstrate_issue_detection()
```

## Task Runners (pyproject.toml)

For convenience, several task runners are available:

```bash
# Main consolidated example
uv run consolidated-debug

# Individual examples by complexity
uv run example-simple       # 01: Basic MCP + smolagents integration
uv run example              # 02: Gemini via OpenAI API example
uv run find-experiments     # 03: Discover real experiments in ClearML
uv run quick-scalar         # 04: Quick scalar convergence analysis
```

## What These Examples Demonstrate

### 🔍 **Experiment Discovery**
- List all ClearML projects
- Find experiments with training data
- Identify experiments with scalar metrics
- Locate best candidates for debugging

### 📈 **Convergence Analysis**
- Training/validation loss patterns
- Accuracy progression analysis
- Learning rate effectiveness
- Convergence rate calculations
- Optimal stopping point detection

### 🚨 **Issue Detection**
- **Overfitting**: Train/val performance gaps
- **Learning Rate Problems**: Oscillations, slow convergence
- **Training Instability**: Loss spikes, NaN values, gradient explosion
- **Poor Generalization**: Validation performance issues

### ⚙️ **Optimization Recommendations**
- Hyperparameter tuning suggestions
- Architecture improvements
- Regularization strategies
- Training stability fixes
- Performance optimization tips

## Common ML Issues Detected

| Issue | Symptoms | Recommendations |
|-------|----------|----------------|
| **Overfitting** | Train accuracy >> Val accuracy, Val loss increasing | Add dropout/weight decay, data augmentation, early stopping |
| **Learning Rate Too High** | Oscillating loss, unstable training | Reduce LR by 5-10x, add gradient clipping |
| **Learning Rate Too Low** | Very slow progress, minimal loss decrease | Increase LR by 2-10x, use warmup, check initialization |
| **Gradient Explosion** | Loss suddenly jumps to high values or NaN | Gradient clipping, lower LR, numerical stability fixes |
| **Vanishing Gradients** | Extremely slow learning, tiny gradient norms | Better initialization, architecture changes, higher LR |
| **Training Instability** | Random loss spikes, inconsistent progress | Gradient clipping, mixed precision fixes, batch size tuning |

## Requirements

- **ClearML Configuration**: `~/.clearml.conf` must be configured
- **API Key**: Gemini API key (provided in examples)
- **Dependencies**: Install with `uv sync --group examples`

```toml
[dependency-groups]
examples = [
    "smolagents[openai,mcp]>=1.20.0",
    "rich>=10.0.0",
]
```

## Configuration

The examples use these default settings:

```python
# Gemini API via OpenAI-compatible endpoint
model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="AIzaSyDAdEToKdFt8SHs25ABz65bx6cedU_zreo",
    temperature=0.1  # Lower for precise analysis
)

# ClearML MCP Server
clearml_server_params = StdioServerParameters(
    command="python",
    args=["-m", "clearml_mcp.clearml_mcp"],
    env=os.environ
)

# Smolagents Configuration
agent = CodeAgent(
    tools=clearml_tools,
    model=model,
    add_base_tools=False,  # Use only ClearML tools
    verbosity_level=2,     # Show detailed tool usage
    max_steps=2           # Limit steps to avoid long runs
)
```

## Error Handling

The examples include robust error handling for common issues:

- **Rate Limiting (429)**: Graceful fallback with analysis summaries
- **API Errors**: Clear error messages and suggestions
- **Missing Experiments**: Demo scenarios when no real data exists
- **Keyboard Interrupts**: Clean shutdown and status messages
- **Import Errors**: Clear dependency installation instructions

## Tips for Real Usage

1. **Start with the consolidated example** - It has the most features and best UX
2. **Check your ClearML config** - Ensure `~/.clearml.conf` is properly set up
3. **Use realistic experiment IDs** - The examples will adapt to your actual data
4. **Monitor API usage** - Gemini has rate limits, examples handle this gracefully
5. **Customize analysis depth** - Adjust `max_steps` and `verbosity_level` as needed

## Next Steps

After running these examples:

1. **Try with your real experiments** - Replace demo data with actual experiment IDs
2. **Customize the analysis** - Modify queries for your specific ML domains
3. **Integrate into your workflow** - Use the patterns in your own debugging scripts
4. **Extend the capabilities** - Add new analysis types and debugging patterns

---

For more information, see the main project documentation and the ClearML MCP server implementation in `src/clearml_mcp/clearml_mcp.py`.
