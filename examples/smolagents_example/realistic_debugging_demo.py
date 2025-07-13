#!/usr/bin/env python3
"""
Realistic ML Debugging Demo - Simulating Common Issues

Since the ClearML instance appears to have no real experiments, this demo
shows how the debugging agent would work with common ML training issues.

This demonstrates realistic scenarios:
1. Overfitting detection
2. Learning rate issues  
3. Training instability
4. Convergence problems
5. Hyperparameter optimization suggestions
"""

import os
from mcp import StdioServerParameters

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyDAdEToKdFt8SHs25ABz65bx6cedU_zreo"

try:
    from smolagents import MCPClient, CodeAgent
    from smolagents import OpenAIServerModel
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("❌ Required packages not found. Install with: uv sync --group examples")
    raise

console = Console()


def demonstrate_realistic_debugging():
    """Show realistic ML debugging scenarios and how the agent would handle them."""
    
    console.print(Panel.fit(
        "[bold red]🐛 Realistic ML Debugging Scenarios[/bold red]\n"
        "[dim]Demonstrating how to debug common ML training issues[/dim]",
        border_style="red"
    ))
    
    # Initialize Gemini model
    model = OpenAIServerModel(
        model_id="gemini-2.0-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_API_KEY,
        temperature=0.3  # Higher temperature for creative debugging insights
    )
    
    # Configure ClearML MCP server
    clearml_server_params = StdioServerParameters(
        command="python",
        args=["-m", "clearml_mcp.clearml_mcp"],
        env=os.environ
    )
    
    # Realistic debugging scenarios
    scenarios = [
        {
            "title": "🔥 Overfitting Detection",
            "description": "Model performing well on training but poorly on validation",
            "query": """
            As an ML debugging expert, analyze this hypothetical scenario:
            
            SCENARIO: A deep learning model shows these patterns:
            - Training accuracy: 98% 
            - Validation accuracy: 72%
            - Training loss decreases smoothly to 0.01
            - Validation loss increases after epoch 15
            - Learning rate: 0.01
            - No regularization used
            - Dataset: 10k training, 2k validation samples
            
            ANALYSIS NEEDED:
            1. What is the primary issue here?
            2. What are the warning signs?
            3. What specific solutions would you recommend?
            4. How would you modify the training approach?
            5. What hyperparameters need adjustment?
            
            Provide a detailed debugging analysis as if this were a real ClearML experiment.
            """,
            "expected_issues": ["Overfitting", "No regularization", "High learning rate", "Small dataset"]
        },
        {
            "title": "📉 Learning Rate Problems", 
            "description": "Training loss oscillating or not decreasing properly",
            "query": """
            As an ML debugging expert, analyze this training scenario:
            
            SCENARIO: Neural network training shows erratic behavior:
            - Loss oscillates wildly: 2.1 → 0.8 → 3.2 → 1.1 → 4.5
            - Learning rate: 0.1 (fixed)
            - Optimizer: SGD with momentum 0.9
            - Batch size: 32
            - Model: 5-layer CNN
            - Training keeps going for 100 epochs but never converges
            
            DEBUGGING QUESTIONS:
            1. What's causing the oscillating loss?
            2. Is the learning rate appropriate?
            3. Should we use learning rate scheduling?
            4. What optimizer changes would help?
            5. How can we stabilize training?
            
            Provide specific numerical recommendations and explain the underlying causes.
            """,
            "expected_issues": ["Learning rate too high", "No LR scheduling", "Gradient exploding"]
        },
        {
            "title": "🐌 Slow Convergence Analysis",
            "description": "Model training extremely slowly with minimal progress",
            "query": """
            Analyze this slow training scenario:
            
            SCENARIO: Training is painfully slow:
            - After 50 epochs: loss only dropped from 2.3 → 2.1
            - Learning rate: 0.0001
            - Gradient norms: consistently around 0.001
            - Model: Large transformer (100M parameters)
            - Batch size: 8 (limited by GPU memory)
            - No weight initialization strategy
            - Using Adam optimizer with default settings
            
            INVESTIGATION NEEDED:
            1. Why is learning so slow?
            2. Is this vanishing gradients?
            3. Are the hyperparameters appropriate for this model size?
            4. What initialization strategy should be used?
            5. How can we accelerate training?
            
            Recommend specific changes to get this model training effectively.
            """,
            "expected_issues": ["Learning rate too low", "Poor initialization", "Small batch size", "Vanishing gradients"]
        },
        {
            "title": "⚡ Training Instability",
            "description": "Loss suddenly explodes or training becomes unstable",
            "query": """
            Debug this training instability:
            
            SCENARIO: Training was going well, then disaster:
            - Epochs 1-20: Loss decreasing nicely (2.1 → 0.8)
            - Epoch 21: Loss suddenly jumps to 15.6
            - Epoch 22: Loss becomes NaN
            - Learning rate: 0.001 (seemed stable before)
            - Using mixed precision training
            - Gradient clipping: None
            - Batch size: 64
            
            ROOT CAUSE ANALYSIS:
            1. What caused the sudden loss explosion?
            2. Why did it happen at epoch 21 specifically?
            3. How does mixed precision relate to this?
            4. What safety measures were missing?
            5. How can we prevent this in future runs?
            
            Provide a forensic analysis and prevention strategy.
            """,
            "expected_issues": ["Gradient explosion", "No gradient clipping", "Mixed precision issues", "Numerical instability"]
        }
    ]
    
    with MCPClient(clearml_server_params) as clearml_tools:
        agent = CodeAgent(
            tools=clearml_tools,
            model=model,
            add_base_tools=False,
            verbosity_level=0  # Reduce verbosity for cleaner output
        )
        
        for i, scenario in enumerate(scenarios, 1):
            console.print(Panel(
                f"[bold red]{scenario['title']}[/bold red]\n\n"
                f"[yellow]Scenario:[/yellow] {scenario['description']}\n\n"
                f"[dim]Expected Issues:[/dim] {', '.join(scenario['expected_issues'])}",
                border_style="red",
                padding=(1, 2)
            ))
            
            console.print(f"\n[yellow]🤔 Analyzing scenario {i}...[/yellow]")
            
            try:
                result = agent.run(scenario['query'])
                
                console.print(Panel(
                    f"[bold green]🩺 Expert Debugging Analysis[/bold green]\n\n{result}",
                    border_style="green",
                    padding=(1, 2)
                ))
                
            except Exception as e:
                if "429" in str(e):
                    console.print(Panel(
                        f"[bold yellow]⏳ Rate Limited[/bold yellow]\n\n"
                        f"Hit Gemini API rate limit. The analysis would continue with:\n\n"
                        f"[bold]Expected Analysis for '{scenario['title']}':[/bold]\n"
                        f"• Primary issues: {', '.join(scenario['expected_issues'])}\n"
                        f"• Detailed diagnostic steps\n"
                        f"• Specific hyperparameter recommendations\n"
                        f"• Prevention strategies for future experiments",
                        border_style="yellow",
                        padding=(1, 2)
                    ))
                else:
                    console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")
            
            console.print()
    
    # Summary
    console.print(Panel.fit(
        "[bold green]🎯 Debugging Demo Complete![/bold green]\n\n"
        "[dim]This demonstrates how the ClearML MCP + Smolagents system would:\n"
        "• Identify common ML training issues\n"
        "• Provide expert-level debugging analysis\n"
        "• Recommend specific fixes and improvements\n"
        "• Help prevent similar issues in future experiments[/dim]",
        border_style="green"
    ))


def show_debugging_capabilities():
    """Show what the debugging system can analyze."""
    
    console.print(Panel.fit(
        "[bold blue]🔍 ML Debugging Capabilities[/bold blue]\n"
        "[dim]What our ClearML + Smolagents system can detect and fix[/dim]",
        border_style="blue"
    ))
    
    capabilities = Table(title="Debugging Analysis Capabilities")
    capabilities.add_column("Issue Category", style="cyan", no_wrap=True)
    capabilities.add_column("What We Detect", style="yellow")
    capabilities.add_column("Recommendations", style="green")
    
    capabilities.add_row(
        "Overfitting",
        "• Train/val accuracy gap\n• Validation loss increasing\n• Perfect training metrics",
        "• Add regularization\n• Reduce model complexity\n• Data augmentation\n• Early stopping"
    )
    
    capabilities.add_row(
        "Learning Rate Issues", 
        "• Oscillating loss\n• No convergence\n• Gradient explosion",
        "• LR scheduling\n• Gradient clipping\n• Optimizer tuning"
    )
    
    capabilities.add_row(
        "Convergence Problems",
        "• Slow/no progress\n• Plateau detection\n• Vanishing gradients",
        "• Initialization fixes\n• Architecture changes\n• Batch size tuning"
    )
    
    capabilities.add_row(
        "Training Instability",
        "• NaN losses\n• Sudden spikes\n• Mixed precision issues",
        "• Gradient clipping\n• Loss scaling\n• Numerical stability"
    )
    
    capabilities.add_row(
        "Hyperparameter Issues",
        "• Suboptimal settings\n• Poor combinations\n• Scale mismatches",
        "• Parameter tuning\n• Search strategies\n• Best practices"
    )
    
    console.print(capabilities)
    console.print()


def main():
    """Main demo function."""
    
    console.print(Panel.fit(
        "[bold magenta]🧪 Realistic ML Debugging Demo[/bold magenta]\n"
        "[dim]Demonstrating AI-powered experiment debugging with common ML issues[/dim]",
        border_style="magenta"
    ))
    
    console.print("\n[bold]This demo shows how our ClearML MCP + Smolagents system would debug real ML issues:[/bold]")
    console.print()
    
    # Show capabilities first
    show_debugging_capabilities()
    
    # Ask if user wants to see scenarios
    try:
        choice = console.input("[bold]Run debugging scenarios? [green][y][/green]es or [red][n][/red]o (default: yes): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "y"  # Default to yes
        console.print("[dim]Running scenarios (non-interactive execution)[/dim]")
    
    if choice.startswith('n'):
        console.print("[yellow]Demo skipped. The system is ready for real experiment debugging![/yellow]")
        return
    
    try:
        demonstrate_realistic_debugging()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Demo error: {str(e)}[/red]")


if __name__ == "__main__":
    main()