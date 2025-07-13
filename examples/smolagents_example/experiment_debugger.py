#!/usr/bin/env python3
"""
ClearML Experiment Debugger using Smolagents with Gemini API

This example demonstrates a realistic ML debugging scenario where we analyze
a specific ClearML experiment to determine if it's performing well and identify
potential issues or improvements.

Features:
- Comprehensive experiment health analysis
- Performance trend analysis
- Hyperparameter evaluation
- Training issue detection
- Actionable recommendations

Requirements:
- smolagents with OpenAI and MCP support
- clearml-mcp server (this project)  
- Google Gemini API access
- ClearML configuration (~/.clearml.conf)
"""

import os
from mcp import StdioServerParameters

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyDAdEToKdFt8SHs25ABz65bx6cedU_zreo"

# Your actual experiment ID
EXPERIMENT_ID = "efe5f7a6c5f34a15b4bfbf1c33660e20"

try:
    from smolagents import MCPClient, CodeAgent
    from smolagents import OpenAIServerModel
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
except ImportError:
    print("❌ Required packages not found. Install with:")
    print("   uv sync --group examples")
    print("   or")
    print("   pip install 'smolagents[openai,mcp]' rich")
    raise

console = Console()


def create_experiment_debugger():
    """Create an ML experiment debugging agent using Gemini 2.0 Flash."""
    
    # Initialize Gemini model via OpenAI-compatible API
    model = OpenAIServerModel(
        model_id="gemini-2.0-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_API_KEY,
        temperature=0.2  # Slightly higher for creative debugging insights
    )
    
    # Configure ClearML MCP server parameters
    clearml_server_params = StdioServerParameters(
        command="python",
        args=["-m", "clearml_mcp.clearml_mcp"],
        env=os.environ
    )
    
    return model, clearml_server_params


def debug_experiment():
    """Perform comprehensive debugging analysis of the specified experiment."""
    
    console.print(Panel.fit(
        f"[bold red]🐛 ClearML Experiment Debugger[/bold red]\n"
        f"[dim]Analyzing experiment: {EXPERIMENT_ID}[/dim]",
        border_style="red"
    ))
    
    # Create the model and server parameters
    model, clearml_server_params = create_experiment_debugger()
    
    # Debugging analysis queries designed to identify real issues
    debugging_queries = [
        {
            "title": "🔍 Experiment Health Check",
            "query": f"""
            Analyze experiment '{EXPERIMENT_ID}' comprehensively to determine if it's performing well.
            
            Please examine:
            1. Get the basic experiment info (status, type, project)
            2. Get all training metrics and analyze trends
            3. Get hyperparameters and evaluate if they're reasonable
            4. Get any artifacts to understand outputs
            
            Then provide a detailed health assessment:
            - Is the experiment successful or failing?
            - Are the metrics trending in the right direction?
            - Are there signs of overfitting, underfitting, or other issues?
            - Are the hyperparameters reasonable?
            - What's the overall training quality?
            
            Give me specific insights and red flags if any exist.
            """,
            "icon": "🩺"
        },
        {
            "title": "📈 Performance Trend Analysis",
            "query": f"""
            Focus specifically on the training metrics for experiment '{EXPERIMENT_ID}'.
            
            Analyze the performance trends and identify:
            1. Is the loss decreasing properly?
            2. Are there signs of convergence or divergence?
            3. Is the training stable or erratic?
            4. Are there sudden spikes or drops in metrics?
            5. How many iterations/epochs did it run?
            6. Did training complete successfully or stop early?
            
            Look for common training issues like:
            - Learning rate too high/low
            - Gradient explosion/vanishing
            - Overfitting patterns
            - Training instability
            - Poor convergence
            
            Provide specific numerical evidence from the metrics.
            """,
            "icon": "📊"
        },
        {
            "title": "⚙️ Hyperparameter Analysis",
            "query": f"""
            Examine the hyperparameters for experiment '{EXPERIMENT_ID}' and evaluate their appropriateness.
            
            Look at parameters like:
            - Learning rate and optimization settings
            - Batch size and training configuration
            - Model architecture parameters
            - Regularization settings
            - Any custom parameters
            
            Assess:
            1. Are these parameters reasonable for this type of experiment?
            2. Do you see any obvious misconfigurations?
            3. What hyperparameters might be causing performance issues?
            4. What changes would you recommend to improve performance?
            
            Be specific about which parameters look problematic and why.
            """,
            "icon": "🎛️"
        },
        {
            "title": "🚨 Issue Detection & Recommendations",
            "query": f"""
            Based on all the information gathered about experiment '{EXPERIMENT_ID}', provide:
            
            1. **Critical Issues Found**: List any serious problems that need immediate attention
            2. **Warning Signs**: Identify concerning patterns that might lead to problems
            3. **Performance Bottlenecks**: What's limiting the experiment's success?
            4. **Actionable Recommendations**: Specific steps to improve the experiment
            5. **Next Steps**: What should the researcher do next?
            
            Format this as a debugging report that would help a data scientist understand:
            - What went wrong (if anything)
            - Why it went wrong
            - How to fix it
            - How to prevent similar issues
            
            Be practical and specific in your recommendations.
            """,
            "icon": "🔧"
        }
    ]
    
    # Connect to ClearML MCP server and run debugging analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        connect_task = progress.add_task("[cyan]Connecting to ClearML MCP server...", total=None)
        
        with MCPClient(clearml_server_params) as clearml_tools:
            progress.update(connect_task, description="[green]✅ Connected to ClearML MCP server")
            progress.stop()
            
            console.print(f"[green]🛠️  Available diagnostic tools: {len(clearml_tools)} ClearML MCP tools[/green]")
            
            # Create debugging agent
            agent = CodeAgent(
                tools=clearml_tools,
                model=model,
                add_base_tools=False,  # Only use ClearML tools for focused analysis
                verbosity_level=2      # Show detailed tool usage for debugging
            )
            
            console.print()
            
            # Run each debugging analysis
            for i, analysis in enumerate(debugging_queries, 1):
                console.print(Panel(
                    f"[bold red]{analysis['icon']} {analysis['title']}[/bold red]\n\n"
                    f"[dim]Analysis Focus:[/dim] {analysis['query'][:200]}...",
                    border_style="red",
                    padding=(1, 2)
                ))
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as analysis_progress:
                    
                    task = analysis_progress.add_task("[yellow]🤔 Debugging agent analyzing...", total=None)
                    
                    try:
                        result = agent.run(analysis['query'])
                        analysis_progress.update(task, description="[green]✅ Analysis complete!")
                        analysis_progress.stop()
                        
                        console.print(Panel(
                            f"[bold green]🩺 Diagnostic Results[/bold green]\n\n{result}",
                            border_style="green",
                            padding=(1, 2)
                        ))
                        
                    except Exception as e:
                        analysis_progress.update(task, description="[red]❌ Analysis failed")
                        analysis_progress.stop()
                        
                        console.print(Panel(
                            f"[bold red]❌ Debugging Failed[/bold red]\n\n"
                            f"Error: {str(e)}\n\n"
                            f"[dim]This might indicate:[/dim]\n"
                            f"• Experiment ID doesn't exist\n"
                            f"• Network connectivity issues\n"
                            f"• ClearML permission problems\n"
                            f"• API rate limiting",
                            border_style="red",
                            padding=(1, 2)
                        ))
                
                console.print()
        
        # Summary
        console.print(Panel.fit(
            f"[bold green]🎯 Debugging Analysis Complete![/bold green]\n"
            f"[dim]Experiment {EXPERIMENT_ID} has been thoroughly analyzed[/dim]",
            border_style="green"
        ))


def quick_health_check():
    """Perform a quick health check of the experiment."""
    
    console.print(Panel.fit(
        f"[bold blue]⚡ Quick Health Check[/bold blue]\n"
        f"[dim]Rapid assessment of experiment: {EXPERIMENT_ID}[/dim]",
        border_style="blue"
    ))
    
    model, clearml_server_params = create_experiment_debugger()
    
    quick_query = f"""
    Perform a rapid health check on experiment '{EXPERIMENT_ID}':
    
    1. Get basic info and check if experiment completed successfully
    2. Get metrics and do a quick trend analysis
    3. Provide a simple verdict: Is this experiment HEALTHY, CONCERNING, or FAILED?
    
    Give me a concise summary in 3-5 bullet points about the experiment's health.
    """
    
    with MCPClient(clearml_server_params) as clearml_tools:
        agent = CodeAgent(
            tools=clearml_tools,
            model=model,
            add_base_tools=False,
            verbosity_level=1
        )
        
        console.print("\n[yellow]🤔 Analyzing...[/yellow]")
        
        try:
            result = agent.run(quick_query)
            console.print(Panel(
                f"[bold blue]⚡ Quick Assessment[/bold blue]\n\n{result}",
                border_style="blue",
                padding=(1, 2)
            ))
        except Exception as e:
            console.print(f"[red]❌ Quick check failed: {str(e)}[/red]")


def main():
    """Main function with options for different types of analysis."""
    
    console.print(Panel.fit(
        "[bold magenta]🔬 ClearML Experiment Debugger[/bold magenta]\n"
        "[dim]Realistic ML debugging with Smolagents + Gemini + ClearML MCP[/dim]",
        border_style="magenta"
    ))
    
    # Show experiment details
    console.print(f"\n[bold]🎯 Target Experiment:[/bold] [cyan]{EXPERIMENT_ID}[/cyan]")
    console.print("[dim]This tool will analyze your experiment to identify issues and provide recommendations[/dim]")
    console.print()
    
    # Analysis options
    console.print("[bold]Available Analysis Types:[/bold]")
    console.print("1. [green]Full Debugging Analysis[/green] - Comprehensive 4-step diagnostic")
    console.print("2. [blue]Quick Health Check[/blue] - Rapid assessment")
    console.print()
    
    try:
        choice = console.input("[bold]Choose analysis type - [green][1][/green] Full or [blue][2][/blue] Quick (default: 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"  # Default to full analysis
        console.print("[dim]Running full analysis (non-interactive execution)[/dim]")
    
    try:
        if choice == "2":
            quick_health_check()
        else:
            debug_experiment()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Debugging interrupted by user[/yellow]")
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Debugging Error[/bold red]\n\n"
            f"{str(e)}\n\n"
            f"[bold]Troubleshooting:[/bold]\n"
            f"1. Verify experiment ID exists: [cyan]{EXPERIMENT_ID}[/cyan]\n"
            f"2. Check ClearML connection: [cyan]clearml-task --help[/cyan]\n"
            f"3. Test MCP server: [cyan]python -m clearml_mcp.clearml_mcp[/cyan]\n"
            f"4. Verify Gemini API access",
            border_style="red",
            padding=(1, 2)
        ))


if __name__ == "__main__":
    main()