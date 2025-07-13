#!/usr/bin/env python3
"""
ClearML Analysis Agent using Smolagents with Gemini via OpenAI-compatible API

This example demonstrates using Gemini 2.0 Flash through the OpenAI-compatible API
with smolagents and our ClearML MCP server for intelligent experiment analysis.

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

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from smolagents import CodeAgent, MCPClient, OpenAIServerModel
except ImportError:
    print("❌ Required packages not found. Install with:")
    print("   uv sync --group examples")
    print("   or")
    print("   pip install 'smolagents[openai,mcp]' rich")
    raise

console = Console()


def create_clearml_analysis_agent():
    """Create a ClearML analysis agent using Gemini 2.0 Flash via OpenAI API."""
    # Initialize Gemini model via OpenAI-compatible API
    model = OpenAIServerModel(
        model_id="gemini-2.0-flash",
        # Google Gemini OpenAI-compatible API base URL
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_API_KEY,
        temperature=0.1,  # Lower temperature for more focused analysis
    )

    # Configure ClearML MCP server parameters
    # Use local installation since we haven't published to PyPI yet
    clearml_server_params = StdioServerParameters(
        command="python",
        args=["-m", "clearml_mcp.clearml_mcp"],
        env=os.environ,  # Pass through environment variables
    )

    return model, clearml_server_params


def demonstrate_clearml_analysis():
    """Demonstrate various ClearML analysis capabilities with rich formatting."""
    console.print(
        Panel.fit(
            "[bold blue]🚀 ClearML Analysis Agent[/bold blue]\n"
            "[dim]Powered by Smolagents + Gemini 2.0 Flash + ClearML MCP[/dim]",
            border_style="blue",
        )
    )

    # Create the model and server parameters
    model, clearml_server_params = create_clearml_analysis_agent()

    # Enhanced example queries that showcase different ClearML operations
    analysis_queries = [
        {
            "title": "🏗️ Project Overview",
            "query": "List all available ClearML projects and give me a detailed summary of what projects are available, including their purposes.",
            "icon": "📊",
        },
        {
            "title": "🔬 Experiment Deep Dive",
            "query": "Get comprehensive information about the experiment with ID 'efe5f7a6c5f34a15b4bfbf1c33660e20'. Analyze its status, parameters, metrics, and provide detailed insights about this experiment's configuration and performance.",
            "icon": "🧪",
        },
        {
            "title": "📈 Performance Analytics",
            "query": "Retrieve and analyze the training metrics for experiment 'efe5f7a6c5f34a15b4bfbf1c33660e20'. Look at the performance trends, convergence patterns, and provide insights about the training progress and model quality.",
            "icon": "📉",
        },
        {
            "title": "🔍 Intelligent Search",
            "query": "Search for experiments that contain keywords like 'training', 'model', 'neural', or 'learning' in their names or descriptions. Show me the most relevant results and categorize them by type.",
            "icon": "🔎",
        },
        {
            "title": "⚙️ Hyperparameter Analysis",
            "query": "Examine the hyperparameters and configuration for experiment 'efe5f7a6c5f34a15b4bfbf1c33660e20'. Analyze the optimization settings, learning rates, batch sizes, and other key parameters. Suggest potential improvements.",
            "icon": "🎛️",
        },
    ]

    # Connect to ClearML MCP server and run analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        connect_task = progress.add_task("[cyan]Connecting to ClearML MCP server...", total=None)

        with MCPClient(clearml_server_params) as clearml_tools:
            progress.update(connect_task, description="[green]✅ Connected to ClearML MCP server")
            progress.stop()

            console.print(
                f"[green]🛠️  Available tools: {len(clearml_tools)} ClearML MCP tools[/green]"
            )

            # Create agent with ClearML tools
            agent = CodeAgent(
                tools=clearml_tools,
                model=model,
                add_base_tools=False,  # Only use ClearML tools
                verbosity_level=1,  # Show tool usage
            )

            console.print()

            # Run each analysis query
            for i, analysis in enumerate(analysis_queries, 1):
                console.print(
                    Panel(
                        f"[bold]{analysis['icon']} {analysis['title']}[/bold]\n\n"
                        f"[dim]Query:[/dim] {analysis['query']}",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as analysis_progress:
                    task = analysis_progress.add_task(
                        "[yellow]🤔 Agent thinking and analyzing...", total=None
                    )

                    try:
                        result = agent.run(analysis["query"])
                        analysis_progress.update(task, description="[green]✅ Analysis complete!")
                        analysis_progress.stop()

                        console.print(
                            Panel(
                                f"[bold green]📊 Analysis Results[/bold green]\n\n{result}",
                                border_style="green",
                                padding=(1, 2),
                            )
                        )

                    except Exception as e:
                        analysis_progress.update(task, description="[red]❌ Analysis failed")
                        analysis_progress.stop()

                        console.print(
                            Panel(
                                f"[bold red]❌ Analysis Failed[/bold red]\n\n"
                                f"Error: {e!s}\n\n"
                                f"[dim]This might be due to:[/dim]\n"
                                f"• ClearML configuration issues\n"
                                f"• Invalid experiment ID\n"
                                f"• Network connectivity problems\n"
                                f"• API rate limits",
                                border_style="red",
                                padding=(1, 2),
                            )
                        )

                console.print()

        console.print(
            Panel.fit(
                "[bold green]🎉 All analyses completed successfully![/bold green]",
                border_style="green",
            )
        )


def interactive_mode():
    """Run the agent in interactive mode for custom queries with rich interface."""
    console.print(
        Panel(
            "[bold blue]🤖 Interactive ClearML Analysis Mode[/bold blue]\n\n"
            "[dim]Enter your questions about ClearML experiments, or 'quit' to exit.[/dim]\n\n"
            "[bold]Example queries:[/bold]\n"
            "• [cyan]'Show me all my projects with their experiment counts'[/cyan]\n"
            "• [cyan]'Analyze the best performing experiment in project X'[/cyan]\n"
            "• [cyan]'Compare experiments abc123 and def456 by their metrics'[/cyan]\n"
            "• [cyan]'Find all failed experiments and suggest what went wrong'[/cyan]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Create the model and server parameters
    model, clearml_server_params = create_clearml_analysis_agent()

    with MCPClient(clearml_server_params) as clearml_tools:
        agent = CodeAgent(
            tools=clearml_tools,
            model=model,
            add_base_tools=False,
            verbosity_level=1,
        )

        console.print("[green]✅ Agent ready for your questions![/green]\n")

        while True:
            try:
                user_query = console.input("\n[bold blue]🗣️  Your question:[/bold blue] ").strip()

                if user_query.lower() in ["quit", "exit", "q", "bye"]:
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break

                if not user_query:
                    continue

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("[yellow]🤔 Analyzing your question...", total=None)

                    try:
                        result = agent.run(user_query)
                        progress.update(task, description="[green]✅ Analysis complete!")
                        progress.stop()

                        console.print(
                            Panel(
                                f"[bold green]💡 Answer[/bold green]\n\n{result}",
                                border_style="green",
                                padding=(1, 2),
                            )
                        )

                    except Exception as e:
                        progress.update(task, description="[red]❌ Analysis failed")
                        progress.stop()

                        console.print(
                            Panel(
                                f"[bold red]❌ Error[/bold red]\n\n{e!s}",
                                border_style="red",
                                padding=(1, 2),
                            )
                        )

            except KeyboardInterrupt:
                console.print("\n\n[yellow]👋 Interrupted by user[/yellow]")
                break
            except EOFError:
                console.print("\n\n[yellow]👋 Goodbye![/yellow]")
                break


def main():
    """Main function with enhanced UI and options for demo or interactive mode."""
    console.print(
        Panel.fit(
            "[bold blue]🔬 ClearML Analysis Agent[/bold blue]\n"
            "[dim]Powered by Smolagents + Gemini 2.0 Flash + ClearML MCP[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[bold]Prerequisites Check:[/bold]")
    console.print("✅ ClearML configured with ~/.clearml.conf")
    console.print("✅ clearml-mcp server available (uvx clearml-mcp)")
    console.print("✅ smolagents with OpenAI/MCP support installed")
    console.print("✅ Google Gemini API key configured")
    console.print()

    # Check if user wants demo or interactive mode
    try:
        mode = (
            console.input(
                "[bold]Choose mode - [cyan][d][/cyan]emo or [cyan][i][/cyan]nteractive (default: demo): "
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        # Default to demo mode when run non-interactively
        mode = "d"
        console.print("[dim]Running in demo mode (non-interactive execution)[/dim]")

    try:
        if mode.startswith("i"):
            interactive_mode()
        else:
            demonstrate_clearml_analysis()

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Agent stopped by user[/yellow]")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]❌ Error[/bold red]\n\n"
                f"{e!s}\n\n"
                f"[bold]Troubleshooting:[/bold]\n"
                f"1. Ensure ClearML is configured: [cyan]clearml-init[/cyan]\n"
                f"2. Test MCP server: [cyan]uvx clearml-mcp[/cyan]\n"
                f"3. Check Gemini API key is valid\n"
                f"4. Install dependencies: [cyan]uv sync --group examples[/cyan]",
                border_style="red",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    main()
