#!/usr/bin/env python3
"""
Find Real ClearML Experiments

This script helps find actual experiment IDs in your ClearML instance
so we can debug real experiments instead of fake ones.
"""

import os

from mcp import StdioServerParameters

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyDAdEToKdFt8SHs25ABz65bx6cedU_zreo"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from smolagents import CodeAgent, MCPClient, OpenAIServerModel
except ImportError:
    print("❌ Required packages not found. Install with: uv sync --group examples")
    raise

console = Console()


def find_experiments():
    """Find real experiments in the ClearML instance."""
    console.print(
        Panel.fit(
            "[bold blue]🔍 Finding Real ClearML Experiments[/bold blue]\n"
            "[dim]Let's discover what experiments actually exist[/dim]",
            border_style="blue",
        )
    )

    # Initialize Gemini model
    model = OpenAIServerModel(
        model_id="gemini-2.0-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

    # Configure ClearML MCP server
    clearml_server_params = StdioServerParameters(
        command="python",
        args=["-m", "clearml_mcp.clearml_mcp"],
        env=os.environ,
    )

    with MCPClient(clearml_server_params) as clearml_tools:
        agent = CodeAgent(
            tools=clearml_tools,
            model=model,
            add_base_tools=False,
            verbosity_level=1,
        )

        query = """
        Find real ClearML experiments by doing the following:

        1. List all projects and find ones that might have experiments
        2. For the main projects (not subprojects), try to list tasks
        3. If you find any real experiments, show me their IDs, names, and status
        4. Focus on finding at least 1-3 real experiment IDs that we can debug

        I need actual experiment IDs that exist, not example ones.
        """

        console.print("\n[yellow]🤔 Searching for real experiments...[/yellow]")

        try:
            result = agent.run(query)
            console.print(
                Panel(
                    f"[bold green]🎯 Found Experiments[/bold green]\n\n{result}",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        except Exception as e:
            console.print(f"[red]❌ Search failed: {e!s}[/red]")


if __name__ == "__main__":
    find_experiments()
