#!/usr/bin/env python3
"""
ClearML MCP Demo Script
Demonstrates real-time analysis of ClearML experiment data using MCP tools.
"""

import os
import time

from mcp import StdioServerParameters
from rich.console import Console
from rich.panel import Panel
from smolagents import CodeAgent, MCPClient, OpenAIServerModel

console = Console()


# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("dotenv package not found, skipping.")

# Set up Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")


class ClearMLDemo:
    """Demo class for ClearML MCP integration."""

    def __init__(self):
        """Initialize the demo with OpenAI client and ClearML MCP connection."""
        self.model = OpenAIServerModel(
            model_id="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY,
            temperature=0.1,
        )
        self.clearml_server_params = StdioServerParameters(
            command="python",
            args=["-m", "clearml_mcp.clearml_mcp"],
            env=os.environ,
        )
        self.experiment_id = None

    def run_demo(self):
        """Run the complete demo workflow."""
        # Show demo introduction
        console.print(
            Panel.fit(
                "[bold blue]🎯 ClearML MCP Demo[/bold blue]\n"
                "[dim]Real-time analysis of ClearML experiment data[/dim]",
                border_style="blue",
            )
        )
        console.print()

        # Step 1: Find the experiment
        self._find_experiment()

        # Step 2: Analyze the experiment
        self._analyze_experiment()

        # Demo complete
        console.print()
        console.print(
            Panel.fit(
                "[bold green]✅ Demo Complete![/bold green]\n"
                "[dim]ClearML MCP analysis finished successfully[/dim]",
                border_style="green",
            )
        )

        # Final pause for demo wrap-up
        if os.getenv("DEMO_MODE") == "1":
            console.print("[dim]⏱️  Demo completed - thank you for watching![/dim]")
            time.sleep(3)

    def _find_experiment(self):
        """Find and validate the target experiment."""
        console.print("[yellow]📍 Step 1: Finding target experiment...[/yellow]")

        # Add demo pause
        if os.getenv("DEMO_MODE") == "1":
            console.print("[dim]⏱️  Pausing for demo readability...[/dim]")
            time.sleep(4)

        target_experiment_name = "PyTorch with TensorBoard"

        find_query = f"""
        Discover and validate a ClearML experiment for analysis:

        TARGET: Find a PyTorch experiment with TensorBoard logging in ClearML examples

        DISCOVERY STRATEGY:
        1. Use find_project_by_pattern with pattern "Pytorch" to find PyTorch-related projects
        2. Look for projects that contain "Frameworks" and "Pytorch" in the name
        3. Once you find the right project, use find_experiment_in_project with pattern "TensorBoard"
        4. Find the experiment named "{target_experiment_name}"
        5. Get the experiment ID, status, and verify it has scalar metrics
        6. If dynamic discovery fails, fall back to known experiment ID: e-efe5f7a6c5f34a15b4bfbf1c33660e20

        TOOLS TO USE:
        - find_project_by_pattern(pattern="Pytorch")
        - find_experiment_in_project(project_name=<found_project>, experiment_pattern="TensorBoard")
        - get_task_info(task_id=<found_experiment_id>) to validate

        IMPORTANT: Start your response with "EXPERIMENT_ID: [the actual ID]" followed by your discovery process and validation.
        """

        # Print the query
        console.print(
            Panel(
                f"[bold blue]📤 Query to LLM[/bold blue]\n\n[dim]{find_query.strip()}[/dim]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Give viewers time to read the query
        if os.getenv("DEMO_MODE") == "1":
            console.print("[dim]⏱️  Allowing time to read query...[/dim]")
            time.sleep(3)

        with MCPClient(self.clearml_server_params) as clearml_tools:
            agent = CodeAgent(
                tools=clearml_tools,
                model=self.model,
                add_base_tools=False,
                verbosity_level=1,
                max_steps=5,
            )

            try:
                result = agent.run(find_query)
                console.print(
                    Panel(
                        f"[bold green]🔍 Experiment Discovery[/bold green]\n\n{result}",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

                # Extract experiment ID from the result
                if "EXPERIMENT_ID:" in result:
                    lines = result.split("\n")
                    for line in lines:
                        if line.strip().startswith("EXPERIMENT_ID:"):
                            self.experiment_id = line.split("EXPERIMENT_ID:")[1].strip()
                            break

                # Fallback if experiment ID extraction failed
                if not self.experiment_id:
                    console.print(
                        "[yellow]⚠️  Search had issues - using known experiment ID for demo[/yellow]"
                    )
                    console.print(
                        "[cyan]💡 In production, you'd retry the search or browse the project manually[/cyan]"
                    )
                    self.experiment_id = "e-efe5f7a6c5f34a15b4bfbf1c33660e20"
                    console.print(f"[green]✅ Using experiment: {self.experiment_id}[/green]")

                # Add demo pause
                if os.getenv("DEMO_MODE") == "1":
                    console.print("[dim]⏱️  Allowing time to read discovery results...[/dim]")
                    time.sleep(6)

            except Exception as e:
                console.print(f"[red]❌ Experiment discovery failed: {e!s}[/red]")

    def _analyze_experiment(self):
        """Analyze the experiment's scalar convergence patterns."""
        console.print("[yellow]📊 Step 2: Analyzing scalar convergence patterns...[/yellow]")

        # Check if we have an experiment ID from the discovery step
        if not self.experiment_id:
            console.print("[red]❌ No experiment ID found from discovery step[/red]")
            return

        # Add demo pause
        if os.getenv("DEMO_MODE") == "1":
            console.print("[dim]⏱️  Pausing for demo readability...[/dim]")
            time.sleep(4)

        console.print(f"[cyan]🔍 Using discovered experiment ID: {self.experiment_id}[/cyan]")

        # Add a pause to show the discovered ID
        if os.getenv("DEMO_MODE") == "1":
            console.print("[dim]⏱️  Setting up analysis workflow...[/dim]")
            time.sleep(3)

        analysis_query = f"""
        Analyze this ClearML experiment's scalar convergence patterns:

        EXPERIMENT ID: {self.experiment_id}

        DETAILED ANALYSIS TASKS:
        1. Use ClearML MCP tools to retrieve all scalar metrics from this experiment
        2. Extract training and validation metrics (loss, accuracy, etc.)
        3. Calculate convergence rates from the retrieved values
        4. Identify convergence quality (good/concerning/poor)
        5. Check for overfitting signs in training vs validation metrics
        6. Assess learning rate appropriateness based on convergence patterns
        7. Determine optimal stopping point from the data
        8. Provide specific, actionable recommendations

        Focus on numerical evidence and actionable insights from the REAL experiment data.
        Show your work with calculations and specific metric values.
        """

        # Print the query
        console.print(
            Panel(
                f"[bold blue]📤 Query to LLM[/bold blue]\n\n[dim]{analysis_query.strip()}[/dim]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Give viewers time to read the analysis query
        if os.getenv("DEMO_MODE") == "1":
            console.print("[dim]⏱️  Allowing time to read analysis query...[/dim]")
            time.sleep(4)

        with MCPClient(self.clearml_server_params) as clearml_tools:
            agent = CodeAgent(
                tools=clearml_tools,
                model=self.model,
                add_base_tools=False,
                verbosity_level=1,
                max_steps=10,
            )

            try:
                result = agent.run(analysis_query)
                console.print(
                    Panel(
                        f"[bold green]📊 Convergence Analysis[/bold green]\n\n{result}",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

                # Add demo pause
                if os.getenv("DEMO_MODE") == "1":
                    console.print("[dim]⏱️  Allowing time to read detailed analysis...[/dim]")
                    time.sleep(8)

            except Exception as e:
                console.print(f"[red]❌ Analysis failed: {e!s}[/red]")


def main():
    """Main demo function."""
    demo = ClearMLDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
