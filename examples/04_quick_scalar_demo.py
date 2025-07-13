#!/usr/bin/env python3
"""
Quick Scalar Analysis Demo

Show scalar convergence analysis without long searches.
Focus on analyzing realistic training patterns and convergence.
"""

import os

from mcp import StdioServerParameters

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyDAdEToKdFt8SHs25ABz65bx6cedU_zreo"

try:
    from rich.console import Console
    from rich.panel import Panel
    from smolagents import CodeAgent, MCPClient, OpenAIServerModel
except ImportError:
    print("❌ Required packages not found. Install with: uv sync --group examples")
    raise

console = Console()


def quick_scalar_analysis():
    """Quick demonstration of scalar pattern analysis."""
    console.print(
        Panel.fit(
            "[bold blue]📊 Quick Scalar Convergence Analysis[/bold blue]\n"
            "[dim]Analyzing training patterns for convergence debugging[/dim]",
            border_style="blue",
        )
    )

    model = OpenAIServerModel(
        model_id="gemini-2.0-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

    clearml_server_params = StdioServerParameters(
        command="python",
        args=["-m", "clearml_mcp.clearml_mcp"],
        env=os.environ,
    )

    # Quick scalar analysis scenario
    analysis_query = """
    Analyze this training scenario with specific scalar data:

    EXPERIMENT SCALARS:
    Training Loss: [2.3, 1.8, 1.4, 1.1, 0.9, 0.8, 0.7, 0.65, 0.62, 0.60]
    Validation Loss: [2.4, 1.9, 1.5, 1.2, 1.0, 0.85, 0.75, 0.72, 0.70, 0.68]
    Training Accuracy: [0.2, 0.35, 0.48, 0.58, 0.66, 0.72, 0.77, 0.81, 0.83, 0.85]
    Validation Accuracy: [0.18, 0.32, 0.45, 0.55, 0.63, 0.69, 0.74, 0.78, 0.80, 0.82]
    Learning Rate: [0.001] * 10 (fixed)
    Epochs: 10

    CONVERGENCE ANALYSIS:
    1. Calculate the convergence rate from the loss values
    2. Identify if this shows good, concerning, or poor convergence
    3. Check for overfitting signs in train vs validation metrics
    4. Assess if the learning rate is appropriate
    5. Determine optimal stopping point
    6. Provide specific recommendations based on the scalar trends

    Focus on the numerical evidence and trends in the data.
    """

    with MCPClient(clearml_server_params) as clearml_tools:
        agent = CodeAgent(
            tools=clearml_tools,
            model=model,
            add_base_tools=False,
            verbosity_level=0,
            max_steps=1,  # Just one analysis step
        )

        console.print("\n[yellow]📈 Analyzing scalar convergence patterns...[/yellow]")

        try:
            result = agent.run(analysis_query)

            console.print(
                Panel(
                    f"[bold green]📊 Scalar Convergence Analysis[/bold green]\n\n{result}",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        except Exception as e:
            if "429" in str(e):
                console.print(
                    Panel(
                        "[bold yellow]⏳ Rate Limited[/bold yellow]\n\n"
                        "Analysis would show:\n\n"
                        "[bold]Convergence Assessment: GOOD[/bold]\n"
                        "• Training loss: Smooth decrease (2.3 → 0.60)\n"
                        "• Validation loss: Following training (2.4 → 0.68)\n"
                        "• Small gap: 0.60 vs 0.68 (healthy)\n"
                        "• Convergence rate: ~74% improvement\n"
                        "• Learning rate: 0.001 appears optimal\n"
                        "• Recommendation: Continue 2-3 more epochs",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )
            else:
                console.print(f"[red]❌ Analysis failed: {e!s}[/red]")


if __name__ == "__main__":
    quick_scalar_analysis()
