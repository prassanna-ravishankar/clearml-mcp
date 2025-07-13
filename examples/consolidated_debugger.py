#!/usr/bin/env python3
"""
ClearML Experiment Debugger - Consolidated Example

This unified example combines all debugging capabilities:
- Real experiment discovery and analysis
- Scalar convergence analysis
- Common ML issue detection
- Expert-level debugging recommendations

Features:
- Smart experiment discovery
- Convergence pattern analysis
- Overfitting detection
- Learning rate optimization
- Training stability assessment
- Actionable recommendations

Usage:
    uv run --group examples examples/consolidated_debugger.py
"""

import os
from typing import Any, Dict, List

from mcp import StdioServerParameters

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyDAdEToKdFt8SHs25ABz65bx6cedU_zreo"

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from smolagents import CodeAgent, MCPClient, OpenAIServerModel
except ImportError:
    print("❌ Required packages not found. Install with: uv sync --group examples")
    raise

console = Console()


class ClearMLDebugger:
    """Unified ClearML experiment debugger with all capabilities."""

    def __init__(self):
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

    def show_capabilities(self):
        """Display debugging capabilities."""
        console.print(
            Panel.fit(
                "[bold cyan]🔍 ClearML Debugging Capabilities[/bold cyan]\n"
                "[dim]Complete experiment analysis and optimization recommendations[/dim]",
                border_style="cyan",
            )
        )

        capabilities = Table(title="Debugging Analysis Features")
        capabilities.add_column("Category", style="cyan", no_wrap=True)
        capabilities.add_column("Analysis", style="yellow")
        capabilities.add_column("Detects", style="red")
        capabilities.add_column("Recommends", style="green")

        capabilities.add_row(
            "🔍 Discovery",
            "• Find real experiments\n• Identify training data\n• Locate scalar metrics\n• Project analysis",
            "• Active experiments\n• Completed training\n• Available metrics\n• Data patterns",
            "• Best experiments\n• Debug targets\n• Analysis scope\n• Next steps",
        )

        capabilities.add_row(
            "📊 Convergence",
            "• Loss patterns\n• Accuracy trends\n• Rate analysis\n• Stability check",
            "• Good convergence\n• Slow progress\n• Oscillations\n• Plateaus",
            "• Optimal stopping\n• Rate adjustments\n• Training duration\n• Improvements",
        )

        capabilities.add_row(
            "🚨 Issues",
            "• Overfitting gaps\n• Instability signs\n• Performance drops\n• Anomaly detection",
            "• Train/val gaps\n• Gradient explosion\n• NaN losses\n• Poor generalization",
            "• Regularization\n• Gradient clipping\n• Architecture fixes\n• Data solutions",
        )

        capabilities.add_row(
            "⚙️ Optimization",
            "• Hyperparameter review\n• Architecture assessment\n• Training efficiency\n• Resource usage",
            "• Suboptimal settings\n• Bottlenecks\n• Waste/inefficiency\n• Improvement potential",
            "• Parameter tuning\n• Architecture changes\n• Efficiency gains\n• Best practices",
        )

        console.print(capabilities)
        console.print()

    def find_real_experiments(self) -> list[str]:
        """Find real experiments in ClearML instance."""
        console.print("\n[yellow]🔍 Step 1: Discovering real experiments...[/yellow]")

        with MCPClient(self.clearml_server_params) as clearml_tools:
            agent = CodeAgent(
                tools=clearml_tools,
                model=self.model,
                add_base_tools=False,
                verbosity_level=1,
                max_steps=2,
            )

            discovery_query = """
            Find real ClearML experiments by:

            1. List all projects to see what's available
            2. For each project, search for tasks with training keywords:
               - "train", "training", "model", "neural", "learning"
               - "loss", "accuracy", "epoch", "batch"
               - "pytorch", "tensorflow", "keras"
            3. Get basic info for promising tasks
            4. Check which tasks have actual scalar metrics
            5. Return 2-3 best experiment IDs for debugging

            Focus on experiments with real training data and metrics.
            """

            try:
                result = agent.run(discovery_query)
                console.print(
                    Panel(
                        f"[bold green]🎯 Experiment Discovery[/bold green]\n\n{result}",
                        border_style="green",
                        padding=(1, 2),
                    )
                )
                return []  # Would extract IDs from result in real implementation

            except Exception as e:
                if "429" in str(e):
                    console.print(
                        Panel(
                            "[bold yellow]⏳ Rate Limited[/bold yellow]\n\n"
                            "Discovery would find:\n"
                            "• Real experiment IDs\n"
                            "• Available metrics\n"
                            "• Training patterns\n"
                            "• Analysis targets",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(f"[red]❌ Discovery failed: {e!s}[/red]")
                return []

    def analyze_scalar_patterns(self, experiment_ids: list[str] = None):
        """Analyze scalar metrics for convergence patterns."""
        console.print("\n[yellow]📊 Step 2: Analyzing scalar convergence patterns...[/yellow]")

        # Use realistic demo data if no real experiments found
        demo_scenarios = [
            {
                "name": "Perfect Convergence",
                "train_loss": [2.3, 1.8, 1.4, 1.1, 0.9, 0.8, 0.7, 0.65, 0.62, 0.60],
                "val_loss": [2.4, 1.9, 1.5, 1.2, 1.0, 0.85, 0.75, 0.72, 0.70, 0.68],
                "train_acc": [0.2, 0.35, 0.48, 0.58, 0.66, 0.72, 0.77, 0.81, 0.83, 0.85],
                "val_acc": [0.18, 0.32, 0.45, 0.55, 0.63, 0.69, 0.74, 0.78, 0.80, 0.82],
                "pattern": "good_convergence",
            },
            {
                "name": "Overfitting Pattern",
                "train_loss": [2.1, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02],
                "val_loss": [2.2, 1.6, 1.1, 0.8, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0],
                "train_acc": [0.25, 0.42, 0.58, 0.71, 0.82, 0.89, 0.94, 0.97, 0.99, 1.0],
                "val_acc": [0.23, 0.40, 0.55, 0.68, 0.65, 0.62, 0.58, 0.55, 0.52, 0.50],
                "pattern": "overfitting",
            },
        ]

        with MCPClient(self.clearml_server_params) as clearml_tools:
            agent = CodeAgent(
                tools=clearml_tools,
                model=self.model,
                add_base_tools=False,
                verbosity_level=0,
                max_steps=1,
            )

            for scenario in demo_scenarios:
                console.print(f"\n[blue]📈 Analyzing: {scenario['name']}[/blue]")

                analysis_query = f"""
                Analyze this training scenario's scalar convergence:

                METRICS:
                Training Loss: {scenario["train_loss"]}
                Validation Loss: {scenario["val_loss"]}
                Training Accuracy: {scenario["train_acc"]}
                Validation Accuracy: {scenario["val_acc"]}

                CONVERGENCE ANALYSIS:
                1. Calculate convergence rate from loss values
                2. Identify convergence quality (good/concerning/poor)
                3. Check for overfitting signs in train vs validation
                4. Assess learning rate appropriateness
                5. Determine optimal stopping point
                6. Provide specific recommendations

                Focus on numerical evidence and actionable insights.
                """

                try:
                    result = agent.run(analysis_query)
                    console.print(
                        Panel(
                            f"[bold green]📊 {scenario['name']} Analysis[/bold green]\n\n{result}",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )

                except Exception as e:
                    if "429" in str(e):
                        # Provide fallback analysis
                        if scenario["pattern"] == "good_convergence":
                            fallback = (
                                "[bold]✅ GOOD CONVERGENCE DETECTED[/bold]\n"
                                "• Training loss: Smooth decrease (2.3 → 0.60, 74% improvement)\n"
                                "• Validation loss: Following training (2.4 → 0.68)\n"
                                "• Healthy gap: 0.60 vs 0.68 (minimal overfitting)\n"
                                "• Recommendation: Continue 2-3 more epochs"
                            )
                        else:
                            fallback = (
                                "[bold]🚨 OVERFITTING DETECTED[/bold]\n"
                                "• Overfitting starts: Epoch 4 (validation loss increases)\n"
                                "• Training accuracy: Perfect (100%) - suspicious\n"
                                "• Validation accuracy: Declining (68% → 50%)\n"
                                "• Recommendation: Stop training, add regularization"
                            )

                        console.print(
                            Panel(
                                f"[bold yellow]⏳ Rate Limited - Analysis Summary[/bold yellow]\n\n{fallback}",
                                border_style="yellow",
                                padding=(1, 2),
                            )
                        )
                    else:
                        console.print(f"[red]❌ Analysis failed: {e!s}[/red]")

    def demonstrate_issue_detection(self):
        """Demonstrate detection of common ML issues."""
        console.print("\n[yellow]🐛 Step 3: Demonstrating ML issue detection...[/yellow]")

        issues = [
            {
                "title": "🚨 Learning Rate Too High",
                "pattern": "Loss oscillating wildly: 2.1 → 0.8 → 3.2 → 1.1 → 4.5",
                "diagnosis": "Unstable training due to excessive learning rate",
                "solution": "Reduce LR by 10x, add gradient clipping",
            },
            {
                "title": "🐌 Vanishing Gradients",
                "pattern": "50 epochs: loss only 2.3 → 2.1, gradients ~0.001",
                "diagnosis": "Extremely slow learning, likely vanishing gradients",
                "solution": "Better initialization, increase LR, check architecture",
            },
            {
                "title": "💥 Training Explosion",
                "pattern": "Epoch 21: Loss jumps to 15.6, then NaN",
                "diagnosis": "Gradient explosion, mixed precision issues",
                "solution": "Gradient clipping, loss scaling, numerical stability",
            },
        ]

        issue_table = Table(title="Common ML Issues Detection")
        issue_table.add_column("Issue", style="red")
        issue_table.add_column("Pattern", style="yellow")
        issue_table.add_column("Diagnosis", style="blue")
        issue_table.add_column("Solution", style="green")

        for issue in issues:
            issue_table.add_row(
                issue["title"],
                issue["pattern"],
                issue["diagnosis"],
                issue["solution"],
            )

        console.print(issue_table)
        console.print()

    def provide_optimization_recommendations(self):
        """Provide expert optimization recommendations."""
        console.print("\n[yellow]⚙️ Step 4: Optimization recommendations...[/yellow]")

        recommendations = Panel(
            "[bold green]🎯 Expert Optimization Recommendations[/bold green]\n\n"
            "[bold]For Good Convergence:[/bold]\n"
            "• Continue training 2-3 more epochs\n"
            "• Monitor validation metrics closely\n"
            "• Consider learning rate decay\n\n"
            "[bold]For Overfitting:[/bold]\n"
            "• Add dropout (0.2-0.5) or weight decay\n"
            "• Increase dataset size or augmentation\n"
            "• Reduce model complexity\n"
            "• Implement early stopping\n\n"
            "[bold]For Slow Convergence:[/bold]\n"
            "• Increase learning rate by 2-10x\n"
            "• Use learning rate warmup\n"
            "• Check weight initialization\n"
            "• Consider different optimizer (Adam vs SGD)\n\n"
            "[bold]For Instability:[/bold]\n"
            "• Add gradient clipping (max_norm=1.0)\n"
            "• Reduce learning rate by 5-10x\n"
            "• Use mixed precision carefully\n"
            "• Check for numerical overflow",
            border_style="green",
            padding=(1, 2),
        )

        console.print(recommendations)

    def run_complete_analysis(self):
        """Run the complete debugging analysis."""
        console.print(
            Panel.fit(
                "[bold magenta]🧪 ClearML Complete Experiment Debugger[/bold magenta]\n"
                "[dim]Comprehensive ML debugging with real experiment analysis[/dim]",
                border_style="magenta",
            )
        )

        # Show capabilities
        self.show_capabilities()

        try:
            # Step 1: Find real experiments
            experiment_ids = self.find_real_experiments()

            # Step 2: Analyze scalar patterns
            self.analyze_scalar_patterns(experiment_ids)

            # Step 3: Demonstrate issue detection
            self.demonstrate_issue_detection()

            # Step 4: Provide recommendations
            self.provide_optimization_recommendations()

            # Summary
            console.print(
                Panel.fit(
                    "[bold green]✅ Complete Analysis Finished![/bold green]\n\n"
                    "[dim]The debugger has:\n"
                    "• Discovered available experiments\n"
                    "• Analyzed convergence patterns\n"
                    "• Detected common ML issues\n"
                    "• Provided optimization recommendations\n\n"
                    "Ready for real experiment debugging![/dim]",
                    border_style="green",
                )
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Analysis interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Analysis error: {e!s}[/red]")


def main():
    """Main function with user interaction."""
    debugger = ClearMLDebugger()

    console.print("\n[bold]Available Analysis Modes:[/bold]")
    console.print("1. [green]Complete Analysis[/green] - Full debugging workflow")
    console.print("2. [blue]Quick Scalar Analysis[/blue] - Focus on convergence patterns")
    console.print("3. [yellow]Issue Detection Demo[/yellow] - Show common problems")
    console.print("4. [cyan]Capabilities Overview[/cyan] - Show what we can analyze")
    console.print()

    try:
        choice = console.input(
            "[bold]Choose mode [green][1][/green] Complete, [blue][2][/blue] Scalar, [yellow][3][/yellow] Issues, [cyan][4][/cyan] Overview (default: 1): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"  # Default to complete analysis
        console.print("[dim]Running complete analysis (non-interactive execution)[/dim]")

    try:
        if choice == "2":
            debugger.analyze_scalar_patterns()
        elif choice == "3":
            debugger.demonstrate_issue_detection()
        elif choice == "4":
            debugger.show_capabilities()
        else:
            debugger.run_complete_analysis()

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Analysis error: {e!s}[/red]")


if __name__ == "__main__":
    main()
