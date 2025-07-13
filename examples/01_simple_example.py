#!/usr/bin/env python3
"""
Simple ClearML Analysis Example with Smolagents

A minimal example showing how to use the ClearML MCP server with smolagents
and Gemini to analyze ML experiments.
"""

import os

from mcp import StdioServerParameters

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

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

from smolagents import CodeAgent, MCPClient
from smolagents.models import GeminiModel


def main():
    """Simple example of ClearML analysis with smolagents."""
    print("🔬 Simple ClearML Analysis Example")
    print("=" * 40)

    # Initialize Gemini model
    model = GeminiModel(
        model_id="gemini-1.5-flash",
        api_key=GEMINI_API_KEY,
    )

    # Configure ClearML MCP server
    clearml_server_params = StdioServerParameters(
        command="uvx",
        args=["clearml-mcp"],
    )

    # Connect to ClearML MCP server and create agent
    with MCPClient(clearml_server_params) as tools:
        agent = CodeAgent(
            tools=tools,
            model=model,
            add_base_tools=True,
        )

        # Simple analysis query
        query = "List all available ClearML projects and tell me what you find."

        print(f"🤖 Agent Query: {query}")
        print("-" * 40)

        try:
            result = agent.run(query)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure ClearML is configured and clearml-mcp is available.")


if __name__ == "__main__":
    main()
