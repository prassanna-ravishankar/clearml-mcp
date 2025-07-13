#!/usr/bin/env python3
"""
ClearML Analysis Agent using Smolagents with Gemini API

This example demonstrates how to create an intelligent agent that can analyze
ClearML experiments using our ClearML MCP server with the smolagents library
and Google's Gemini API.

Requirements:
- smolagents with MCP support
- clearml-mcp server (this project)
- Google Gemini API access
- ClearML configuration (~/.clearml.conf)
"""

import os
from mcp import StdioServerParameters

# Set up Gemini API key
GEMINI_API_KEY = "AIzaSyDAdEToKdFt8SHs25ABz65bx6cedU_zreo"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    from smolagents import MCPClient, CodeAgent
    from smolagents.models import GeminiModel
except ImportError:
    print("❌ Smolagents not found. Install with:")
    print("   pip install 'smolagents[gemini,mcp]'")
    print("   or")
    print("   uv add 'smolagents[gemini,mcp]'")
    raise


def create_clearml_analysis_agent():
    """Create a ClearML analysis agent using Gemini and our MCP server."""
    
    # Initialize Gemini model
    model = GeminiModel(
        model_id="gemini-1.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.1  # Lower temperature for more focused analysis
    )
    
    # Configure ClearML MCP server parameters
    clearml_server_params = StdioServerParameters(
        command="uvx",
        args=["clearml-mcp"],
        env=os.environ  # Pass through environment variables
    )
    
    return model, clearml_server_params


def demonstrate_clearml_analysis():
    """Demonstrate various ClearML analysis capabilities."""
    
    print("🚀 Starting ClearML Analysis Agent with Gemini...")
    print("=" * 60)
    
    # Create the model and server parameters
    model, clearml_server_params = create_clearml_analysis_agent()
    
    # Example queries that showcase different ClearML operations
    analysis_queries = [
        {
            "title": "Project Overview",
            "query": "List all available ClearML projects and give me a summary of what projects are available."
        },
        {
            "title": "Experiment Analysis", 
            "query": "Get detailed information about the experiment with ID 'efe5f7a6c5f34a15b4bfbf1c33660e20'. Analyze its status, parameters, and provide insights about this experiment."
        },
        {
            "title": "Performance Metrics",
            "query": "Get the training metrics for experiment 'efe5f7a6c5f34a15b4bfbf1c33660e20' and analyze the performance trends. What can you tell me about the training progress?"
        },
        {
            "title": "Experiment Search",
            "query": "Search for experiments that contain 'training' or 'model' in their names. Show me the top 5 most relevant results and summarize their key characteristics."
        },
        {
            "title": "Hyperparameter Analysis",
            "query": "Get the parameters for experiment 'efe5f7a6c5f34a15b4bfbf1c33660e20' and analyze the hyperparameter configuration. What optimization settings were used?"
        }
    ]
    
    # Connect to ClearML MCP server and run analysis
    with MCPClient(clearml_server_params) as clearml_tools:
        print(f"✅ Connected to ClearML MCP server")
        print(f"🛠️  Available tools: {len(clearml_tools)} ClearML tools")
        
        # Create agent with ClearML tools
        agent = CodeAgent(
            tools=clearml_tools,
            model=model,
            add_base_tools=True,  # Include built-in tools
            max_iterations=15,    # Allow more iterations for complex analysis
            verbosity_level=1     # Show tool usage
        )
        
        # Run each analysis query
        for i, analysis in enumerate(analysis_queries, 1):
            print(f"\n🔍 Analysis {i}: {analysis['title']}")
            print(f"📝 Query: {analysis['query']}")
            print("-" * 50)
            
            try:
                result = agent.run(analysis['query'])
                print(f"✅ Analysis Complete!")
                print(f"📊 Result: {result}")
                
            except Exception as e:
                print(f"❌ Analysis failed: {str(e)}")
                print("This might be due to:")
                print("  - ClearML configuration issues")
                print("  - Invalid experiment ID")
                print("  - Network connectivity")
            
            print("\n" + "=" * 60)
        
        print("\n🎉 All analyses completed!")


def interactive_mode():
    """Run the agent in interactive mode for custom queries."""
    
    print("\n🤖 Interactive ClearML Analysis Mode")
    print("Enter your questions about ClearML experiments, or 'quit' to exit.")
    print("Examples:")
    print("  - 'Show me all projects'") 
    print("  - 'Analyze experiment xyz123'")
    print("  - 'Compare experiments abc and def'")
    print("-" * 50)
    
    # Create the model and server parameters
    model, clearml_server_params = create_clearml_analysis_agent()
    
    with MCPClient(clearml_server_params) as clearml_tools:
        agent = CodeAgent(
            tools=clearml_tools,
            model=model,
            add_base_tools=True,
            max_iterations=10,
            verbosity_level=1
        )
        
        while True:
            try:
                user_query = input("\n🗣️  Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not user_query:
                    continue
                    
                print("\n🤔 Analyzing...")
                result = agent.run(user_query)
                print(f"\n💡 Answer: {result}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    """Main function with options for demo or interactive mode."""
    
    print("🔬 ClearML Analysis Agent")
    print("Powered by Smolagents + Gemini + ClearML MCP")
    print()
    print("Prerequisites:")
    print("✅ ClearML configured with ~/.clearml.conf")
    print("✅ clearml-mcp server available (uvx clearml-mcp)")
    print("✅ smolagents with MCP support installed")
    print("✅ Google Gemini API key configured")
    print()
    
    # Check if user wants demo or interactive mode
    mode = input("Choose mode - [d]emo or [i]nteractive (default: demo): ").strip().lower()
    
    try:
        if mode.startswith('i'):
            interactive_mode()
        else:
            demonstrate_clearml_analysis()
            
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure ClearML is configured: clearml-init")
        print("2. Test MCP server: uvx clearml-mcp")
        print("3. Check Gemini API key is valid")
        print("4. Install dependencies: pip install 'smolagents[gemini,mcp]'")


if __name__ == "__main__":
    main()