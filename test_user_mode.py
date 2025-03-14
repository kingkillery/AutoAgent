import os
import sys
from dotenv import load_dotenv
from rich.console import Console
import json

# Add the project root to the Python path to import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env file
load_dotenv()

# Set environment variables for testing
os.environ["FN_CALL"] = "False"  # Simulate non-function calling mode
os.environ["MC_MODE"] = "True"

# Import AutoAgent modules after setting environment variables
from autoagent.core import MetaChain
from autoagent.types import Agent

def transfer_to_websurfer_agent(dummy=None):
    """Transfer to Web Surfer Agent"""
    return {"message": "Transferring to Web Surfer Agent"}

def transfer_to_filesurfer_agent(dummy=None):
    """Transfer to File Surfer Agent"""
    return {"message": "Transferring to File Surfer Agent"}

def test_user_mode_gemini():
    """Test user mode with Gemini."""
    console = Console()
    console.print("[bold green]Testing User Mode with Gemini...[/bold green]")
    
    # Create agent
    agent = Agent(
        name="Test Triage Agent",
        model="gemini/gemini-2.0-flash",
        instructions="""You are a helpful assistant that can help the user with their request.
Based on the state of solving user's task, your responsibility is to determine which agent is best suited to handle the user's request under the current context, and transfer the conversation to that agent. And you should not stop to try to solve the user's request by transferring to another agent only until the task is completed.

There are two agents you can transfer to:
1. use `transfer_to_filesurfer_agent` to transfer to File Surfer Agent, it can help you to open any type of local files and browse the content of them.
2. use `transfer_to_websurfer_agent` to transfer to Web Surfer Agent, it can help you to open any website and browse any content on it.
""",
        functions=[
            transfer_to_websurfer_agent,
            transfer_to_filesurfer_agent
        ]
    )
    
    # Create chat history
    messages = [
        {"role": "user", "content": "Please visit reddit.com in the web browser and share a *link* to each story on the front page"}
    ]
    
    # Create MetaChain instance
    client = MetaChain(log_path=None)
    
    try:
        # Run the agent
        response = client.run(agent, messages, {}, debug=True)
        
        # Check if there are tool_calls in the response
        tool_calls = response.messages[-1].get("tool_calls", None)
        console.print(f"[bold blue]Tool Calls:[/bold blue] {tool_calls}")
        
        if tool_calls:
            console.print("[bold green]SUCCESS: Tool calls were properly populated![/bold green]")
        else:
            console.print("[bold red]FAILURE: Tool calls are still None![/bold red]")
        
        # Print the full response for debugging
        console.print("[bold yellow]Full Response:[/bold yellow]")
        console.print(json.dumps(response.messages[-1], indent=2))
    except Exception as e:
        console.print(f"[bold red]ERROR: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    test_user_mode_gemini() 