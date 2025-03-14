import os
import sys
import asyncio
from dotenv import load_dotenv
from rich.console import Console

# Add the project root to the Python path to import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env file
load_dotenv()

# Import AutoAgent modules
from autoagent.core import MetaChain
from autoagent.agents import get_openai_assistant_agent

# Define some simple tool functions
def get_current_time(context_variables=None):
    """Get the current time."""
    from datetime import datetime
    return {"value": f"The current time is {datetime.now().strftime('%H:%M:%S')}"}

def get_weather(context_variables=None, city: str = "New York"):
    """Get the weather for a city."""
    return {"value": f"The weather in {city} is sunny and 75°F."}

async def main():
    """Test the OpenAI Assistant Agent with the AutoAgent system."""
    console = Console()
    console.print("[bold green]Testing OpenAI Assistant Agent with AutoAgent...[/bold green]")
    
    # Create the OpenAI Assistant Agent
    agent = get_openai_assistant_agent(
        name="Test Assistant",
        tools=[get_current_time, get_weather],
        instructions="You are a helpful assistant that can provide information about the current time and weather."
    )
    
    console.print(f"[bold green]Assistant created with ID: {agent.assistant_id}[/bold green]")
    
    # Create chat history
    messages = [
        {"role": "user", "content": "What time is it now?"}
    ]
    
    # Create context variables
    context_variables = {}
    
    # Create MetaChain instance
    client = MetaChain(log_path=None)
    
    try:
        # Run the agent
        console.print("[bold blue]Running the agent...[/bold blue]")
        response = await agent.run_func(messages, context_variables)
        
        # Print the response
        console.print("[bold yellow]Response:[/bold yellow]")
        console.print(response.messages[0]["content"])
        
        # Update context variables
        context_variables = response.context_variables
        
        # Ask another question
        messages.append({"role": "assistant", "content": response.messages[0]["content"]})
        messages.append({"role": "user", "content": "What's the weather in San Francisco?"})
        
        # Run the agent again
        console.print("\n[bold blue]Running the agent again...[/bold blue]")
        response = await agent.run_func(messages, context_variables)
        
        # Print the response
        console.print("[bold yellow]Response:[/bold yellow]")
        console.print(response.messages[0]["content"])
        
    except Exception as e:
        console.print(f"[bold red]ERROR: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    finally:
        console.print("[bold blue]Test completed.[/bold blue]")

if __name__ == "__main__":
    asyncio.run(main()) 