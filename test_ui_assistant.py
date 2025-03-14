import os
import sys
import logging
from rich.console import Console

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autoagent import MetaChain
from autoagent.agents.ui_assistant_agent import get_ui_assistant_agent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UIAssistantTest")

def test_ui_assistant():
    """Test the UI Assistant agent with a simple query"""
    console = Console()
    
    try:
        # Create the UI Assistant agent
        console.print("[bold blue]Creating UI Assistant agent...[/bold blue]")
        agent = get_ui_assistant_agent("gpt-4o")
        
        # Create MetaChain client
        console.print("[bold blue]Creating MetaChain client...[/bold blue]")
        client = MetaChain()
        
        # Test messages
        messages = [
            {"role": "user", "content": "Can you visit wikipedia.org and search for 'artificial intelligence'?"}
        ]
        
        # Run the agent
        console.print("[bold green]Running UI Assistant agent...[/bold green]")
        console.print("[bold yellow]This may take a moment as it processes the request...[/bold yellow]")
        
        response = client.run(agent, messages, {}, debug=True)
        
        # Display the response
        console.print("\n[bold blue]UI Assistant Response:[/bold blue]")
        console.print(response.messages[-1]['content'])
        
        console.print("\n[bold green]Test completed successfully![/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error testing UI Assistant: {str(e)}[/bold red]")
        return False

if __name__ == "__main__":
    console = Console()
    console.print("[bold green]===== UI Assistant Test =====[/bold green]")
    
    result = test_ui_assistant()
    
    if result:
        console.print("[bold green]All tests passed![/bold green]")
    else:
        console.print("[bold red]Tests failed![/bold red]") 