import asyncio
from typing import List, Optional, Callable

from autoagent.registry import register_agent
from autoagent.util import function_to_json
from autoagent.agents.openai_assistant_agent import OpenAIAssistantAgent

@register_agent(name="openai_assistant_agent")
def get_openai_assistant_agent(
    context_variables: dict = None,
    name: str = "OpenAI Assistant",
    tools: List[Callable] = None,
    instructions: str = None,
    assistant_id: Optional[str] = None,
    model: str = "gpt-4o"
):
    """
    Create an OpenAI Assistant Agent.
    
    Args:
        context_variables: Dictionary of context variables
        name: The name of the agent
        tools: List of tool functions to register with the assistant
        instructions: Instructions for the assistant
        assistant_id: Optional ID of an existing OpenAI Assistant
        model: The model to use for the assistant (default: gpt-4o)
        
    Returns:
        An OpenAIAssistantAgent instance
    """
    # Create the agent
    agent = OpenAIAssistantAgent(
        name=name,
        assistant_id=assistant_id,
        model=model
    )
    
    # Convert tools to OpenAI format
    openai_tools = []
    if tools:
        openai_tools = [function_to_json(tool) for tool in tools]
        # Store the tool functions in the agent
        for tool in tools:
            agent._tool_functions[tool.__name__] = tool
    
    # Set the tools and instructions for later initialization
    agent._tools = openai_tools
    agent._instructions = instructions
    
    return agent 