import asyncio
from typing import Any, Dict, List, Optional, Union
import json

import openai
from openai.types.beta.threads import Run
from pydantic import PrivateAttr

from autoagent.types import Agent, Response
from autoagent.util import function_to_json
from autoagent.context_models import AssistantContext, convert_dict_to_context, convert_context_to_dict


class OpenAIAssistantAgent(Agent):
    """
    An agent that uses the OpenAI Assistants API to manage its state and interactions.
    
    This agent doesn't use the traditional functions, instructions, or examples fields
    of the base Agent class. Instead, it manages these through the OpenAI Assistant itself.
    """
    # Private attributes that are not part of the Pydantic model
    _client: Any = PrivateAttr(default=None)
    _assistant: Any = PrivateAttr(default=None)
    _tools: List[Dict] = PrivateAttr(default_factory=list)
    _instructions: Optional[str] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _tool_functions: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Public attributes that are part of the Pydantic model
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    
    def __init__(
        self,
        name: str,
        assistant_id: Optional[str] = None,
        model: str = "gpt-4o",
        client: Optional[openai.AsyncOpenAI] = None,
        **kwargs
    ):
        """
        Initialize an OpenAI Assistant Agent.
        
        Args:
            name: The name of the agent
            assistant_id: Optional ID of an existing OpenAI Assistant
            model: The model to use for the assistant (default: gpt-4o)
            client: Optional OpenAI client instance
            **kwargs: Additional arguments to pass to the base Agent class
        """
        super().__init__(name=name, model=model, **kwargs)
        self._client = client or openai.AsyncOpenAI()
        self.assistant_id = assistant_id
        self.thread_id = None
        self._assistant = None
        self._initialized = False
        self._tool_functions = {}  # Store the actual tool functions
    
    async def initialize(self):
        """
        Initialize or retrieve the OpenAI Assistant.
        """
        if self._initialized:
            return self._assistant
            
        try:
            # Register tools with the registry if they're not already registered
            self._register_tools_with_registry()
            
            if self.assistant_id:
                # Retrieve existing assistant
                self._assistant = await self._client.beta.assistants.retrieve(self.assistant_id)
                
                # Update assistant if tools or instructions are provided
                if self._tools or self._instructions:
                    update_params = {}
                    if self._tools:
                        update_params["tools"] = self._tools
                    if self._instructions:
                        update_params["instructions"] = self._instructions
                    
                    self._assistant = await self._client.beta.assistants.update(
                        self.assistant_id,
                        **update_params
                    )
            else:
                # Create a new assistant
                default_instructions = self._instructions or f"You are {self.name}, a helpful assistant that uses tools to complete tasks."
                
                self._assistant = await self._client.beta.assistants.create(
                    name=self.name,
                    model=self.model,
                    instructions=default_instructions,
                    tools=self._tools or []
                )
                self.assistant_id = self._assistant.id
                
            self._initialized = True
            return self._assistant
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI Assistant: {str(e)}")
    
    def _register_tools_with_registry(self):
        """
        Register the tools with the registry if they're not already registered.
        This ensures that the tools are available to other agents in the system.
        """
        try:
            from autoagent.registry import registry
            
            # Register each tool function with the registry
            for tool_name, tool_func in self._tool_functions.items():
                if tool_name not in registry.tools:
                    # Create a wrapper function that will be registered
                    def create_wrapper(func):
                        def wrapper(context_variables=None, **kwargs):
                            return func(context_variables, **kwargs)
                        wrapper.__name__ = func.__name__
                        wrapper.__doc__ = func.__doc__
                        return wrapper
                    
                    # Register the wrapper function
                    registry.tools[tool_name] = create_wrapper(tool_func)
        except Exception as e:
            # Don't fail initialization if registry registration fails
            print(f"Warning: Failed to register tools with registry: {str(e)}")
    
    async def run_func(
        self, 
        messages: List[Dict[str, Any]], 
        context_variables: Dict[str, Any] = None,
        **kwargs
    ) -> Response:
        """
        Run the OpenAI Assistant with the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            context_variables: Dictionary of context variables
            **kwargs: Additional arguments
            
        Returns:
            Response object with the assistant's response
        """
        try:
            # Convert context_variables to AssistantContext if it's a dict
            context = context_variables
            if isinstance(context_variables, dict):
                context = convert_dict_to_context(context_variables, AssistantContext)
            
            # Ensure assistant is initialized
            if not self._initialized:
                await self.initialize()
            
            # Create a thread if one doesn't exist
            if not self.thread_id and not context.thread_id:
                thread = await self._client.beta.threads.create()
                self.thread_id = thread.id
                context.thread_id = thread.id
            elif context.thread_id:
                self.thread_id = context.thread_id
            
            # Add user messages to the thread
            for message in messages:
                if message["role"] == "user":
                    await self._client.beta.threads.messages.create(
                        thread_id=self.thread_id,
                        role="user",
                        content=message["content"]
                    )
            
            # Create a run
            run = await self._client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            )
            
            # Poll the run until it's complete
            run = await self._poll_run(run.id)
            
            # Retrieve messages from the thread
            thread_messages = await self._client.beta.threads.messages.list(
                thread_id=self.thread_id
            )
            
            # Get the latest assistant message
            assistant_messages = [
                msg for msg in thread_messages.data 
                if msg.role == "assistant"
            ]
            
            if not assistant_messages:
                return Response(content="No response from assistant.")
            
            # Get the most recent assistant message
            latest_message = assistant_messages[0]
            
            # Convert to Response format
            content = ""
            for content_item in latest_message.content:
                if content_item.type == "text":
                    content += content_item.text.value
            
            # Update context with assistant and thread IDs
            context.assistant_id = self.assistant_id
            context.thread_id = self.thread_id
            
            # Convert context back to dict if needed
            context_dict = context
            if isinstance(context, AssistantContext):
                context_dict = convert_context_to_dict(context)
            
            return Response(
                messages=[{"role": "assistant", "content": content}],
                agent=self,
                context_variables=context_dict
            )
            
        except Exception as e:
            return Response(
                messages=[{"role": "assistant", "content": f"Error: {str(e)}"}],
                agent=self,
                context_variables={}
            )
    
    async def _poll_run(self, run_id: str) -> Run:
        """
        Poll the run until it's complete.
        
        Args:
            run_id: The ID of the run to poll
            
        Returns:
            The completed run
        """
        while True:
            run = await self._client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run_id
            )
            
            if run.status in ["completed", "failed", "cancelled", "expired"]:
                return run
            
            # Handle tool calls if needed
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                
                tool_outputs = await self._handle_tool_calls(tool_calls, context_variables={})
                
                await self._client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )
            
            # Wait before polling again
            await asyncio.sleep(1)
    
    async def _handle_tool_calls(self, tool_calls, context_variables):
        """
        Handle tool calls from the OpenAI Assistant.
        
        Args:
            tool_calls: List of tool calls from the OpenAI Assistant
            context_variables: Dictionary of context variables
            
        Returns:
            List of tool outputs
        """
        tool_outputs = []
        
        for tool_call in tool_calls:
            try:
                # Find the tool function
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # First check if the tool is in our local tool functions
                if tool_name in self._tool_functions:
                    tool_func = self._tool_functions[tool_name]
                    
                    # Call the tool function
                    result = tool_func(context_variables, **tool_args)
                    
                    # Update context variables if needed
                    if hasattr(result, 'context_variables'):
                        context_variables.update(result.context_variables)
                    
                    # Get the tool output
                    if isinstance(result, dict) and "value" in result:
                        output = result["value"]
                    elif hasattr(result, 'value'):
                        output = result.value
                    else:
                        output = str(result)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output
                    })
                # Then check the registry
                elif hasattr(self, '_registry') and tool_name in self._registry.tools:
                    tool_func = self._registry.tools[tool_name]
                    
                    # Call the tool function
                    result = tool_func(context_variables, **tool_args)
                    
                    # Update context variables if needed
                    if hasattr(result, 'context_variables'):
                        context_variables.update(result.context_variables)
                    
                    # Get the tool output
                    if isinstance(result, dict) and "value" in result:
                        output = result["value"]
                    elif hasattr(result, 'value'):
                        output = result.value
                    else:
                        output = str(result)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output
                    })
                else:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Tool '{tool_name}' not found"
                    })
            except Exception as e:
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Error executing tool: {str(e)}"
                })
        
        return tool_outputs 