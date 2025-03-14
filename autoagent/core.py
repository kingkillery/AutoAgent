# Standard library imports
import copy
import json
import os
from collections import defaultdict
import traceback
from typing import List, Callable, Union
from datetime import datetime
# Local imports
from .util import function_to_json, debug_print, merge_chunk, pretty_print_messages
from .types import (
    Agent, AgentFunction, Message, ChatCompletionMessageToolCall,
    Function, Response, Result,
)
from litellm import completion, acompletion
from pathlib import Path
from .logger import MetaChainLogger, LoggerManager
from httpx import RemoteProtocolError, ConnectError
from litellm.exceptions import APIError
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)
from openai import AsyncOpenAI
import litellm
import inspect
from constant import MC_MODE, FN_CALL, API_BASE_URL, NOT_SUPPORT_SENDER, ADD_USER, NON_FN_CALL
from autoagent.fn_call_converter import (
    convert_tools_to_description, convert_non_fncall_messages_to_fncall_messages,
    SYSTEM_PROMPT_SUFFIX_TEMPLATE, convert_fn_messages_to_non_fn_messages, interleave_user_into_messages
)
from litellm.types.utils import Message as litellmMessage
from autoagent.agent_utils import detect_agent_reference

litellm.set_verbose = True  # Enable detailed logging

def should_retry_error(exception):
    if MC_MODE is False:
        print(f"Caught exception: {type(exception).__name__} - {str(exception)}")

    if isinstance(exception, (APIError, RemoteProtocolError, ConnectError)):
        return True

    error_msg = str(exception).lower()
    return any([
        "connection error" in error_msg,
        "server disconnected" in error_msg,
        "eof occurred" in error_msg,
        "timeout" in error_msg,
        "event loop is closed" in error_msg,
        "anthropicexception" in error_msg,
        "openrouterexception" in error_msg,
        "bad request" in error_msg,
        "too many requests" in error_msg,
        "service unavailable" in error_msg,
        "gateway timeout" in error_msg,
        "internal server error" in error_msg
    ])

__CTX_VARS_NAME__ = "context_variables"
logger = LoggerManager.get_logger()

class MetaChain:
    def __init__(self, log_path: Union[str, None, MetaChainLogger] = None):
        if logger:
            self.logger = logger
        elif isinstance(log_path, MetaChainLogger):
            self.logger = log_path
        else:
            self.logger = MetaChainLogger(log_path=log_path)

    def _prepare_messages_and_tools(self, agent, history, context_variables):
        context_variables = defaultdict(str, context_variables)
        instructions = agent.instructions(context_variables) if callable(agent.instructions) else agent.instructions

        if agent.examples:
            examples = agent.examples(context_variables) if callable(agent.examples) else agent.examples
            history = examples + history

        messages = [{"role": "system", "content": instructions}] + history
        tools = [function_to_json(f) for f in agent.functions]

        # Process tools
        for tool in tools:
            params = tool["function"]["parameters"]
            if "properties" not in params:
                params["properties"] = {}
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params.get("required", []):
                params["required"].remove(__CTX_VARS_NAME__)
            
            # If properties is empty or contains only object types without properties, add a dummy string parameter
            if not params["properties"] or all(
                prop.get("type") == "object" and (not prop.get("properties") or len(prop.get("properties", {})) == 0)
                for prop in params["properties"].values()
            ):
                params["properties"] = {
                    "dummy": {
                        "type": "string",
                        "description": "Dummy parameter added to meet non-empty properties requirement."
                    }
                }

        return messages, tools

    def _prepare_completion_params(self, create_model, messages, tools, stream, fn_call=True):
        if fn_call:
            assert litellm.supports_function_calling(model=create_model), (
                f"Model {create_model} does not support function calling, please set `FN_CALL=False` to use non-function calling mode"
            )
            # For Gemini model, ensure we include the tools parameter
            if "gemini" in create_model:
                google_api_key = os.getenv("GEMINI_API_KEY")
                if not google_api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")

                return {
                    "model": create_model,
                    "messages": messages,
                    "stream": stream,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_tokens": 8192,
                    "tools": tools,  # Always include tools for Gemini
                    "headers": {
                        "x-goog-api-key": google_api_key,
                        "Content-Type": "application/json"
                    }
                }
            else:
                return {
                    "model": create_model,
                    "messages": messages,
                    "stream": stream,
                    "tools": tools
                }
        else:
            assert create_model, "Model must be specified for non-function calling mode"

            # Special case for Gemini models that still use function calling, 
            # but with custom instructions in the message
            if "gemini" in create_model:
                # Still include tools parameter for Gemini, but also add instructions in the message 
                # so it knows how to format the response
                last_content = messages[-1]["content"]
                tools_description = convert_tools_to_description(tools)
                messages[-1]["content"] = (
                    last_content
                    + "\n[IMPORTANT] You MUST use the tools provided to complete the task.\n"
                    + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=tools_description)
                )

                google_api_key = os.getenv("GEMINI_API_KEY")
                if not google_api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")

                return {
                    "model": create_model,
                    "messages": messages,
                    "stream": stream,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_tokens": 8192,
                    "tools": tools,  # Include tools parameter even in non-function calling mode for Gemini
                    "headers": {
                        "x-goog-api-key": google_api_key,
                        "Content-Type": "application/json"
                    }
                }
            else:
                # For other models, proceed with normal non-function calling conversion
                last_content = messages[-1]["content"]
                tools_description = convert_tools_to_description(tools)
                messages[-1]["content"] = (
                    last_content
                    + "\n[IMPORTANT] You MUST use the tools provided to complete the task.\n"
                    + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=tools_description)
                )

                # Handle models that don't support sender field
                if any(not_sender_model in create_model for not_sender_model in NOT_SUPPORT_SENDER):
                    for message in messages:
                        if 'sender' in message:
                            del message['sender']

                if NON_FN_CALL:
                    messages = convert_fn_messages_to_non_fn_messages(messages)
                if ADD_USER and messages[-1]["role"] != "user":
                    messages = interleave_user_into_messages(messages)

                # Base params
                params = {
                    "model": create_model,
                    "messages": messages,
                    "stream": stream,
                }

                # Configure provider-specific settings
                if create_model.startswith("google/"):
                    params.update({
                        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
                        "headers": {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {os.getenv('GOOGLE_API_KEY')}"
                        }
                    })
                else:
                    params.update({
                        "base_url": "https://openrouter.ai/api/v1",
                        "headers": {
                            "HTTP-Referer": "https://github.com/nerdsaver/AutoAgent",
                            "X-Title": "AutoAgent",
                            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                        }
                    })

                return params

    def _handle_provider_failure(self, messages, stream, error_msg):
        # Try OpenRouter thinking model first
        self.logger.info("Primary provider failed, trying thinking model on OpenRouter...", title="MODEL UPGRADE", color="yellow")
        try:
            thinking_params = {
                "model": "openrouter/gemini/gemini-2.0-flash-thinking-exp:free",
                "messages": messages,
                "stream": stream,
                "base_url": "https://openrouter.ai/api/v1",
                "headers": {
                    "HTTP-Referer": "https://github.com/nerdsaver/AutoAgent",
                    "X-Title": "AutoAgent",
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                }
            }
            return completion(**thinking_params)
        except Exception as thinking_e:
            # Fall back to Google AI Studio
            self.logger.info("OpenRouter thinking model failed, falling back to Google AI Studio...", title="FAILOVER", color="yellow")
            try:
                google_api_key = os.getenv("GOOGLE_API_KEY")
                self.logger.info(f"Google API Key present: {bool(google_api_key)}", title="API Key Check", color="yellow")

                backup_params = {
                    "model": "gemini/gemini-2.0-flash-thinking-exp-01-21",
                    "messages": messages,
                    "stream": stream,
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
                    "custom_llm_provider": "google",
                    "headers": {
                        "Content-Type": "application/json",
                        "x-goog-api-key": google_api_key
                    }
                }

                debug_params = {**backup_params}
                if "api_key" in debug_params:
                    debug_params["api_key"] = "*****"
                self.logger.info(f"Google AI Studio params: {json.dumps(debug_params, indent=2)}",
                                title="Request Parameters", color="yellow")

                return completion(**backup_params)
            except Exception as backup_e:
                self.logger.info(f"All providers failed. Last error: {str(backup_e)}",
                                title="BACKUP ERROR", color="red")
                raise backup_e

    def _is_agent_mentioned(self, content, agent_name, debug=False):
        """Check if an agent is mentioned in the content."""
        # Check if content is None and handle it
        if content is None:
            return False
            
        agent_name_lower = agent_name.lower()
        content_lower = content.lower()
        
        print(f"DEBUG: _is_agent_mentioned checking for '{agent_name_lower}' in content") if debug else None
        
        # List of patterns to check for agent mentions
        patterns = [
            f"transfer to {agent_name_lower}",
            f"transfer to the {agent_name_lower}",
            f"transferred to {agent_name_lower}",
            f"transferred to the {agent_name_lower}",
            f"i have transferred to {agent_name_lower}",
            f"i have transferred to the {agent_name_lower}",
            f"i've transferred to {agent_name_lower}",
            f"i've transferred to the {agent_name_lower}",
            f"transferring to {agent_name_lower}",
            f"transferring to the {agent_name_lower}",
            f"i am transferring to {agent_name_lower}",
            f"i am transferring to the {agent_name_lower}",
            f"i'm transferring to {agent_name_lower}",
            f"i'm transferring to the {agent_name_lower}",
            f"transferring you to {agent_name_lower}",
            f"transferring you to the {agent_name_lower}",
            f"transferring the conversation to {agent_name_lower}",
            f"transferring the conversation to the {agent_name_lower}",
            f"i am transferring the conversation to {agent_name_lower}",
            f"i am transferring the conversation to the {agent_name_lower}",
            f"i'm transferring the conversation to {agent_name_lower}",
            f"i'm transferring the conversation to the {agent_name_lower}",
            f"i am transferring you to {agent_name_lower}",
            f"i am transferring you to the {agent_name_lower}",
            f"i'll transfer to {agent_name_lower}",
            f"i'll transfer to the {agent_name_lower}",
            f"i will transfer to {agent_name_lower}",
            f"i will transfer to the {agent_name_lower}",
            f"i'll transfer you to {agent_name_lower}",
            f"i'll transfer you to the {agent_name_lower}",
            f"i will transfer you to {agent_name_lower}",
            f"i will transfer you to the {agent_name_lower}",
            f"i have transferred the request to the {agent_name_lower}",
            f"i've transferred the request to the {agent_name_lower}",
            f"i have transferred the request to {agent_name_lower}",
            f"i've transferred the request to {agent_name_lower}",
            f"connecting you to {agent_name_lower}",
            f"connecting you to the {agent_name_lower}",
            f"connect you to {agent_name_lower}",
            f"connect you to the {agent_name_lower}",
            f"will connect you to {agent_name_lower}",
            f"will connect you to the {agent_name_lower}",
            f"using {agent_name_lower} to",
            f"use {agent_name_lower} to",
            f"through {agent_name_lower}",
            f"with {agent_name_lower}",
            f"handing over to {agent_name_lower}",
            f"handing over to the {agent_name_lower}",
            f"hand over to {agent_name_lower}",
            f"hand over to the {agent_name_lower}",
            f"hand this over to {agent_name_lower}",
            f"hand this over to the {agent_name_lower}",
            f"passing to {agent_name_lower}",
            f"passing to the {agent_name_lower}",
            f"pass to {agent_name_lower}",
            f"pass to the {agent_name_lower}",
            f"letting {agent_name_lower} handle",
            f"let {agent_name_lower} handle",
            f"have {agent_name_lower} handle",
            f"having {agent_name_lower} handle",
            f"{agent_name_lower} can handle",
            f"{agent_name_lower} will handle",
            f"{agent_name_lower} is better suited",
            f"{agent_name_lower} is better equipped",
            f"i am now the {agent_name_lower}",
            f"i'm now the {agent_name_lower}",
            f"now acting as the {agent_name_lower}",
            f"now functioning as the {agent_name_lower}",
            f"i am the {agent_name_lower}",
            f"i'm the {agent_name_lower}",
            f"as the {agent_name_lower}, i",
            f"acting as the {agent_name_lower}",
            f"functioning as the {agent_name_lower}",
        ]
        
        # Special handling for "Web Surfer agent"
        if agent_name.lower() == "websurfer":
            special_patterns = [
                "transferred to the web surfer agent",
                "i have transferred to the web surfer agent",
                "i've transferred to the web surfer agent",
                "transferred to web surfer agent", 
                "i have transferred to web surfer agent",
                "i've transferred to web surfer agent",
                "i am now the web surfer agent",
                "i'm now the web surfer agent", 
                "now acting as the web surfer agent",
                "now functioning as the web surfer agent",
                "transferring the conversation to the web surfer agent",
                "i am transferring the conversation to the web surfer agent",
                "i'm transferring the conversation to the web surfer agent"
            ]
            patterns.extend(special_patterns)
            
            # Additional debug for Web Surfer agent
            print(f"DEBUG: Checking for Web Surfer agent with special patterns: {special_patterns}") if debug else None
            for pattern in special_patterns:
                if pattern in content_lower:
                    print(f"DEBUG: MATCH FOUND! Pattern '{pattern}' found in content") if debug else None
                    return True
        
        for pattern in patterns:
            if pattern in content_lower:
                print(f"DEBUG: Pattern '{pattern}' found in content") if debug else None
                return True
        
        # Check if we have an exact word match for the agent name
        if f" {agent_name_lower} " in f" {content_lower} ":
            print(f"DEBUG: Direct mention of agent '{agent_name_lower}' found") if debug else None
            return True
            
        print(f"DEBUG: No agent mention pattern detected") if debug else None
        return False

    def _detect_agent_reference(self, content, tools, debug=False):
        """
        Detect if the response content mentions transferring to an agent.
        """
        return detect_agent_reference(content, tools, debug)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=should_retry_error,
        before_sleep=lambda retry_state: print(f"Retrying... (attempt {retry_state.attempt_number})")
    )
    def get_chat_completion(self, agent, history, context_variables, model_override, stream, debug):
        messages, tools = self._prepare_messages_and_tools(agent, history, context_variables)
        create_model = model_override or agent.model

        try:
            params = self._prepare_completion_params(create_model, messages, tools, stream, fn_call=FN_CALL)
            print("Input to litellm:", params) if debug else None
            completion_response = completion(**params)

            # Handle non-function calling mode response conversion
            if not FN_CALL or "gemini" in create_model:
                # For Gemini or non-function calling mode, check if we need to convert the response
                if not completion_response.choices[0].message.tool_calls:
                    content = completion_response.choices[0].message.content
                    
                    # Debug output for content
                    print(f"DEBUG: Response content: '{content}'")
                    
                    # Check if content is None and handle it
                    if content is None:
                        content = ""  # Set content to empty string if it's None
                    
                    # Truncate extremely long content (likely repetitive or hallucination)
                    max_content_length = 2000  # Reasonable limit for content length
                    if len(content) > max_content_length:
                        print(f"WARNING: Truncating extremely long content ({len(content)} chars) to {max_content_length} chars")
                        content = content[:max_content_length] + "... [content truncated due to excessive length]"
                    
                    # Special handling for WebSurfer Agent that mentions transfer but doesn't use the tool
                    if "websurfer agent" in agent.name.lower() and "transfer back to the system triage agent" in content.lower() and len(content) > 300:
                        print(f"WARNING: WebSurfer Agent mentions transferring but doesn't use tool, forcing tool call")
                        tool_call = {
                            'index': 0,
                            'id': f'forced_transfer_{len(history):02d}',
                            'type': 'function',
                            'function': {
                                'name': 'transfer_back_to_triage_agent',
                                'arguments': json.dumps({"task_status": "Task completed, transferring back to System Triage Agent"})
                            }
                        }
                        completion_response.choices[0].message = litellmMessage(
                            content=content[:300] + "... [content truncated]",
                            role="assistant",
                            tool_calls=[ChatCompletionMessageToolCall(**tool_call)]
                        )
                        return completion_response
                    
                    # Check if the content contains function call format
                    if "<function=" in content and "</function>" in content:
                        print("DEBUG: Function call format detected")
                        # Convert text to tool_calls
                        last_message = [{"role": "assistant", "content": content}]
                        converted_message = convert_non_fncall_messages_to_fncall_messages(last_message, tools)
                        
                        # Check if conversion resulted in tool calls
                        if converted_message[0].get("tool_calls"):
                            print(f"DEBUG: Converted tool calls: {converted_message[0].get('tool_calls')}")
                            converted_tool_calls = [
                                ChatCompletionMessageToolCall(**tool_call)
                                for tool_call in converted_message[0].get("tool_calls", [])
                            ]
                            completion_response.choices[0].message = litellmMessage(
                                content=converted_message[0]["content"],
                                role="assistant",
                                tool_calls=converted_tool_calls
                            )
                    # Special case: If the response mentions the agent but doesn't include a function call
                    elif self._detect_agent_reference(content, tools, debug):
                        print("DEBUG: Agent reference detected") if debug else None
                        # Find the mentioned agent
                        for tool in tools:
                            agent_name = tool['function']['name'].replace('transfer_to_', '').replace('_agent', '')
                            if self._is_agent_mentioned(content.lower(), agent_name, debug):
                                print(f"DEBUG: Agent '{agent_name}' is mentioned") if debug else None
                                # Create a tool call for the mentioned agent
                                tool_call = {
                                    'index': 1,
                                    'id': f'toolu_{len(history):02d}',
                                    'type': 'function',
                                    'function': {
                                        'name': tool['function']['name'],
                                        'arguments': json.dumps({"dummy": "true"})  # Use "dummy" parameter instead of "query"
                                    }
                                }
                                print(f"DEBUG: Created tool call: {tool_call}") if debug else None
                                completion_response.choices[0].message = litellmMessage(
                                    content=content,
                                    role="assistant",
                                    tool_calls=[ChatCompletionMessageToolCall(**tool_call)]
                                )
                                print(f"DEBUG: Updated message with tool calls: {completion_response.choices[0].message.tool_calls}") if debug else None
                                break
                else:
                    print(f"DEBUG: Tool calls already present: {completion_response.choices[0].message.tool_calls}")

            return completion_response
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["openrouterexception", "bad request", "service unavailable"]):
                return self._handle_provider_failure(messages, stream, error_msg)
            else:
                raise e

    def handle_function_result(self, result, debug) -> Result:
        if isinstance(result, Result):
            return result
        elif isinstance(result, Agent):
            return Result(value=json.dumps({"assistant": result.name}), agent=result)
        else:
            try:
                return Result(value=str(result))
            except Exception as e:
                error_message = (
                    f"Failed to cast response to string: {result}. "
                    f"Make sure agent functions return a string or Result object. "
                    f"Error: {str(e)}"
                )
                self.logger.info(error_message, title="Handle Function Result Error", color="red")
                raise TypeError(error_message)

    def handle_tool_calls(self, tool_calls, functions, context_variables, debug, handle_mm_func=None) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # Handle missing tool case
            if name not in function_map:
                self.logger.info(
                    f"Tool {name} not found in function map. You are recommended to use `run_tool` to run this tool.",
                    title="Tool Call Error", color="red"
                )
                partial_response.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": f"Error: Tool {name} not found. You are recommended to use `run_tool` to run this tool."
                })
                continue

            # Execute function
            args = json.loads(tool_call.function.arguments)
            func = function_map[name]

            # Pass context_variables if signature allows, ensure it's at least an empty dict
            if context_variables is None:
                context_variables = {}
                
            if __CTX_VARS_NAME__ in inspect.signature(func).parameters:
                args[__CTX_VARS_NAME__] = context_variables

            # Handle WebSurfer special tools that need local_root and workplace_name
            if name.startswith("wsurfer_") and __CTX_VARS_NAME__ in args:
                if "local_root" not in args[__CTX_VARS_NAME__]:
                    args[__CTX_VARS_NAME__]["local_root"] = os.path.expanduser("~")
                if "workplace_name" not in args[__CTX_VARS_NAME__]:
                    args[__CTX_VARS_NAME__]["workplace_name"] = "websurfer_workplace"

            try:
                raw_result = func(**args)
                result = self.handle_function_result(raw_result, debug)
            except Exception as e:
                self.logger.info(f"Error executing tool {name}: {str(e)}", 
                                title="Tool Execution Error", color="red")
                traceback.print_exc()
                result = Result(
                    value=f"Error executing {name}: {str(e)}. Please try again or use a different approach."
                )

            # Add tool response
            partial_response.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result.value
            })
            self.logger.pretty_print_messages(partial_response.messages[-1])

            # Handle image result
            if result.image:
                assert handle_mm_func, f"handle_mm_func is not provided, but an image is returned by tool call {name}"
                partial_response.messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": handle_mm_func(name, tool_call.function.arguments)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result.image}"}}
                    ]
                })

            # Update context and agent
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(self, agent, messages, context_variables={}, model_override=None,
                      debug=False, max_turns=float("inf"), execute_tools=True):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:
            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(lambda: {"function": {"arguments": "", "name": ""}, "id": "", "type": ""})
            }

            # Get completion
            completion = self.get_chat_completion(
                agent=active_agent, history=history, context_variables=context_variables,
                model_override=model_override, stream=True, debug=debug
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None

            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # Process tool calls
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"]
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # Handle function calls
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables
            )
        }

    def run(self, agent, messages, context_variables={}, model_override=None, stream=False,
           debug=True, max_turns=float("inf"), execute_tools=True) -> Response:
        # Check if the agent is an OpenAIAssistantAgent
        from autoagent.agents.openai_assistant_agent import OpenAIAssistantAgent
        if isinstance(agent, OpenAIAssistantAgent):
            # For OpenAIAssistantAgent, we use its run_func method directly
            return agent.run_func(messages=messages, context_variables=context_variables)
            
        if stream:
            return self.run_and_stream(
                agent=agent, messages=messages, context_variables=context_variables,
                model_override=model_override, debug=debug, max_turns=max_turns, execute_tools=execute_tools
            )

        active_agent = agent
        enter_agent = agent
        context_variables = copy.copy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info("Receiveing the task:", history[-1]['content'], title="Receive Task", color="green")

        while len(history) - init_len < max_turns and active_agent:
            # Get completion
            completion = self.get_chat_completion(
                agent=active_agent, history=history, context_variables=context_variables,
                model_override=model_override, stream=stream, debug=debug
            )
            message = completion.choices[0].message
            message.sender = active_agent.name
            self.logger.pretty_print_messages(message)
            history.append(json.loads(message.model_dump_json()))

            # Check for end conditions
            if enter_agent.tool_choice != "required":
                if (not message.tool_calls and active_agent.name == enter_agent.name) or not execute_tools:
                    self.logger.info("Ending turn.", title="End Turn", color="red")
                    break
            else:
                if message.tool_calls and message.tool_calls[0].function.name in ["case_resolved", "case_not_resolved"]:
                    self.logger.info(f"Ending turn with {message.tool_calls[0].function.name}.", title="End Turn", color="red")
                    partial_response = self.handle_tool_calls(
                        message.tool_calls, active_agent.functions, context_variables, debug,
                        handle_mm_func=active_agent.handle_mm_func
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break
                elif not execute_tools:
                    self.logger.info("Ending turn (execute_tools=False).", title="End Turn", color="red")
                    break

            # Process tool calls
            if message.tool_calls:
                partial_response = self.handle_tool_calls(
                    message.tool_calls, active_agent.functions, context_variables, debug,
                    handle_mm_func=active_agent.handle_mm_func
                )
            else:
                partial_response = Response(messages=[message])

            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        result = Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables
        )

        if isinstance(result, Response):
            return result
        else:
            response_content = json.dumps(result, ensure_ascii=False, indent=2)
            return Response(
                messages=[{"role": "assistant", "content": response_content}],
                agent=active_agent,
                context_variables=context_variables
            )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=10, max=180),
        retry=should_retry_error,
        before_sleep=lambda retry_state: print(f"Retrying... (attempt {retry_state.attempt_number})")
    )
    async def get_chat_completion_async(self, agent, history, context_variables, model_override, stream, debug):
        messages, tools = self._prepare_messages_and_tools(agent, history, context_variables)
        create_model = model_override or agent.model

        try:
            params = self._prepare_completion_params(create_model, messages, tools, stream, fn_call=FN_CALL)
            completion_response = await acompletion(**params)

            # Handle non-function calling mode response conversion
            if not FN_CALL or "gemini" in create_model:
                # For Gemini or non-function calling mode, check if we need to convert the response
                if not completion_response.choices[0].message.tool_calls:
                    content = completion_response.choices[0].message.content
                    
                    # Debug output for content
                    print(f"DEBUG: Response content: '{content}'")
                    
                    # Check if content is None and handle it
                    if content is None:
                        content = ""  # Set content to empty string if it's None
                    
                    # Truncate extremely long content (likely repetitive or hallucination)
                    max_content_length = 2000  # Reasonable limit for content length
                    if len(content) > max_content_length:
                        print(f"WARNING: Truncating extremely long content ({len(content)} chars) to {max_content_length} chars")
                        content = content[:max_content_length] + "... [content truncated due to excessive length]"
                    
                    # Special handling for WebSurfer Agent that mentions transfer but doesn't use the tool
                    if "websurfer agent" in agent.name.lower() and "transfer back to the system triage agent" in content.lower() and len(content) > 300:
                        print(f"WARNING: WebSurfer Agent mentions transferring but doesn't use tool, forcing tool call")
                        tool_call = {
                            'index': 0,
                            'id': f'forced_transfer_{len(history):02d}',
                            'type': 'function',
                            'function': {
                                'name': 'transfer_back_to_triage_agent',
                                'arguments': json.dumps({"task_status": "Task completed, transferring back to System Triage Agent"})
                            }
                        }
                        completion_response.choices[0].message = litellmMessage(
                            content=content[:300] + "... [content truncated]",
                            role="assistant",
                            tool_calls=[ChatCompletionMessageToolCall(**tool_call)]
                        )
                        return completion_response
                    
                    # Check if the content contains function call format
                    if "<function=" in content and "</function>" in content:
                        print("DEBUG: Function call format detected")
                        # Convert text to tool_calls
                        last_message = [{"role": "assistant", "content": content}]
                        converted_message = convert_non_fncall_messages_to_fncall_messages(last_message, tools)
                        
                        # Check if conversion resulted in tool calls
                        if converted_message[0].get("tool_calls"):
                            print(f"DEBUG: Converted tool calls: {converted_message[0].get('tool_calls')}") if debug else None
                            converted_tool_calls = [
                                ChatCompletionMessageToolCall(**tool_call)
                                for tool_call in converted_message[0].get("tool_calls", [])
                            ]
                            completion_response.choices[0].message = litellmMessage(
                                content=converted_message[0]["content"],
                                role="assistant",
                                tool_calls=converted_tool_calls
                            )
                    # Special case: If the response mentions the agent but doesn't include a function call
                    elif self._detect_agent_reference(content, tools, debug):
                        print("DEBUG: Agent reference detected") if debug else None
                        # Find the mentioned agent
                        for tool in tools:
                            agent_name = tool['function']['name'].replace('transfer_to_', '').replace('_agent', '')
                            if self._is_agent_mentioned(content.lower(), agent_name, debug):
                                print(f"DEBUG: Agent '{agent_name}' is mentioned") if debug else None
                                # Create a tool call for the mentioned agent
                                tool_call = {
                                    'index': 1,
                                    'id': f'toolu_{len(history):02d}',
                                    'type': 'function',
                                    'function': {
                                        'name': tool['function']['name'],
                                        'arguments': json.dumps({"dummy": "true"})  # Use "dummy" parameter instead of "query"
                                    }
                                }
                                print(f"DEBUG: Created tool call: {tool_call}") if debug else None
                                completion_response.choices[0].message = litellmMessage(
                                    content=content,
                                    role="assistant",
                                    tool_calls=[ChatCompletionMessageToolCall(**tool_call)]
                                )
                                print(f"DEBUG: Updated message with tool calls: {completion_response.choices[0].message.tool_calls}") if debug else None
                                break
                else:
                    print(f"DEBUG: Tool calls already present: {completion_response.choices[0].message.tool_calls}") if debug else None

            return completion_response
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["openrouterexception", "bad request", "service unavailable"]):
                # First try OpenRouter thinking model
                self.logger.info("Primary provider failed, trying thinking model on OpenRouter...", title="MODEL UPGRADE", color="yellow")
                try:
                    thinking_params = {
                        "model": "openrouter/google/gemini-2.0-flash-thinking-exp:free",
                        "messages": messages,
                        "stream": stream,
                        "base_url": "https://openrouter.ai/api/v1",
                        "headers": {
                            "HTTP-Referer": "https://github.com/nerdsaver/AutoAgent",
                            "X-Title": "AutoAgent",
                            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                        }
                    }
                    return await acompletion(**thinking_params)
                except Exception:
                    # Fall back to Google AI Studio
                    self.logger.info("OpenRouter thinking model failed, falling back to Google AI Studio...", title="FAILOVER", color="yellow")
                    google_api_key = os.getenv("GOOGLE_API_KEY")
                    self.logger.info(f"Google API Key present: {bool(google_api_key)}", title="API Key Check", color="yellow")

                    backup_params = {
                        "model": "google/gemini-2.0-flash-thinking-exp-01-21",
                        "messages": messages,
                        "stream": stream,
                        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
                        "custom_llm_provider": "google",
                        "headers": {
                            "Content-Type": "application/json",
                            "x-goog-api-key": google_api_key
                        }
                    }

                    debug_params = {**backup_params}
                    if "api_key" in debug_params:
                        debug_params["api_key"] = "*****"
                    self.logger.info(f"Google AI Studio params: {json.dumps(debug_params, indent=2)}",
                                    title="Request Parameters", color="yellow")

                    try:
                        return await acompletion(**backup_params)
                    except Exception as backup_e:
                        self.logger.info(f"All providers failed. Last error: {str(backup_e)}",
                                        title="BACKUP ERROR", color="red")
                        raise backup_e
            else:
                raise e

    async def run_async(self, agent, messages, context_variables={}, model_override=None,
                       stream=False, debug=True, max_turns=float("inf"), execute_tools=True) -> Response:
        # Check if the agent is an OpenAIAssistantAgent
        from autoagent.agents.openai_assistant_agent import OpenAIAssistantAgent
        if isinstance(agent, OpenAIAssistantAgent):
            # For OpenAIAssistantAgent, we use its run_func method directly
            return await agent.run_func(messages=messages, context_variables=context_variables)
            
        assert not stream, "Async run does not support stream"
        active_agent = agent
        enter_agent = agent
        context_variables = copy.copy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info("Receiveing the task:", history[-1]['content'], title="Receive Task", color="green")

        while len(history) - init_len < max_turns and active_agent:
            completion = await self.get_chat_completion_async(
                agent=active_agent, history=history, context_variables=context_variables,
                model_override=model_override, stream=stream, debug=debug
            )
            message = completion.choices[0].message
            message.sender = active_agent.name
            self.logger.pretty_print_messages(message)
            history.append(json.loads(message.model_dump_json()))

            # Check for end conditions
            if enter_agent.tool_choice != "required":
                if (not message.tool_calls and active_agent.name == enter_agent.name) or not execute_tools:
                    self.logger.info("Ending turn.", title="End Turn", color="red")
                    break
            else:
                if message.tool_calls and message.tool_calls[0].function.name in ["case_resolved", "case_not_resolved"]:
                    self.logger.info(f"Ending turn with {message.tool_calls[0].function.name}.", title="End Turn", color="red")
                    partial_response = self.handle_tool_calls(
                        message.tool_calls, active_agent.functions, context_variables, debug,
                        handle_mm_func=active_agent.handle_mm_func
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break
                elif not execute_tools:
                    self.logger.info("Ending turn (execute_tools=False).", title="End Turn", color="red")
                    break

            # Process tool calls
            if message.tool_calls:
                partial_response = self.handle_tool_calls(
                    message.tool_calls, active_agent.functions, context_variables, debug,
                    handle_mm_func=active_agent.handle_mm_func
                )
            else:
                partial_response = Response(messages=[message])

            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        result = Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables
        )

        response_content = json.dumps(result, ensure_ascii=False, indent=2)
        return Response(
            messages=[{"role": "assistant", "content": response_content}],
            agent=active_agent,
            context_variables=context_variables
        )