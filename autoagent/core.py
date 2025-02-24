# Standard library imports
import copy
import json
import os
from collections import defaultdict
from typing import List, Callable, Union
from datetime import datetime
# Local imports
from .util import function_to_json, debug_print, merge_chunk, pretty_print_messages
from .types import (
    Agent,
    AgentFunction,
    Message,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)
from litellm import completion, acompletion
from pathlib import Path
from .logger import MetaChainLogger, LoggerManager
from httpx import RemoteProtocolError, ConnectError
from litellm.exceptions import APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import AsyncOpenAI
import litellm
import inspect
from constant import MC_MODE, FN_CALL, API_BASE_URL, NOT_SUPPORT_SENDER, ADD_USER, NON_FN_CALL
from autoagent.fn_call_converter import convert_tools_to_description, convert_non_fncall_messages_to_fncall_messages, SYSTEM_PROMPT_SUFFIX_TEMPLATE, convert_fn_messages_to_non_fn_messages, interleave_user_into_messages
from litellm.types.utils import Message as litellmMessage

litellm.set_verbose = True  # Enable detailed logging

def should_retry_error(exception):
    if MC_MODE is False:
        print(f"Caught exception: {type(exception).__name__} - {str(exception)}")
    # Match more error types
    if isinstance(exception, (APIError, RemoteProtocolError, ConnectError)):
        return True
    # Match via error message
    error_msg = str(exception).lower()
    return any([
        "connection error" in error_msg,
        "server disconnected" in error_msg,
        "eof occurred" in error_msg,
        "timeout" in error_msg,
        "event loop is closed" in error_msg,  # handle event-loop issues
        "anthropicexception" in error_msg,    # handle Anthropic-related errors
        "openrouterexception" in error_msg,   # handle OpenRouter-specific errors
        "bad request" in error_msg,           # handle 400 errors that may be temporary
        "too many requests" in error_msg,     # explicitly handle rate limiting
        "service unavailable" in error_msg,   # handle 503 errors
        "gateway timeout" in error_msg,       # handle 504 errors
        "internal server error" in error_msg  # handle 500 errors
    ])

__CTX_VARS_NAME__ = "context_variables"
logger = LoggerManager.get_logger()

class MetaChain:
    def __init__(self, log_path: Union[str, None, MetaChainLogger] = None):
        """
        log_path: path of log file, None
        """
        if logger:
            self.logger = logger
        elif isinstance(log_path, MetaChainLogger):
            self.logger = log_path
        else:
            self.logger = MetaChainLogger(log_path=log_path)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=should_retry_error,
        before_sleep=lambda retry_state: print(f"Retrying... (attempt {retry_state.attempt_number})")
    )
    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> Message:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        if agent.examples:
            examples = agent.examples(context_variables) if callable(agent.examples) else agent.examples
            history = examples + history

        messages = [{"role": "system", "content": instructions}] + history

        tools = [function_to_json(f) for f in agent.functions]
        # Hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        try:
            if FN_CALL:
                create_model = model_override or agent.model
                assert litellm.supports_function_calling(model=create_model) == True, (
                    f"Model {create_model} does not support function calling, please set `FN_CALL=False` to use non-function calling mode"
                )
                create_params = {
                    "model": create_model,
                    "messages": messages,
                    "tools": tools or None,
                    "tool_choice": agent.tool_choice,
                    "stream": stream,
                    "base_url": "https://openrouter.ai/api/v1",
                    "headers": {
                        "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                        "X-Title": "AutoAgent",
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                    }
                }

                NO_SENDER_MODE = False
                for not_sender_model in NOT_SUPPORT_SENDER:
                    if not_sender_model in create_model:
                        NO_SENDER_MODE = True
                        break

                if NO_SENDER_MODE:
                    messages = create_params["messages"]
                    for message in messages:
                        if 'sender' in message:
                            del message['sender']
                    create_params["messages"] = messages

                # Ensure model name doesn't trigger OpenAI detection
                if "gpt" in create_model.lower() and not create_model.startswith(("openrouter/", "google/")):
                    create_model = f"openrouter/{create_model}"
                    create_params["model"] = create_model

                if tools and "gpt" in create_params["model"]:
                    create_params["parallel_tool_calls"] = agent.parallel_tool_calls

                # Use OpenRouter or Google AI Studio based on model prefix
                if create_model.startswith("google/"):
                    create_params["base_url"] = "https://generativelanguage.googleapis.com/v1beta/models"  # Remove trailing slash
                    create_params["headers"] = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {os.getenv('GOOGLE_API_KEY')}"
                    }
                else:
                    # Default to OpenRouter for all other models
                    create_params["base_url"] = "https://openrouter.ai/api/v1"
                    create_params["headers"] = {
                        "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                        "X-Title": "AutoAgent",
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                    }

                completion_response = completion(**create_params)

            else:
                create_model = model_override or agent.model
                assert agent.tool_choice == "required", (
                    f"Non-function calling mode MUST use tool_choice = 'required' rather than {agent.tool_choice}"
                )
                last_content = messages[-1]["content"]
                tools_description = convert_tools_to_description(tools)
                messages[-1]["content"] = (
                    last_content
                    + "\n[IMPORTANT] You MUST use the tools provided to complete the task.\n"
                    + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=tools_description)
                )

                NO_SENDER_MODE = False
                for not_sender_model in NOT_SUPPORT_SENDER:
                    if not_sender_model in create_model:
                        NO_SENDER_MODE = True
                        break

                if NO_SENDER_MODE:
                    for message in messages:
                        if 'sender' in message:
                            del message['sender']

                if NON_FN_CALL:
                    messages = convert_fn_messages_to_non_fn_messages(messages)
                if ADD_USER and messages[-1]["role"] != "user":
                    messages = interleave_user_into_messages(messages)

                create_params = {
                    "model": create_model,
                    "messages": messages,
                    "stream": stream,
                }

                # Add OpenRouter configuration if using an OpenRouter model
                if "openrouter/" in create_model or create_model.startswith("google/"):
                    create_params.update({
                        "base_url": "https://openrouter.ai/api/v1",
                        "headers": {
                            "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                            "X-Title": "AutoAgent",
                            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                        }
                    })
                else:
                    create_params["base_url"] = "https://generativelanguage.googleapis.com/v1beta/models/"
                    create_params["headers"] = {
                        "Authorization": f"Bearer {os.getenv('GOOGLE_API_KEY')}"
                    }

                completion_response = completion(**create_params)

                last_message = [
                    {"role": "assistant", "content": completion_response.choices[0].message.content}
                ]
                converted_message = convert_non_fncall_messages_to_fncall_messages(last_message, tools)
                # Use .get("tool_calls", []) to avoid KeyError
                converted_tool_calls = [
                    ChatCompletionMessageToolCall(**tool_call)
                    for tool_call in converted_message[0].get("tool_calls", [])
                ]
                completion_response.choices[0].message = litellmMessage(
                    content=converted_message[0]["content"],
                    role="assistant",
                    tool_calls=converted_tool_calls
                )

        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["openrouterexception", "bad request", "service unavailable"]):
                # First try the thinking model on OpenRouter
                self.logger.info("Primary provider failed, trying thinking model on OpenRouter...", title="MODEL UPGRADE", color="yellow")
                try:
                    thinking_params = {
                        "model": "openrouter/google/gemini-2.0-flash-thinking-exp:free",  # Added openrouter/ prefix
                        "messages": messages,
                        "stream": stream,
                        "base_url": "https://openrouter.ai/api/v1",
                        "headers": {
                            "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                            "X-Title": "AutoAgent",
                            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                        }
                    }
                    completion_response = completion(**thinking_params)
                    return completion_response
                except Exception as thinking_e:
                    # If OpenRouter thinking model fails, fall back to Google AI Studio
                    self.logger.info("OpenRouter thinking model failed, falling back to Google AI Studio...", title="FAILOVER", color="yellow")
                    try:
                        # Debug log the API key (safely)
                        google_api_key = os.getenv("GOOGLE_API_KEY")
                        self.logger.info(
                            f"Google API Key present: {bool(google_api_key)}", 
                            title="API Key Check",
                            color="yellow"
                        )
                        
                        backup_params = {
                            "model": "gemini-2.0-flash-thinking-exp-01-21",
                            "messages": messages,
                            "stream": stream,
                            "base_url": "https://generativelanguage.googleapis.com/v1beta/models",  # Remove trailing slash
                            "headers": {
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {google_api_key}"  # Use consistent auth method
                            }
                        }
                        
                        # Debug log the request parameters (excluding sensitive data)
                        debug_params = {**backup_params}
                        if "headers" in debug_params:
                            debug_params["headers"] = {
                                k: v for k, v in debug_params["headers"].items() 
                                if k != "Authorization"
                            }
                        self.logger.info(
                            f"Google AI Studio params: {json.dumps(debug_params, indent=2)}", 
                            title="Request Parameters",
                            color="yellow"
                        )
                        
                        completion_response = completion(**backup_params)
                        return completion_response
                    except Exception as backup_e:
                        # Log final error and raise
                        self.logger.info(
                            f"All providers failed. Last error: {str(backup_e)}", 
                            title="BACKUP ERROR", 
                            color="red"
                        )
                        raise backup_e
            else:
                raise e

        return completion_response

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )

            case _:
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

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
        handle_mm_func: Callable[[], str] = None,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # Handle missing tool case, skip to next tool
            if name not in function_map:
                self.logger.info(
                    f"Tool {name} not found in function map. You are recommended to use `run_tool` to run this tool.",
                    title="Tool Call Error",
                    color="red",
                )
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": f"Error: Tool {name} not found. You are recommended to use `run_tool` to run this tool.",
                    }
                )
                continue

            args = json.loads(tool_call.function.arguments)
            func = function_map[name]

            # Pass context_variables to agent functions if signature allows
            if __CTX_VARS_NAME__ in inspect.signature(func).parameters.keys():
                args[__CTX_VARS_NAME__] = context_variables

            raw_result = function_map[name](**args)
            result: Result = self.handle_function_result(raw_result, debug)

            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result.value,
                }
            )
            self.logger.pretty_print_messages(partial_response.messages[-1])

            if result.image:
                assert handle_mm_func, (
                    f"handle_mm_func is not provided, but an image is returned by tool call {name}({tool_call.function.arguments})"
                )
                partial_response.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": handle_mm_func(name, tool_call.function.arguments)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{result.image}"
                                }
                            },
                        ],
                    }
                )

            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
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
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # Get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
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

            # Convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"],
                    function=function,
                    type=tool_call["type"],
                )
                tool_calls.append(tool_call_object)

            # Handle function calls, update context_variables, possibly switch agents
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
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = True,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )

        active_agent = agent
        enter_agent = agent
        context_variables = copy.copy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info("Receiveing the task:", history[-1]['content'], title="Receive Task", color="green")

        while len(history) - init_len < max_turns and active_agent:
            # Get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message: Message = completion.choices[0].message
            message.sender = active_agent.name
            self.logger.pretty_print_messages(message)
            history.append(json.loads(message.model_dump_json()))  # to avoid OpenAI types (?)

            if enter_agent.tool_choice != "required":
                if (not message.tool_calls and active_agent.name == enter_agent.name) or not execute_tools:
                    self.logger.info("Ending turn.", title="End Turn", color="red")
                    break
            else:
                if (message.tool_calls and message.tool_calls[0].function.name == "case_resolved") or not execute_tools:
                    self.logger.info("Ending turn with case resolved.", title="End Turn", color="red")
                    partial_response = self.handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break
                elif (message.tool_calls and message.tool_calls[0].function.name == "case_not_resolved") or not execute_tools:
                    self.logger.info("Ending turn with case not resolved.", title="End Turn", color="red")
                    partial_response = self.handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break

            if message.tool_calls:
                partial_response = self.handle_tool_calls(
                    message.tool_calls,
                    active_agent.functions,
                    context_variables,
                    debug,
                    handle_mm_func=active_agent.handle_mm_func,
                )
            else:
                partial_response = Response(messages=[message])

            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=10, max=180),
        retry=should_retry_error,
        before_sleep=lambda retry_state: print(f"Retrying... (attempt {retry_state.attempt_number})")
    )
    async def get_chat_completion_async(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> Message:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        if agent.examples:
            examples = agent.examples(context_variables) if callable(agent.examples) else agent.examples
            history = examples + history

        messages = [{"role": "system", "content": instructions}] + history

        tools = [function_to_json(f) for f in agent.functions]
        # Hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        try:
            if FN_CALL:
                create_model = model_override or agent.model
                assert litellm.supports_function_calling(model=create_model) == True, (
                    f"Model {create_model} does not support function calling, please set `FN_CALL=False` to use non-function calling mode"
                )

                create_params = {
                    "model": create_model,
                    "messages": messages,
                    "tools": tools or None,
                    "tool_choice": agent.tool_choice,
                    "stream": stream,
                    "base_url": "https://openrouter.ai/api/v1",
                    "headers": {
                        "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                        "X-Title": "AutoAgent",
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                    }
                }

                NO_SENDER_MODE = False
                for not_sender_model in NOT_SUPPORT_SENDER:
                    if not_sender_model in create_model:
                        NO_SENDER_MODE = True
                        break

                if NO_SENDER_MODE:
                    messages = create_params["messages"]
                    for message in messages:
                        if 'sender' in message:
                            del message['sender']
                    create_params["messages"] = messages

                # Ensure model name doesn't trigger OpenAI detection
                if "gpt" in create_model.lower() and not create_model.startswith(("openrouter/", "google/")):
                    create_model = f"openrouter/{create_model}"
                    create_params["model"] = create_model

                if tools and "gpt" in create_params["model"]:
                    create_params["parallel_tool_calls"] = agent.parallel_tool_calls

                # Use OpenRouter or Google AI Studio based on model prefix
                if create_model.startswith("google/"):
                    create_params["base_url"] = "https://generativelanguage.googleapis.com/v1beta/models"  # Remove trailing slash
                    create_params["headers"] = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {os.getenv('GOOGLE_API_KEY')}"
                    }
                else:
                    # Default to OpenRouter for all other models
                    create_params["base_url"] = "https://openrouter.ai/api/v1"
                    create_params["headers"] = {
                        "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                        "X-Title": "AutoAgent",
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                    }

                completion_response = await acompletion(**create_params)

            else:
                create_model = model_override or agent.model
                assert agent.tool_choice == "required", (
                    f"Non-function calling mode MUST use tool_choice = 'required' rather than {agent.tool_choice}"
                )

                last_content = messages[-1]["content"]
                tools_description = convert_tools_to_description(tools)
                messages[-1]["content"] = (
                    last_content
                    + "\n[IMPORTANT] You MUST use the tools provided to complete the task.\n"
                    + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=tools_description)
                )

                NO_SENDER_MODE = False
                for not_sender_model in NOT_SUPPORT_SENDER:
                    if not_sender_model in create_model:
                        NO_SENDER_MODE = True
                        break

                if NO_SENDER_MODE:
                    for message in messages:
                        if 'sender' in message:
                            del message['sender']

                if NON_FN_CALL:
                    messages = convert_fn_messages_to_non_fn_messages(messages)
                if ADD_USER and messages[-1]["role"] != "user":
                    messages = interleave_user_into_messages(messages)

                create_params = {
                    "model": create_model,
                    "messages": messages,
                    "stream": stream,
                    "base_url": "https://openrouter.ai/api/v1",
                    "headers": {
                        "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                        "X-Title": "AutoAgent",
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                    }
                }

                # Use OpenRouter or Google AI Studio based on model prefix
                if create_model.startswith("google/"):
                    create_params["base_url"] = "https://generativelanguage.googleapis.com/v1beta/models"  # Remove trailing slash
                    create_params["headers"] = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {os.getenv('GOOGLE_API_KEY')}"
                    }
                else:
                    # Default to OpenRouter for all other models
                    create_params["base_url"] = "https://openrouter.ai/api/v1"
                    create_params["headers"] = {
                        "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                        "X-Title": "AutoAgent",
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                    }

                completion_response = await acompletion(**create_params)

                last_message = [{"role": "assistant", "content": completion_response.choices[0].message.content}]
                converted_message = convert_non_fncall_messages_to_fncall_messages(last_message, tools)
                # Use .get("tool_calls", []) to avoid KeyError
                converted_tool_calls = [
                    ChatCompletionMessageToolCall(**tool_call)
                    for tool_call in converted_message[0].get("tool_calls", [])
                ]
                completion_response.choices[0].message = litellmMessage(
                    content=converted_message[0]["content"],
                    role="assistant",
                    tool_calls=converted_tool_calls
                )

        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["openrouterexception", "bad request", "service unavailable"]):
                # First try the thinking model on OpenRouter
                self.logger.info("Primary provider failed, trying thinking model on OpenRouter...", title="MODEL UPGRADE", color="yellow")
                try:
                    thinking_params = {
                        "model": "openrouter/google/gemini-2.0-flash-thinking-exp:free",  # Added openrouter/ prefix
                        "messages": messages,
                        "stream": stream,
                        "base_url": "https://openrouter.ai/api/v1",
                        "headers": {
                            "HTTP-Referer": "https://github.com/prestoncn/AutoAgent",
                            "X-Title": "AutoAgent",
                            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
                        }
                    }
                    completion_response = await acompletion(**thinking_params)
                    return completion_response
                except Exception as thinking_e:
                    # If OpenRouter thinking model fails, fall back to Google AI Studio
                    self.logger.info("OpenRouter thinking model failed, falling back to Google AI Studio...", title="FAILOVER", color="yellow")
                    try:
                        # Debug log the API key (safely)
                        google_api_key = os.getenv("GOOGLE_API_KEY")
                        self.logger.info(
                            f"Google API Key present: {bool(google_api_key)}", 
                            title="API Key Check",
                            color="yellow"
                        )
                        
                        backup_params = {
                            "model": "gemini-2.0-flash-thinking-exp-01-21",
                            "messages": messages,
                            "stream": stream,
                            "base_url": "https://generativelanguage.googleapis.com/v1beta/models",  # Remove trailing slash
                            "headers": {
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {google_api_key}"  # Use consistent auth method
                            }
                        }
                        
                        # Debug log the request parameters (excluding sensitive data)
                        debug_params = {**backup_params}
                        if "headers" in debug_params:
                            debug_params["headers"] = {
                                k: v for k, v in debug_params["headers"].items() 
                                if k != "Authorization"
                            }
                        self.logger.info(
                            f"Google AI Studio params: {json.dumps(debug_params, indent=2)}", 
                            title="Request Parameters",
                            color="yellow"
                        )
                        
                        completion_response = completion(**backup_params)
                        return completion_response
                    except Exception as backup_e:
                        # Log final error and raise
                        self.logger.info(
                            f"All providers failed. Last error: {str(backup_e)}", 
                            title="BACKUP ERROR", 
                            color="red"
                        )
                        raise backup_e
            else:
                raise e

        return completion_response

    async def run_async(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = True,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        assert stream == False, "Async run does not support stream"
        active_agent = agent
        enter_agent = agent
        context_variables = copy.copy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info("Receiveing the task:", history[-1]['content'], title="Receive Task", color="green")

        while len(history) - init_len < max_turns and active_agent:
            completion = await self.get_chat_completion_async(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message: Message = completion.choices[0].message
            message.sender = active_agent.name
            self.logger.pretty_print_messages(message)
            history.append(json.loads(message.model_dump_json()))

            if enter_agent.tool_choice != "required":
                if (not message.tool_calls and active_agent.name == enter_agent.name) or not execute_tools:
                    self.logger.info("Ending turn.", title="End Turn", color="red")
                    break
            else:
                if (message.tool_calls and message.tool_calls[0].function.name == "case_resolved") or not execute_tools:
                    self.logger.info("Ending turn with case resolved.", title="End Turn", color="red")
                    partial_response = self.handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break
                elif (message.tool_calls and message.tool_calls[0].function.name == "case_not_resolved") or not execute_tools:
                    self.logger.info("Ending turn with case not resolved.", title="End Turn", color="red")
                    partial_response = self.handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    break

            if message.tool_calls:
                partial_response = self.handle_tool_calls(
                    message.tool_calls,
                    active_agent.functions,
                    context_variables,
                    debug,
                    handle_mm_func=active_agent.handle_mm_func,
                )
            else:
                partial_response = Response(messages=[message])

            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )

