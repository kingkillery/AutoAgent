import os
import litellm
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Enable verbose logging
litellm.set_verbose = True

def test_gemini_tool_calls():
    # System message with instruction to use tools
    system_message = {
        "role": "system", 
        "content": """You are a helpful assistant that can help the user with their request.
Based on the state of solving user's task, your responsibility is to determine which agent is best suited to handle the user's request under the current context, and transfer the conversation to that agent. And you should not stop to try to solve the user's request by transferring to another agent only until the task is completed.

There are three agents you can transfer to:
1. use `transfer_to_filesurfer_agent` to transfer to File Surfer Agent, it can help you to open any type of local files and browse the content of them.
2. use `transfer_to_websurfer_agent` to transfer to Web Surfer Agent, it can help you to open any website and browse any content on it.
3. use `transfer_to_coding_agent` to transfer to Coding Agent, it can help you to write code to solve the user's request, especially some complex tasks.
"""
    }
    
    # User request that requires tool use
    user_message = {
        "role": "user", 
        "content": "Please visit reddit.com in the web browser and share a *link* to each story on the front page"
    }
    
    # Define the tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "transfer_to_websurfer_agent",
                "description": "Transfer to Web Surfer Agent to browse websites",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user query to transfer to the Web Surfer Agent"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transfer_to_filesurfer_agent",
                "description": "Transfer to File Surfer Agent to browse local files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user query to transfer to the File Surfer Agent"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transfer_to_coding_agent",
                "description": "Transfer to Coding Agent to write code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user query to transfer to the Coding Agent"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # First, try with function calling
    try:
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        params = {
            "model": "gemini/gemini-2.0-flash",
            "messages": [system_message, user_message],
            "stream": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 8192,
            "tools": tools,
            "headers": {
                "x-goog-api-key": google_api_key,
                "Content-Type": "application/json"
            }
        }
        
        print("Sending request to Gemini with the following parameters:")
        print(json.dumps({k: v for k, v in params.items() if k != "headers"}, indent=2))
        
        response = litellm.completion(**params)
        
        print("\nResponse:")
        print(f"Content: {response.choices[0].message.content}")
        print(f"Tool calls: {response.choices[0].message.tool_calls}")
        
        # Try without using tools param
        del params["tools"]
        print("\n\nTrying again without tools parameter:")
        response_no_tools = litellm.completion(**params)
        
        print("\nResponse without tools:")
        print(f"Content: {response_no_tools.choices[0].message.content}")
        print(f"Tool calls: {response_no_tools.choices[0].message.tool_calls}")
        
        # Try with non-function calling mode
        print("\n\nTrying in non-function calling mode:")
        system_with_suffix = {
            "role": "system",
            "content": system_message["content"] + "\n\nYou have access to the following functions:\n\n" + 
                      "1. transfer_to_filesurfer_agent(query: string): Transfer to File Surfer Agent\n" +
                      "2. transfer_to_websurfer_agent(query: string): Transfer to Web Surfer Agent\n" +
                      "3. transfer_to_coding_agent(query: string): Transfer to Coding Agent\n\n" +
                      "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
                      "<function=function_name>\n<parameter=parameter_name>parameter_value</parameter>\n</function>"
        }
        
        params_non_fn = {
            "model": "gemini/gemini-2.0-flash",
            "messages": [system_with_suffix, user_message],
            "stream": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 8192,
            "headers": {
                "x-goog-api-key": google_api_key,
                "Content-Type": "application/json"
            }
        }
        
        response_non_fn = litellm.completion(**params_non_fn)
        
        print("\nResponse in non-function calling mode:")
        print(f"Content: {response_non_fn.choices[0].message.content}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_gemini_tool_calls() 