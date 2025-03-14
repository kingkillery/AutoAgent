from autoagent.types import Agent
from autoagent.registry import register_agent
from autoagent.environment.web_surfer import WebSurfer
from typing import Dict, Any, Optional
import time
import os

@register_agent(name="UI Assistant", func_name="get_ui_assistant_agent")
def get_ui_assistant_agent(model: str = "gpt-4o", **kwargs):
    """
    Creates a UI Assistant agent that specializes in performing web tasks.
    This agent provides a user-friendly interface for web automation.
    
    Args:
        model: The model to use for the agent
        **kwargs: Additional keyword arguments
        
    Returns:
        An Agent instance configured as a UI Assistant
    """
    
    # Tool functions that use WebSurfer's execute_action method
    def visit_website(context_variables: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Navigate to a specified website URL
        
        Args:
            url: The URL to visit (include https:// for external sites)
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("visit_url", url=url)
    
    def click_element(context_variables: Dict[str, Any], element_id: str) -> Dict[str, Any]:
        """
        Click on an element on the current webpage
        
        Args:
            element_id: The ID of the element to click
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("click", bid=element_id)
    
    def type_text(context_variables: Dict[str, Any], element_id: str, text: str) -> Dict[str, Any]:
        """
        Type text into a form field or text area
        
        Args:
            element_id: The ID of the input element
            text: The text to type
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("input_text", bid=element_id, text=text)
    
    def search_web(context_variables: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Search the web for information
        
        Args:
            query: The search query
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("web_search", query=query, search_engine="google")
    
    def get_page_content(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the content of the current webpage
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        result = web_surfer.execute_action("get_page_content")
        if not result.get("error", False):
            return {
                "content": result.get("content", ""),
                "error": False
            }
        return result
    
    def scroll_down(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scroll down on the current webpage
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("page_down")
    
    def scroll_up(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scroll up on the current webpage
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("page_up")
    
    def go_back(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Go back to the previous webpage
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("history_back")
    
    def analyze_webpage(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current webpage and identify interactive elements
        """
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.analyze_page()
    
    def _get_or_create_websurfer(context_variables: Dict[str, Any]) -> WebSurfer:
        """Get or create a WebSurfer instance from context variables"""
        if "web_surfer" not in context_variables:
            # Get parameters from context or environment
            local_root = context_variables.get("local_root")
            if not local_root:
                local_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workspace")
                
            workplace_name = context_variables.get("workplace_name", "ui_assistant_workplace")
            
            # Create WebSurfer instance
            web_surfer = WebSurfer(
                local_root=local_root,
                workplace_name=workplace_name
            )
            
            # Store in context for future use
            context_variables["web_surfer"] = web_surfer
        
        return context_variables["web_surfer"]
    
    def instructions(context_variables):
        return """I am your UI assistant. I can perform web tasks for you.

I can help you with:
- Visiting websites
- Searching the web for information
- Filling out forms
- Clicking buttons and links
- Extracting information from webpages
- Navigating through websites

Please tell me what web task you'd like me to perform, and I'll guide you through the process.

When I perform web tasks, I'll describe what I'm doing and what I see on the webpage.
If you need me to interact with a specific element, I'll help identify it.

Let me know what you'd like me to help you with today!
"""
    
    tool_list = [
        visit_website,
        click_element,
        type_text,
        search_web,
        get_page_content,
        scroll_down,
        scroll_up,
        go_back,
        analyze_webpage
    ]
    
    return Agent(
        name="UI Assistant", 
        model=model, 
        instructions=instructions,
        functions=tool_list,
        tool_choice="auto", 
        parallel_tool_calls=False
    ) 