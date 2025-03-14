from autoagent.types import Agent
from autoagent.registry import register_agent
from autoagent.environment.web_surfer import WebSurfer
from autoagent.environment.api_handlers import register_api_handlers
from typing import Dict, Any, Optional
import time
import os

@register_agent(name="Web Surfer Agent", func_name="get_websurfer_agent")
def get_websurfer_agent(model: str = "gpt-4o", **kwargs):
    
    # Tool functions that use WebSurfer's execute_action method
    def visit_url(context_variables: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Navigate to a specified URL"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("visit_url", url=url)
    
    def click(context_variables: Dict[str, Any], bid: str, button: str = "left") -> Dict[str, Any]:
        """Click an element on the page"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("click", bid=bid, button=button)
    
    def input_text(context_variables: Dict[str, Any], bid: str, text: str) -> Dict[str, Any]:
        """Input text into a form field"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("input_text", bid=bid, text=text)
    
    def web_search(context_variables: Dict[str, Any], query: str, search_engine: str = "google") -> Dict[str, Any]:
        """Perform a web search using the specified search engine"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("web_search", query=query, search_engine=search_engine)
    
    def get_page_markdown(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Get the markdown representation of the current page"""
        web_surfer = _get_or_create_websurfer(context_variables)
        result = web_surfer.execute_action("get_page_content")
        # Return content in the expected format
        if not result.get("error", False):
            return {
                "markdown": result.get("content", ""),
                "error": False
            }
        return result
    
    def page_down(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll down on the page"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("page_down")
    
    def page_up(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll up on the page"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("page_up")
    
    def history_back(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Go back in browser history"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.execute_action("history_back")
    
    def history_forward(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Go forward in browser history"""
        web_surfer = _get_or_create_websurfer(context_variables)
        # Use history_back for now since history_forward isn't implemented in WebSurfer
        return {"error": True, "message": "history_forward not yet implemented in WebSurfer"}
    
    def sleep(context_variables: Dict[str, Any], seconds: float = 1.0) -> Dict[str, Any]:
        """Sleep for the specified number of seconds"""
        time.sleep(seconds)
        return {"result": f"Slept for {seconds} seconds", "error": False}
    
    def analyze_page(context_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current page and extract useful information like links, forms, and interactive elements"""
        web_surfer = _get_or_create_websurfer(context_variables)
        return web_surfer.analyze_page()
    
    def _get_or_create_websurfer(context_variables: Dict[str, Any]) -> WebSurfer:
        """Get or create a WebSurfer instance from context variables"""
        if "web_surfer" not in context_variables:
            # Get parameters from context or environment
            local_root = context_variables.get("local_root")
            if not local_root and "LOCAL_ROOT" in globals():
                local_root = globals()["LOCAL_ROOT"]
            # Provide a default value if still None
            if local_root is None:
                local_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "workspace")
                
            workplace_name = context_variables.get("workplace_name")
            if not workplace_name and "DOCKER_WORKPLACE_NAME" in globals():
                workplace_name = globals()["DOCKER_WORKPLACE_NAME"]
            # Provide a default value if still None
            if workplace_name is None:
                workplace_name = "default_workplace"
            
            # Create WebSurfer instance
            web_surfer = WebSurfer(
                local_root=local_root,
                workplace_name=workplace_name
            )
            
            # Register API handlers if credentials are available (future enhancement)
            google_api_key = context_variables.get("google_api_key")
            google_cse_id = context_variables.get("google_cse_id")
            if google_api_key and google_cse_id and "register_api_handlers" in globals():
                register_api_handlers(web_surfer, google_api_key, google_cse_id)
            
            # Store in context for future use
            context_variables["web_surfer"] = web_surfer
        
        return context_variables["web_surfer"]
    
    def handle_mm_func(tool_name, tool_args):
        return f"After taking action `{tool_name}({tool_args})`, the image of current page is shown below. Please take the next action based on the image, the current state of the page, and previous actions and observations."
    
    def instructions(context_variables):
        workplace_name = None
        if "web_env" in context_variables and hasattr(context_variables["web_env"], "docker_workplace"):
            workplace_name = context_variables["web_env"].docker_workplace
        elif "workplace_name" in context_variables:
            workplace_name = context_variables["workplace_name"]
        elif "DOCKER_WORKPLACE_NAME" in globals():
            workplace_name = globals()["DOCKER_WORKPLACE_NAME"]
            
        downloads_path = f"{workplace_name}/downloads" if workplace_name else "downloads"
        
        return f"""Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

Note that if you want to analyze a YouTube video, Wikipedia page, or other pages with media content, or analyze text content in detail, use the `get_page_markdown` tool to convert the page to markdown text.

When browsing the web, downloaded files will be saved to `{downloads_path}`. You CANNOT open these files directly - transfer back to the `System Triage Agent` and let them transfer to the `File Surfer Agent` to open the downloaded files.

This agent uses an improved WebSurfer implementation with:
- Better error handling and retry mechanisms
- More reliable navigation
- Advanced page analysis capabilities

Use `analyze_page` to get comprehensive information about the current page including links, forms, and interactive elements.

When you've completed your task, use `transfer_back_to_triage_agent` to return to the `System Triage Agent`. Don't transfer back until the task is fully completed.
"""
    
    tool_list = [
        visit_url,
        click,
        input_text,
        web_search,
        get_page_markdown,
        page_down,
        page_up,
        history_back,
        sleep,
        analyze_page
    ]
    
    return Agent(
        name="Web Surfer Agent", 
        model=model, 
        instructions=instructions,
        functions=tool_list,
        handle_mm_func=handle_mm_func,
        tool_choice="required", 
        parallel_tool_calls=False
    )