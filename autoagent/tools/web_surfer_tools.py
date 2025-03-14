"""
WebSurfer Tools - Improved browser automation tools with better reliability and error handling.
These tools use the WebSurfer class which in turn uses BrowserManager and BrowserController.
"""

from autoagent.registry import register_tool
from autoagent.types import Result
# Use relative import to avoid circular imports
from ..environment.web_surfer import WebSurfer
from typing import Dict, Any, Optional
import traceback
import json
import os

# Global WebSurfer instance
_web_surfer = None

def get_web_surfer(context_variables) -> WebSurfer:
    """
    Get the WebSurfer instance, creating it if necessary.
    Uses lazy initialization - a browser is only created when operations actually need it.
    
    Args:
        context_variables: Context variables
        
    Returns:
        WebSurfer instance
    """
    global _web_surfer
    
    if _web_surfer is None:
        # Get initialization parameters from context_variables
        browsergym_eval_env = context_variables.get("browsergym_eval_env", None)
        
        # Ensure local_root is a valid path
        local_root = context_variables.get("local_root", None)
        if local_root is None:
            local_root = os.path.expanduser("~")
        
        # Ensure workplace_name is a valid string
        workplace_name = context_variables.get("workplace_name", "websurfer_workplace")
        
        # Create WebSurfer instance
        _web_surfer = WebSurfer(
            browsergym_eval_env=browsergym_eval_env,
            local_root=local_root,
            workplace_name=workplace_name
        )
    
    return _web_surfer


@register_tool("wsurfer_visit_url")
def wsurfer_visit_url(context_variables, url: str) -> Result:
    """
    Navigate directly to a provided URL using the browser's address bar.
    Much more reliable than the previous implementation with better error handling and retries.
    
    Args:
        url: The URL to navigate to.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("visit_url", url=url)
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error navigating to {url}: {error_message}",
                image=None
            )
        
        return Result(
            value=f"Successfully navigated to {url}",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error navigating to {url}: {str(e)}",
            image=None
        )


@register_tool("wsurfer_click")
def wsurfer_click(context_variables, bid: str, button: str = "left") -> Result:
    """
    Click an element on the page with improved reliability.
    
    Args:
        bid: The browser ID of the element to click.
        button: The mouse button to use ("left", "middle", or "right").
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("click", bid=bid, button=button)
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error clicking element {bid}: {error_message}",
                image=None
            )
        
        return Result(
            value=f"Successfully clicked element {bid}",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error clicking element {bid}: {str(e)}",
            image=None
        )


@register_tool("wsurfer_input_text")
def wsurfer_input_text(context_variables, bid: str, text: str) -> Result:
    """
    Input text into a form field with improved reliability.
    
    Args:
        bid: The browser ID of the input field.
        text: The text to input.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("input_text", bid=bid, text=text)
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error inputting text into element {bid}: {error_message}",
                image=None
            )
        
        return Result(
            value=f"Successfully input text into element {bid}",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error inputting text into element {bid}: {str(e)}",
            image=None
        )


@register_tool("wsurfer_web_search")
def wsurfer_web_search(context_variables, query: str, search_engine: str = "google") -> Result:
    """
    Perform a web search with improved reliability.
    
    Args:
        query: The search query.
        search_engine: The search engine to use ("google", "bing", or "duckduckgo").
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("web_search", query=query, search_engine=search_engine)
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error performing web search for '{query}': {error_message}",
                image=None
            )
        
        return Result(
            value=f"Successfully performed web search for '{query}'",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error performing web search for '{query}': {str(e)}",
            image=None
        )


@register_tool("wsurfer_get_page_content")
def wsurfer_get_page_content(context_variables) -> Result:
    """
    Get the content of the current page with improved reliability.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("get_page_content")
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error getting page content: {error_message}",
                image=None
            )
        
        content = result.get("content", "")
        url = result.get("url", "")
        
        return Result(
            value=f"Current page: {url}\n\n{content}",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error getting page content: {str(e)}",
            image=None
        )


@register_tool("wsurfer_page_down")
def wsurfer_page_down(context_variables) -> Result:
    """
    Scroll down on the page with improved reliability.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("page_down")
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error scrolling down: {error_message}",
                image=None
            )
        
        return Result(
            value="Successfully scrolled down",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error scrolling down: {str(e)}",
            image=None
        )


@register_tool("wsurfer_page_up")
def wsurfer_page_up(context_variables) -> Result:
    """
    Scroll up on the page with improved reliability.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("page_up")
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error scrolling up: {error_message}",
                image=None
            )
        
        return Result(
            value="Successfully scrolled up",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error scrolling up: {str(e)}",
            image=None
        )


@register_tool("wsurfer_history_back")
def wsurfer_history_back(context_variables) -> Result:
    """
    Go back in browser history with improved reliability.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("history_back")
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error going back in history: {error_message}",
                image=None
            )
        
        return Result(
            value="Successfully went back in history",
            image=result.get("screenshot", None)
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error going back in history: {str(e)}",
            image=None
        )


@register_tool("wsurfer_restart_browser")
def wsurfer_restart_browser(context_variables) -> Result:
    """
    Restart the browser instance if it's in a bad state.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        web_surfer.restart_browser()
        return Result(
            value="Browser restarted successfully",
            image=None
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error restarting browser: {str(e)}",
            image=None
        )


@register_tool("wsurfer_analyze_page")
def wsurfer_analyze_page(context_variables) -> Result:
    """
    Analyze the current page to extract meaningful information about its structure and interactive elements.
    This provides a comprehensive overview of the page including content, links, forms, and interactive elements.
    Use this to understand what actions are possible on the current page.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.analyze_page()
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error analyzing page: {error_message}",
                image=None
            )
        
        # Extract the page details
        title = result.get("title", "")
        url = result.get("url", "")
        content_preview = result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", "")
        links = result.get("links", [])
        forms = result.get("forms", [])
        elements = result.get("elements", [])
        
        # Format a summary of the page
        summary = [f"# Page Analysis: {title}", f"URL: {url}", ""]
        
        # Add interactive elements summary
        if elements:
            summary.append("## Interactive Elements")
            for i, elem in enumerate(elements[:10]):  # Limit to first 10 elements
                elem_type = elem.get("type", "").lower()
                elem_text = elem.get("text", "").strip()[:50]
                elem_bid = elem.get("bid", "")
                
                if elem_bid:
                    summary.append(f"{i+1}. {elem_type}: '{elem_text}' (BID: {elem_bid})")
            
            if len(elements) > 10:
                summary.append(f"... and {len(elements) - 10} more elements")
            summary.append("")
        
        # Add forms summary
        if forms:
            summary.append("## Forms")
            for i, form in enumerate(forms):
                form_id = form.get("id", "")
                form_action = form.get("action", "")
                form_bid = form.get("bid", "")
                
                summary.append(f"Form {i+1}: ID={form_id}, Action={form_action}, BID={form_bid}")
                
                # List form inputs
                inputs = form.get("inputs", [])
                if inputs:
                    summary.append("  Inputs:")
                    for input_item in inputs[:5]:  # Limit to first 5 inputs
                        input_type = input_item.get("type", "")
                        input_name = input_item.get("name", "")
                        input_bid = input_item.get("bid", "")
                        summary.append(f"  - {input_type}: {input_name} (BID: {input_bid})")
                    
                    if len(inputs) > 5:
                        summary.append(f"  ... and {len(inputs) - 5} more inputs")
                
                # List submit buttons
                submit_buttons = form.get("submitButtons", [])
                if submit_buttons:
                    summary.append("  Submit Buttons:")
                    for button in submit_buttons:
                        button_text = button.get("text", "")
                        button_bid = button.get("bid", "")
                        summary.append(f"  - {button_text} (BID: {button_bid})")
                summary.append("")
        
        # Add links summary
        if links:
            summary.append("## Links")
            for i, link in enumerate(links[:10]):  # Limit to first 10 links
                link_text = link.get("text", "").strip()[:50]
                link_href = link.get("href", "")
                link_bid = link.get("bid", "")
                
                summary.append(f"{i+1}. '{link_text}' -> {link_href} (BID: {link_bid})")
            
            if len(links) > 10:
                summary.append(f"... and {len(links) - 10} more links")
            summary.append("")
        
        # Add content preview
        summary.append("## Content Preview")
        summary.append(content_preview)
        
        return Result(
            value="\n".join(summary),
            image=web_surfer.last_observation.get("screenshot", None) if web_surfer.last_observation else None
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error analyzing page: {str(e)}",
            image=None
        )


@register_tool("wsurfer_find_elements")
def wsurfer_find_elements(context_variables, element_type: str = None) -> Result:
    """
    Find interactive elements on the current page, optionally filtered by element type.
    This helps identify elements that can be clicked, forms that can be filled, etc.
    
    Args:
        element_type: Optional HTML element type to filter by (e.g., "button", "a", "input")
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("find_elements", element_type=element_type)
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error finding elements: {error_message}",
                image=None
            )
        
        elements = result.get("elements", [])
        
        if not elements:
            return Result(
                value=f"No {'interactable ' if not element_type else ''}{element_type or ''} elements found on the page.",
                image=web_surfer.last_observation.get("screenshot", None) if web_surfer.last_observation else None
            )
        
        # Format elements as a structured list
        element_type_str = f"{element_type} " if element_type else "interactable "
        header = f"Found {len(elements)} {element_type_str}elements on the page:"
        formatted_elements = []
        
        for i, elem in enumerate(elements):
            elem_type = elem.get("type", "").lower()
            elem_text = elem.get("text", "").strip()[:50] if elem.get("text") else ""
            elem_bid = elem.get("bid", "")
            elem_id = elem.get("id", "")
            elem_name = elem.get("name", "")
            elem_value = elem.get("value", "")
            elem_href = elem.get("href", "")
            
            elem_details = []
            if elem_text:
                elem_details.append(f"Text: '{elem_text}'")
            if elem_bid:
                elem_details.append(f"BID: {elem_bid}")
            if elem_id:
                elem_details.append(f"ID: {elem_id}")
            if elem_name:
                elem_details.append(f"Name: {elem_name}")
            if elem_value:
                elem_details.append(f"Value: {elem_value}")
            if elem_href:
                elem_details.append(f"URL: {elem_href}")
            
            formatted_elements.append(f"{i+1}. {elem_type.upper()}: {', '.join(elem_details)}")
        
        return Result(
            value=header + "\n\n" + "\n".join(formatted_elements),
            image=web_surfer.last_observation.get("screenshot", None) if web_surfer.last_observation else None
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error finding elements: {str(e)}",
            image=None
        )


@register_tool("wsurfer_find_forms")
def wsurfer_find_forms(context_variables) -> Result:
    """
    Find and analyze forms on the current page.
    This helps identify forms, their inputs, and submit buttons to interact with them.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("find_forms")
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error finding forms: {error_message}",
                image=None
            )
        
        forms = result.get("forms", [])
        
        if not forms:
            return Result(
                value="No forms found on the page.",
                image=web_surfer.last_observation.get("screenshot", None) if web_surfer.last_observation else None
            )
        
        # Format forms as a detailed description
        header = f"Found {len(forms)} forms on the page:"
        formatted_forms = []
        
        for i, form in enumerate(forms):
            form_id = form.get("id", "")
            form_action = form.get("action", "")
            form_method = form.get("method", "")
            form_bid = form.get("bid", "")
            
            form_header = f"Form {i+1}:"
            form_details = []
            
            if form_id:
                form_details.append(f"ID: {form_id}")
            if form_action:
                form_details.append(f"Action: {form_action}")
            if form_method:
                form_details.append(f"Method: {form_method}")
            if form_bid:
                form_details.append(f"BID: {form_bid}")
            
            formatted_form = [form_header + " " + ", ".join(form_details)]
            
            # Add form inputs
            inputs = form.get("inputs", [])
            if inputs:
                formatted_form.append("  Inputs:")
                for j, input_item in enumerate(inputs):
                    input_type = input_item.get("type", "")
                    input_name = input_item.get("name", "")
                    input_id = input_item.get("id", "")
                    input_bid = input_item.get("bid", "")
                    input_placeholder = input_item.get("placeholder", "")
                    input_required = "Required" if input_item.get("required", False) else "Optional"
                    
                    input_details = []
                    if input_name:
                        input_details.append(f"Name: {input_name}")
                    if input_id:
                        input_details.append(f"ID: {input_id}")
                    if input_bid:
                        input_details.append(f"BID: {input_bid}")
                    if input_placeholder:
                        input_details.append(f"Placeholder: '{input_placeholder}'")
                    
                    formatted_form.append(f"  {j+1}. {input_type.upper()} - {', '.join(input_details)} ({input_required})")
            
            # Add submit buttons
            submit_buttons = form.get("submitButtons", [])
            if submit_buttons:
                formatted_form.append("  Submit Buttons:")
                for j, button in enumerate(submit_buttons):
                    button_type = button.get("type", "")
                    button_text = button.get("text", "")
                    button_bid = button.get("bid", "")
                    
                    formatted_form.append(f"  {j+1}. {button_type.upper()}: '{button_text}' (BID: {button_bid})")
            
            formatted_forms.append("\n".join(formatted_form))
        
        return Result(
            value=header + "\n\n" + "\n\n".join(formatted_forms),
            image=web_surfer.last_observation.get("screenshot", None) if web_surfer.last_observation else None
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error finding forms: {str(e)}",
            image=None
        )


@register_tool("wsurfer_extract_links")
def wsurfer_extract_links(context_variables) -> Result:
    """
    Extract all links from the current page with their text and URLs.
    This helps identify navigation options and clickable links.
    """
    web_surfer = get_web_surfer(context_variables)
    
    try:
        result = web_surfer.execute_action("extract_links")
        
        if result.get("error", False):
            error_message = result.get("last_action_error", "Unknown error")
            return Result(
                value=f"Error extracting links: {error_message}",
                image=None
            )
        
        links = result.get("links", [])
        
        if not links:
            return Result(
                value="No links found on the page.",
                image=web_surfer.last_observation.get("screenshot", None) if web_surfer.last_observation else None
            )
        
        # Format links as a list
        header = f"Found {len(links)} links on the page:"
        formatted_links = []
        
        for i, link in enumerate(links):
            link_text = link.get("text", "").strip()[:100] if link.get("text") else "[No text]"
            link_href = link.get("href", "")
            link_bid = link.get("bid", "")
            link_title = link.get("title", "")
            
            link_details = []
            if link_bid:
                link_details.append(f"BID: {link_bid}")
            if link_title:
                link_details.append(f"Title: '{link_title}'")
            
            details_str = f" ({', '.join(link_details)})" if link_details else ""
            formatted_links.append(f"{i+1}. '{link_text}' -> {link_href}{details_str}")
        
        return Result(
            value=header + "\n\n" + "\n".join(formatted_links),
            image=web_surfer.last_observation.get("screenshot", None) if web_surfer.last_observation else None
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error extracting links: {str(e)}",
            image=None
        )


@register_tool("wsurfer_close_browser")
def wsurfer_close_browser(context_variables) -> Result:
    """
    Explicitly close the browser instance to free up resources.
    Use this when you're done with browser operations to prevent unused browser instances.
    """
    global _web_surfer
    
    if _web_surfer is None:
        return Result(
            value="No browser instance to close.",
            image=None
        )
    
    try:
        _web_surfer.close()
        _web_surfer = None
        
        return Result(
            value="Browser closed successfully.",
            image=None
        )
    except Exception as e:
        traceback.print_exc()
        return Result(
            value=f"Error closing browser: {str(e)}",
            image=None
        )