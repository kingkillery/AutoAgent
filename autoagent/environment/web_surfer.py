"""
WebSurfer - Improved browser automation with better reliability and error handling.
Uses BrowserManager for singleton browser instance management and BrowserController for structured actions.
"""

import os
import time
import logging
import tenacity
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Literal
from urllib.parse import quote_plus, urlparse
from PIL import Image
import base64
import io
import json
import traceback
from enum import Enum

from .browser_manager import BrowserManager
from .browser_controller import BrowserController
from .browser_env import BrowserEnv
from autoagent.registry import register_tool
from autoagent.types import Result


class PageReadinessState(Enum):
    """Enum to track the readiness state of a page."""
    NOT_STARTED = 0
    LOADING = 1
    INTERACTIVE = 2  # DOM is ready but resources might still be loading
    COMPLETE = 3     # Everything is loaded


class WebSurfer:
    """
    WebSurfer provides browser automation with improved reliability and error handling.
    """
    
    def __init__(self, 
                browsergym_eval_env: str = None,
                local_root: str = None, 
                workplace_name: str = None):
        """
        Initialize WebSurfer with a browser instance.
        
        Args:
            browsergym_eval_env: Evaluation environment name
            local_root: Local root directory
            workplace_name: Workplace name
        """
        self.controller = BrowserController()
        
        # Set default values if parameters are None
        self._browser_init_params = {
            "browsergym_eval_env": browsergym_eval_env,
            "local_root": local_root if local_root is not None else os.getcwd(),
            "workplace_name": workplace_name if workplace_name is not None else "websurfer_workplace"
        }
        
        self._browser = None  # Lazy load when actually needed
        self.logger = logging.getLogger("WebSurfer")
        self._setup_logging()
        self._register_core_actions()
        self.last_error = None
        self.api_handlers = {}  # For hybrid API approach
        self.last_observation = None  # Store the last observation for reference
        self.page_readiness = PageReadinessState.NOT_STARTED
        self.element_wait_timeout = 10  # Default timeout in seconds for element waits
        self.page_load_timeout = 60  # Default timeout in seconds for page loads
    
    @property
    def browser(self):
        """Lazy load the browser only when it's first accessed"""
        if self._browser is None:
            self.logger.info("Initializing browser instance on first use")
            self._browser = BrowserManager.get_instance(
                browsergym_eval_env=self._browser_init_params["browsergym_eval_env"],
                local_root=self._browser_init_params["local_root"],
                workplace_name=self._browser_init_params["workplace_name"]
            )
        return self._browser
    
    def _setup_logging(self):
        """Set up logging for WebSurfer."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _register_core_actions(self):
        """Register core browser actions with the controller."""
        # Create decorators for the controller
        action = self.controller.action
        
        # Core navigation actions
        @action(name="visit_url", description="Navigate to a URL")
        def visit_url(browser: BrowserEnv, url: str, timeout: float = 30, wait_until: str = "domcontentloaded"):
            """
            Navigate to a specified URL with proper error handling and validation.
            
            Args:
                browser: Browser environment
                url: URL to navigate to
                timeout: Maximum time to wait for page load
                wait_until: Page load state to wait for ('domcontentloaded', 'load', or 'networkidle')
            """
            self.logger.info(f"Navigating to URL: {url} (wait_until: {wait_until})")
            
            # Validate and normalize URL
            normalized_url = self._normalize_url(url)
            
            # Track page readiness state
            self.page_readiness = PageReadinessState.LOADING
            
            # Use exponential backoff retry for visit_page with specific wait_until option
            result = self._visit_url_with_retry(browser, normalized_url, timeout, wait_until)
            
            # Update page readiness state based on wait_until parameter
            if wait_until == "domcontentloaded":
                self.page_readiness = PageReadinessState.INTERACTIVE
            elif wait_until == "load" or wait_until == "networkidle":
                self.page_readiness = PageReadinessState.COMPLETE
                
            self.last_observation = result  # Store last observation
            return result
        
        @action(name="click", description="Click an element on the page")
        def click(browser: BrowserEnv, bid: str, button: str = "left", timeout: float = None, wait_for_navigation: bool = True):
            """
            Click an element with the specified bid, waiting for it to be available.
            
            Args:
                browser: Browser environment
                bid: Element bid (DOM identifier)
                button: Mouse button to use ('left', 'right', 'middle')
                timeout: Maximum time to wait for element
                wait_for_navigation: Whether to wait for navigation after click
            """
            timeout = timeout or self.element_wait_timeout
            self.logger.info(f"Clicking element with bid: {bid} (timeout: {timeout}s, wait_for_nav: {wait_for_navigation})")
            
            # First wait for the element to be available and interactable
            element_available = self._wait_for_element(bid, timeout)
            
            if not element_available:
                self.last_error = f"Element with bid '{bid}' not found or not interactable within {timeout} seconds"
                self.logger.error(self.last_error)
                return {
                    "error": True,
                    "last_action_error": self.last_error
                }
            
            # Now perform the click with appropriate wait options
            try:
                if wait_for_navigation:
                    # Use a more complex click that waits for potential navigation
                    action_script = f"""
                    async () => {{
                        const element = document.querySelector('[data-bid="{bid}"]');
                        if (!element) throw new Error("Element not found");
                        
                        // Create a navigation promise before clicking
                        const navigationPromise = page.waitForNavigation({{ timeout: {timeout * 1000}, waitUntil: 'domcontentloaded' }}).catch(e => null);
                        
                        // Perform the click
                        await element.click({{ button: '{button}' }});
                        
                        // Wait for potential navigation or timeout
                        const navigationResult = await Promise.race([
                            navigationPromise,
                            new Promise(resolve => setTimeout(() => resolve(null), 2000))
                        ]);
                        
                        return {{ clicked: true, navigated: navigationResult !== null }};
                    }}
                    """
                    result = BrowserManager.safe_step(f"page.evaluate({repr(action_script)})")
                    
                    # If navigation occurred, update page readiness
                    if result.get("value", {}).get("navigated", False):
                        self.page_readiness = PageReadinessState.INTERACTIVE
                else:
                    # Simple click without waiting for navigation
                    action_str = f"click_id('{bid}', '{button}')"
                    result = BrowserManager.safe_step(action_str)
                
                self.last_observation = result  # Store last observation
                return result
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error clicking element: {e}")
                return {
                    "error": True,
                    "last_action_error": f"Failed to click element: {str(e)}"
                }
        
        @action(name="input_text", description="Input text into a form field")
        def input_text(browser: BrowserEnv, bid: str, text: str, timeout: float = None):
            """
            Input text into a form field with the specified bid, waiting for it to be available.
            
            Args:
                browser: Browser environment
                bid: Element bid (DOM identifier)
                text: Text to input
                timeout: Maximum time to wait for element
            """
            timeout = timeout or self.element_wait_timeout
            self.logger.info(f"Inputting text into element with bid: {bid} (timeout: {timeout}s)")
            
            # First wait for the element to be available and interactable
            element_available = self._wait_for_element(bid, timeout)
            
            if not element_available:
                self.last_error = f"Element with bid '{bid}' not found or not interactable within {timeout} seconds"
                self.logger.error(self.last_error)
                return {
                    "error": True,
                    "last_action_error": self.last_error
                }
            
            # Now perform the input
            try:
                action_str = f"fill('{bid}', '{text}')"
                result = BrowserManager.safe_step(action_str)
                self.last_observation = result  # Store last observation
                return result
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error inputting text: {e}")
                return {
                    "error": True,
                    "last_action_error": f"Failed to input text: {str(e)}"
                }
        
        @action(name="web_search", description="Perform a web search")
        def web_search(browser: BrowserEnv, query: str, search_engine: str = "google"):
            """Perform a web search using the specified search engine."""
            self.logger.info(f"Performing web search for: {query}")
            encoded_query = quote_plus(query)
            
            search_urls = {
                "google": f"https://www.google.com/search?q={encoded_query}&hl=en&gl=US",
                "bing": f"https://www.bing.com/search?q={encoded_query}",
                "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}"
            }
            
            search_url = search_urls.get(search_engine.lower(), search_urls["google"])
            
            # Use domcontentloaded to speed up interaction
            result = self._visit_url_with_retry(browser, search_url, wait_until="domcontentloaded")
            self.last_observation = result  # Store last observation
            return result
        
        @action(name="get_page_content", description="Get the content of the current page")
        def get_page_content(browser: BrowserEnv):
            """Get the text content and metadata of the current page."""
            self.logger.info("Getting page content")
            action_str = "_get_page_markdown()"
            
            try:
                obs = BrowserManager.safe_step(action_str)
                self.last_observation = obs  # Store last observation
                return {
                    "content": obs.get("text_content", ""),
                    "url": obs.get("url", ""),
                    "error": False
                }
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error getting page content: {e}")
                return {
                    "content": "",
                    "url": "",
                    "error": True,
                    "last_action_error": f"Failed to get page content: {str(e)}"
                }
        
        @action(name="page_down", description="Scroll down on the page")
        def page_down(browser: BrowserEnv):
            """Scroll down on the current page."""
            self.logger.info("Scrolling down")
            action_str = "press('PageDown')"
            
            try:
                result = BrowserManager.safe_step(action_str)
                self.last_observation = result  # Store last observation
                return result
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error scrolling down: {e}")
                return {
                    "error": True,
                    "last_action_error": f"Failed to scroll down: {str(e)}"
                }
        
        @action(name="page_up", description="Scroll up on the page")
        def page_up(browser: BrowserEnv):
            """Scroll up on the current page."""
            self.logger.info("Scrolling up")
            action_str = "press('PageUp')"
            
            try:
                result = BrowserManager.safe_step(action_str)
                self.last_observation = result  # Store last observation
                return result
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error scrolling up: {e}")
                return {
                    "error": True,
                    "last_action_error": f"Failed to scroll up: {str(e)}"
                }
        
        @action(name="history_back", description="Go back in browser history")
        def history_back(browser: BrowserEnv):
            """Go back in browser history."""
            self.logger.info("Going back in history")
            action_str = "go_back()"
            
            try:
                result = BrowserManager.safe_step(action_str)
                self.last_observation = result  # Store last observation
                return result
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error going back in history: {e}")
                return {
                    "error": True,
                    "last_action_error": f"Failed to go back in history: {str(e)}"
                }
        
        # Add new interactive capabilities
        @action(name="find_elements", description="Find interactable elements on the page")
        def find_elements(browser: BrowserEnv, element_type: str = None):
            """Find interactable elements on the page, optionally filtered by type."""
            self.logger.info(f"Finding elements of type: {element_type}")
            
            try:
                # Try to find elements even if page is still loading (interactive state)
                if element_type:
                    element_filter = f"document.querySelectorAll('{element_type}')"
                    filter_script = f"Array.from({element_filter}).map(el => ({{ type: el.tagName, id: el.id, name: el.name, value: el.value, bid: el.getAttribute('data-bid'), text: el.textContent, classes: el.className, disabled: el.disabled, action: el.getAttribute('action'), href: el.getAttribute('href') }}))"
                else:
                    filter_script = "Array.from(document.querySelectorAll('a, button, input, select, textarea, form, [role=\"button\"]')).map(el => ({ type: el.tagName, id: el.id, name: el.name, value: el.value, bid: el.getAttribute('data-bid'), text: el.textContent, classes: el.className, disabled: el.disabled, action: el.getAttribute('action'), href: el.getAttribute('href') }))"
                
                elements_result = BrowserManager.safe_step(f"page.evaluate({repr(filter_script)})")
                
                return {
                    "elements": elements_result.get("value", []),
                    "error": False
                }
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error finding elements: {e}")
                return {
                    "elements": [],
                    "error": True,
                    "last_action_error": f"Failed to find elements: {str(e)}"
                }
        
        @action(name="find_forms", description="Find and analyze forms on the page")
        def find_forms(browser: BrowserEnv):
            """Find forms on the page and analyze their structure."""
            self.logger.info("Finding forms on the page")
            
            form_script = """
            Array.from(document.querySelectorAll('form')).map(form => {
                const inputs = Array.from(form.querySelectorAll('input, select, textarea')).map(input => ({
                    type: input.tagName.toLowerCase() === 'input' ? input.type : input.tagName.toLowerCase(),
                    name: input.name,
                    id: input.id,
                    value: input.value,
                    placeholder: input.placeholder,
                    required: input.required,
                    bid: input.getAttribute('data-bid'),
                    disabled: input.disabled
                }));
                
                const submitButtons = Array.from(form.querySelectorAll('button[type="submit"], input[type="submit"]')).map(btn => ({
                    type: btn.tagName.toLowerCase(),
                    text: btn.textContent || btn.value,
                    bid: btn.getAttribute('data-bid')
                }));
                
                return {
                    id: form.id,
                    action: form.action,
                    method: form.method,
                    bid: form.getAttribute('data-bid'),
                    inputs: inputs,
                    submitButtons: submitButtons
                };
            })
            """
            
            try:
                form_result = BrowserManager.safe_step(f"page.evaluate({repr(form_script)})")
                
                return {
                    "forms": form_result.get("value", []),
                    "error": False
                }
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error finding forms: {e}")
                return {
                    "forms": [],
                    "error": True,
                    "last_action_error": f"Failed to find forms: {str(e)}"
                }
        
        @action(name="extract_links", description="Extract all links from the page")
        def extract_links(browser: BrowserEnv):
            """Extract all links from the current page with their text and URLs."""
            self.logger.info("Extracting links from page")
            
            links_script = """
            Array.from(document.querySelectorAll('a[href]')).map(link => ({
                text: link.textContent.trim(),
                href: link.href,
                bid: link.getAttribute('data-bid'),
                title: link.title,
                target: link.target,
                rel: link.rel
            })).filter(link => link.href && !link.href.startsWith('javascript:'))
            """
            
            try:
                links_result = BrowserManager.safe_step(f"page.evaluate({repr(links_script)})")
                
                return {
                    "links": links_result.get("value", []),
                    "error": False
                }
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error extracting links: {e}")
                return {
                    "links": [],
                    "error": True,
                    "last_action_error": f"Failed to extract links: {str(e)}"
                }
        
        # New method for page readiness status
        @action(name="get_page_readiness", description="Get the current page readiness state")
        def get_page_readiness(browser: BrowserEnv):
            """Get the current page readiness state."""
            self.logger.info("Getting page readiness state")
            
            readiness_script = """
            () => {
                return {
                    readyState: document.readyState,
                    domContentLoaded: document.readyState !== 'loading',
                    fullPageLoaded: document.readyState === 'complete'
                }
            }
            """
            
            try:
                readiness_result = BrowserManager.safe_step(f"page.evaluate({readiness_script})")
                ready_state = readiness_result.get("value", {}).get("readyState", "unknown")
                
                # Update our internal tracking based on DOM state
                if ready_state == "loading":
                    self.page_readiness = PageReadinessState.LOADING
                elif ready_state == "interactive":
                    self.page_readiness = PageReadinessState.INTERACTIVE
                elif ready_state == "complete":
                    self.page_readiness = PageReadinessState.COMPLETE
                
                return {
                    "readiness": {
                        "state": ready_state,
                        "dom_content_loaded": ready_state != "loading",
                        "full_page_loaded": ready_state == "complete"
                    },
                    "error": False
                }
            except Exception as e:
                self.last_error = str(e)
                self.logger.error(f"Error getting page readiness: {e}")
                return {
                    "readiness": {"state": "unknown", "dom_content_loaded": False, "full_page_loaded": False},
                    "error": True,
                    "last_action_error": f"Failed to get page readiness: {str(e)}"
                }
                
        # New method for waiting for an element
        @action(name="wait_for_element", description="Wait for an element to be available")
        def wait_for_element_action(browser: BrowserEnv, bid: str, timeout: float = None):
            """
            Wait for an element with the specified bid to be available.
            
            Args:
                browser: Browser environment
                bid: Element bid (DOM identifier)
                timeout: Maximum time to wait in seconds
            """
            timeout = timeout or self.element_wait_timeout
            self.logger.info(f"Waiting for element with bid: {bid} (timeout: {timeout}s)")
            
            element_available = self._wait_for_element(bid, timeout)
            
            return {
                "available": element_available,
                "error": not element_available,
                "last_action_error": None if element_available else f"Element with bid '{bid}' not found within {timeout} seconds"
            }
        
        # New method for early interaction based on readiness state
        @action(name="set_navigation_behavior", description="Set navigation behavior options")
        def set_navigation_behavior(browser: BrowserEnv, page_load_timeout: int = None, element_wait_timeout: int = None, wait_until: str = None):
            """
            Configure navigation behavior options.
            
            Args:
                browser: Browser environment
                page_load_timeout: Maximum time to wait for page loads (seconds)
                element_wait_timeout: Maximum time to wait for elements (seconds)
                wait_until: Default page load state to wait for ('domcontentloaded', 'load', or 'networkidle')
            """
            changes = []
            
            if page_load_timeout is not None:
                self.page_load_timeout = page_load_timeout
                changes.append(f"page_load_timeout set to {page_load_timeout}s")
            
            if element_wait_timeout is not None:
                self.element_wait_timeout = element_wait_timeout
                changes.append(f"element_wait_timeout set to {element_wait_timeout}s")
            
            if wait_until is not None:
                valid_options = ["domcontentloaded", "load", "networkidle"]
                if wait_until in valid_options:
                    # This would be used as the default for visit_url
                    changes.append(f"default wait_until set to '{wait_until}'")
                else:
                    return {
                        "error": True,
                        "last_action_error": f"Invalid wait_until option. Valid options: {', '.join(valid_options)}"
                    }
            
            return {
                "message": f"Navigation behavior updated: {', '.join(changes)}",
                "error": False
            }
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize the URL to ensure it's properly formatted.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        # If URL contains spaces, treat as search query
        if " " in url and not url.startswith(("http://", "https://", "file://", "about:")):
            return f"https://www.google.com/search?q={quote_plus(url)}&hl=en&gl=US"
        
        # Add https:// if missing
        if not url.startswith(("http://", "https://", "file://", "about:")):
            return f"https://{url}"
        
        return url
    
    def _wait_for_element(self, bid: str, timeout: float) -> bool:
        """
        Wait for an element with the specified bid to be available and interactable.
        
        Args:
            bid: Element bid to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if element is available and interactable, False otherwise
        """
        self.logger.info(f"Waiting for element with bid '{bid}' (max {timeout}s)")
        start_time = time.time()
        
        # Generate a script to check if element exists and is interactable
        check_script = f"""
        () => {{
            const element = document.querySelector('[data-bid="{bid}"]');
            if (!element) return {{ found: false, interactable: false }};
            
            // Check if element is visible and enabled
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            const isVisible = style.display !== 'none' && 
                              style.visibility !== 'hidden' && 
                              rect.width > 0 && 
                              rect.height > 0;
            
            // Check if element is not disabled
            const isEnabled = !element.disabled;
            
            return {{ 
                found: true, 
                interactable: isVisible && isEnabled,
                position: {{ 
                    x: rect.x, 
                    y: rect.y, 
                    width: rect.width, 
                    height: rect.height 
                }}
            }};
        }}
        """
        
        # Poll until element is found or timeout
        while time.time() - start_time < timeout:
            try:
                result = BrowserManager.safe_step(f"page.evaluate({check_script})")
                element_status = result.get("value", {})
                
                if element_status.get("found", False) and element_status.get("interactable", False):
                    self.logger.info(f"Element with bid '{bid}' found and interactable after {time.time() - start_time:.2f}s")
                    return True
                
                # If element is found but not interactable, log details
                if element_status.get("found", False):
                    position = element_status.get("position", {})
                    self.logger.debug(f"Element found but not interactable. Position: x={position.get('x')}, y={position.get('y')}, width={position.get('width')}, height={position.get('height')}")
                
                # Sleep before next check
                time.sleep(0.2)
            except Exception as e:
                self.logger.debug(f"Error checking element: {e}")
                # Continue polling
        
        self.logger.warning(f"Timed out waiting for element with bid '{bid}' after {timeout}s")
        return False
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    def _visit_url_with_retry(self, browser: BrowserEnv, url: str, timeout: float = 30, 
                             wait_until: Literal["domcontentloaded", "load", "networkidle"] = "domcontentloaded") -> Dict[str, Any]:
        """
        Visit a URL with retry logic and configurable wait option.
        
        Args:
            browser: Browser instance
            url: URL to visit
            timeout: Timeout in seconds
            wait_until: When to consider navigation complete
                - 'domcontentloaded': Faster, waits for DOM parsing to complete
                - 'load': Medium, waits for page load event
                - 'networkidle': Slower, waits for network to be idle
            
        Returns:
            Browser observation
        """
        # Map our options to browsergym's actual options
        wait_map = {
            "domcontentloaded": "domcontentloaded",
            "load": "load", 
            "networkidle": "networkidle"
        }
        wait_option = wait_map.get(wait_until, "domcontentloaded")
        
        # Log detailed navigation attempt
        self.logger.info(f"Visiting URL {url} with wait_until={wait_option}, timeout={timeout}s")
        start_time = time.time()
        
        # Construct action string with wait option
        action_str = f"_visit_page('{url}', wait_until='{wait_option}')"
        
        try:
            result = BrowserManager.safe_step(action_str, timeout)
            elapsed = time.time() - start_time
            self.logger.info(f"Navigation to {url} completed in {elapsed:.2f}s")
            
            # Update page readiness state based on wait_until parameter
            if wait_until == "domcontentloaded":
                self.page_readiness = PageReadinessState.INTERACTIVE
            elif wait_until == "load" or wait_until == "networkidle":
                self.page_readiness = PageReadinessState.COMPLETE
                
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            self.last_error = str(e)
            self.logger.error(f"Error visiting URL {url} after {elapsed:.2f}s: {e}")
            
            # Try to detect if this is a timeout error
            if "timeout" in str(e).lower():
                # For timeout errors, provide more specific information
                if elapsed >= timeout:
                    raise Exception(f"Navigation to {url} timed out after {timeout} seconds. Try using wait_until='domcontentloaded' for faster response.")
                
            return {
                "error": True,
                "last_action_error": f"Failed to visit URL: {str(e)}"
            }
    
    def register_api_handler(self, domain: str, handler_func: Callable):
        """
        Register an API handler for a specific domain.
        
        Args:
            domain: Domain name (e.g., 'google.com')
            handler_func: API handler function
        """
        self.api_handlers[domain] = handler_func
    
    def execute_action(self, action_name: str, **params) -> Dict[str, Any]:
        """
        Execute a browser action with the controller.
        
        Args:
            action_name: Name of the action to execute
            **params: Parameters for the action
            
        Returns:
            Result of the action
        """
        try:
            # Check if we have an API handler for this action
            if action_name == "visit_url" and "url" in params:
                domain = self._extract_domain(params["url"])
                if domain in self.api_handlers:
                    self.logger.info(f"Using API handler for domain: {domain}")
                    return self.api_handlers[domain](**params)
            
            # Fall back to browser automation
            result = self.controller.execute(action_name, params, self.browser)
            return result
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Error executing action {action_name}: {e}")
            traceback.print_exc()
            return {
                "error": True,
                "last_action_error": f"Failed to execute {action_name}: {str(e)}"
            }
    
    def analyze_page(self) -> Dict[str, Any]:
        """
        Analyze the current page and extract useful information.
        
        Returns:
            Dictionary with page analysis
        """
        try:
            # Get page content
            content_result = self.execute_action("get_page_content")
            
            # Extract links
            links_result = self.execute_action("extract_links")
            
            # Find forms
            forms_result = self.execute_action("find_forms")
            
            # Find interactable elements
            elements_result = self.execute_action("find_elements")
            
            return {
                "title": self.last_observation.get("title", "") if self.last_observation else "",
                "url": content_result.get("url", ""),
                "content": content_result.get("content", ""),
                "links": links_result.get("links", []),
                "forms": forms_result.get("forms", []),
                "elements": elements_result.get("elements", []),
                "error": False
            }
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Error analyzing page: {e}")
            traceback.print_exc()
            return {
                "error": True,
                "last_action_error": f"Failed to analyze page: {str(e)}"
            }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed_url = urlparse(self._normalize_url(url))
            return parsed_url.netloc
        except:
            return ""
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error that occurred."""
        return self.last_error
    
    def restart_browser(self):
        """Restart the browser instance."""
        if self._browser is not None:
            self._browser = BrowserManager.restart_instance()
        
    def close(self):
        """Close the browser instance."""
        if self._browser is not None:
            BrowserManager.close_instance()
            self._browser = None
