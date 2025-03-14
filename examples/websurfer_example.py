"""
Example script demonstrating how to use the improved WebSurfer.
"""

import os
import sys
import logging
import traceback
import time
import signal
from dotenv import load_dotenv

# Add parent directory to path so we can import from autoagent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoagent.environment.web_surfer import WebSurfer
from autoagent.environment.api_handlers import register_api_handlers

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebSurferExample")

# Global variables for handling timeouts and interrupts
web_surfer = None
MAX_COMPLEX_PAGE_WAIT = 60  # Maximum seconds to wait for complex pages

def signal_handler(sig, frame):
    """Handle keyboard interrupts gracefully."""
    logger.info("Keyboard interrupt received. Cleaning up...")
    if web_surfer:
        try:
            web_surfer.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def execute_with_timeout(func, timeout, *args, **kwargs):
    """Execute a function with a timeout."""
    start_time = time.time()
    result = None
    
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        return {"error": True, "last_action_error": str(e)}
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        logger.warning(f"Operation took longer than expected: {elapsed:.2f} seconds")
    
    return result

def main():
    """
    Main function demonstrating WebSurfer usage.
    """
    global web_surfer
    
    # Get local root and workplace name from environment or use defaults
    local_root = os.getenv("LOCAL_ROOT", os.path.expanduser("~"))
    workplace_name = os.getenv("WORKPLACE_NAME", "websurfer_example")
    
    # Create WebSurfer instance
    logger.info(f"Creating WebSurfer instance with local_root={local_root}, workplace_name={workplace_name}")
    try:
        web_surfer = WebSurfer(
            browsergym_eval_env=None,
            local_root=local_root,
            workplace_name=workplace_name
        )
    except Exception as e:
        logger.error(f"Error creating WebSurfer instance: {e}")
        traceback.print_exc()
        return
    
    # Register API handlers if credentials are available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if google_api_key and google_cse_id:
        logger.info("Registering API handlers...")
        register_api_handlers(web_surfer, google_api_key, google_cse_id)
    else:
        logger.info("No Google API credentials found, API fallback for Google Search will not be available")
        register_api_handlers(web_surfer)  # Still register Wikipedia handler
    
    try:
        # Example 1: Navigate to a URL
        logger.info("Example 1: Navigating to example.com...")
        result = web_surfer.execute_action("visit_url", url="https://example.com")
        
        if result.get("error", False):
            logger.error(f"Error: {result.get('last_action_error', 'Unknown error')}")
        else:
            logger.info("Successfully navigated to example.com")
        
        # Example 2: Perform a web search
        logger.info("Example 2: Performing a web search...")
        result = web_surfer.execute_action("web_search", query="python browser automation", search_engine="google")
        
        if result.get("error", False):
            logger.error(f"Error: {result.get('last_action_error', 'Unknown error')}")
        else:
            logger.info("Successfully performed web search")
        
        # Example 3: Get page content
        logger.info("Example 3: Getting page content...")
        result = web_surfer.execute_action("get_page_content")
        
        if result.get("error", False):
            logger.error(f"Error: {result.get('last_action_error', 'Unknown error')}")
        else:
            logger.info(f"Successfully got page content, URL: {result.get('url', 'Unknown URL')}")
            # Print first 100 characters of content
            content = result.get("content", "")
            logger.info(f"Content preview: {content[:100]}...")
        
        # Example 4: Visit Wikipedia using API handler
        logger.info("Example 4: Visiting Wikipedia (should use API handler)...")
        result = web_surfer.execute_action("visit_url", url="https://en.wikipedia.org/wiki/Python_(programming_language)")
        
        if result.get("error", False):
            logger.error(f"Error: {result.get('last_action_error', 'Unknown error')}")
        else:
            logger.info("Successfully visited Wikipedia")
            # Print first 100 characters of content
            content = result.get("content", "")
            logger.info(f"Content preview: {content[:100]}...")
        
        # Example 5: Navigate to a simpler page than W3Schools for click example
        # W3Schools seems to be too complex and causes timeout issues
        logger.info("Example 5: Testing click functionality on a simpler page...")
        logger.info("Navigating to a simple page with clickable elements...")
        
        # Create a simple HTML page in memory to avoid external page load issues
        simple_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Test Page</title>
        </head>
        <body>
            <h1>Test Page for WebSurfer</h1>
            <p>This is a simple test page with clickable elements.</p>
            <a href="#section1" id="link1">Link 1</a>
            <button id="button1">Button 1</button>
            <div id="section1" style="display:none">
                <p>This section appears after clicking Link 1</p>
            </div>
            <script>
                document.getElementById('link1').addEventListener('click', function() {
                    document.getElementById('section1').style.display = 'block';
                });
                document.getElementById('button1').addEventListener('click', function() {
                    alert('Button clicked!');
                });
            </script>
        </body>
        </html>
        """
        
        # Use data URL to load the HTML directly in the browser
        data_url = f"data:text/html;charset=utf-8,{simple_html}"
        result = web_surfer.execute_action("visit_url", url=data_url)
        
        if result.get("error", False):
            logger.error(f"Error: {result.get('last_action_error', 'Unknown error')}")
        else:
            logger.info("Successfully loaded simple test page")
            
            # Analyze the page to find clickable elements
            logger.info("Analyzing page to find clickable elements...")
            analysis_result = web_surfer.analyze_page()
            
            if analysis_result.get("error", False):
                logger.error(f"Error analyzing page: {analysis_result.get('message', 'Unknown error')}")
            else:
                # Try to find a link to click
                links = analysis_result.get("links", [])
                if links:
                    # Find the link with id="link1"
                    target_link = None
                    for link in links:
                        if link.get("id") == "link1":
                            target_link = link
                            break
                    
                    if target_link:
                        link_bid = target_link.get("bid")
                        link_text = target_link.get("text", "Unknown")
                        logger.info(f"Attempting to click link: '{link_text}' with bid: {link_bid}")
                        
                        # Click the link
                        click_result = web_surfer.execute_action("click", bid=link_bid)
                        
                        if click_result.get("error", False):
                            logger.error(f"Error clicking link: {click_result.get('last_action_error', 'Unknown error')}")
                        else:
                            logger.info("Successfully clicked link")
                            
                            # Get the new page content to verify the click worked
                            content_result = web_surfer.execute_action("get_page_content")
                            if not content_result.get("error", False):
                                content = content_result.get("content", "")
                                logger.info(f"Page content after click: {content[:200]}...")
                                if "This section appears after clicking Link 1" in content:
                                    logger.info("Click was successful! Hidden content is now visible.")
                    else:
                        logger.warning("Link with id='link1' not found")
                else:
                    logger.warning("No links found on the page")

        # Optionally try the complex W3Schools page if specifically requested
        if os.getenv("TRY_COMPLEX_PAGE", "false").lower() == "true":
            try_complex_page_example(web_surfer)
            
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during examples: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        logger.info("Cleaning up...")
        if web_surfer:
            web_surfer.close()
        
        logger.info("Example complete!")

def try_complex_page_example(web_surfer):
    """
    Try loading a complex page (W3Schools) which may cause timeout issues.
    This is separated from the main examples to avoid blocking the entire script.
    """
    logger.info("\n--- Optional Complex Page Example ---")
    logger.info("Navigating to W3Schools examples page (this may take some time)...")
    
    try:
        # Use a shorter timeout for the initial request
        result = execute_with_timeout(
            web_surfer.execute_action,
            MAX_COMPLEX_PAGE_WAIT,
            "visit_url", 
            url="https://www.w3schools.com/html/html_examples.asp",
            timeout=MAX_COMPLEX_PAGE_WAIT
        )
        
        if result and not result.get("error", False):
            logger.info("Successfully navigated to W3Schools examples page")
            
            # Try to analyze and click something
            analysis_result = web_surfer.analyze_page()
            
            if not analysis_result.get("error", False):
                links = analysis_result.get("links", [])
                if links:
                    # Find a suitable internal link
                    for link in links:
                        if "w3schools.com" in link.get("href", ""):
                            bid = link.get("bid")
                            text = link.get("text", "Unknown")
                            logger.info(f"Clicking link: {text}")
                            
                            click_result = web_surfer.execute_action("click", bid=bid)
                            if not click_result.get("error", False):
                                logger.info(f"Successfully clicked link to: {click_result.get('url', 'Unknown URL')}")
                            else:
                                logger.error(f"Failed to click link: {click_result.get('last_action_error', 'Unknown error')}")
                            
                            # One click is enough for demonstration
                            break
        else:
            logger.warning("Failed to load W3Schools example page within the timeout period")
            logger.warning("This is expected behavior for complex pages")
            
    except Exception as e:
        logger.error(f"Error during W3Schools example: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()