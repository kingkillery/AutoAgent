# WebSurfer - Improved Browser Automation

This directory contains examples of how to use the improved WebSurfer implementation, which addresses several key challenges in browser automation:

1. **Better Browser Instance Management**: Uses a singleton pattern to ensure only one browser instance exists.
2. **Structured Action System**: Implements a controller pattern for registering and executing browser actions.
3. **Resilient Navigation with Retries**: Adds proper retry mechanisms for browser actions.
4. **Improved Error Handling**: Adds explicit error recovery for common failure scenarios.
5. **Hybrid Approach with APIs**: Allows for API fallbacks when browser automation is not reliable.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required packages: `requests`, `tenacity`, `pydantic`
- For API fallbacks:
  - Google Custom Search API key (optional)
  - Google Custom Search Engine ID (optional)

### Installation

No additional installation is required if you're already using the AutoAgent framework. All the new WebSurfer functionality is integrated into the existing codebase.

### Environment Variables

Create a `.env` file with the following variables:

```
# For Google Search API fallback (optional)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_custom_search_engine_id

# WebSurfer workplace directory
LOCAL_ROOT=/path/to/local/root
WORKPLACE_NAME=websurfer_example
```

## Example Usage

### Basic Example

Run the example script:

```bash
python examples/websurfer_example.py
```

### Code Example

```python
from autoagent.environment.web_surfer import WebSurfer
from autoagent.environment.api_handlers import register_api_handlers

# Create WebSurfer instance
web_surfer = WebSurfer(
    browsergym_eval_env=None,
    local_root="/path/to/local/root",
    workplace_name="workplace_name"
)

# Register API handlers (optional)
register_api_handlers(web_surfer, google_api_key, google_cse_id)

# Navigate to a URL
result = web_surfer.execute_action("visit_url", url="https://example.com")

# Perform a web search
result = web_surfer.execute_action("web_search", query="python browser automation")

# Get page content
result = web_surfer.execute_action("get_page_content")

# Clean up
web_surfer.close()
```

### Using Tool Wrappers

```python
from autoagent.tools.web_surfer_tools import wsurfer_visit_url, wsurfer_click

# Context variables should contain initialization parameters
context_variables = {
    "local_root": "/path/to/local/root",
    "workplace_name": "workplace_name"
}

# Navigate to a URL
result = wsurfer_visit_url(context_variables, url="https://example.com")

# Click an element
result = wsurfer_click(context_variables, bid="element_id")
```

## Available Actions

The WebSurfer supports the following core actions:

- `visit_url`: Navigate to a URL
- `click`: Click an element on the page
- `input_text`: Input text into a form field
- `web_search`: Perform a web search
- `get_page_content`: Get the content of the current page
- `page_down`: Scroll down on the page
- `page_up`: Scroll up on the page
- `history_back`: Go back in browser history

## Extending WebSurfer

### Adding Custom Actions

You can add custom actions to the WebSurfer controller:

```python
from autoagent.environment.web_surfer import WebSurfer

web_surfer = WebSurfer(...)

# Access the controller directly
@web_surfer.controller.action(name="custom_action")
def custom_action(browser, param1: str, param2: int = 0):
    """Custom action documentation"""
    # Implementation
    return result
```

### Adding API Handlers

You can add custom API handlers to provide fallbacks for specific domains:

```python
class CustomApiHandler:
    def __call__(self, url: str, **kwargs):
        # Implementation
        return {
            "content": "API content",
            "url": url,
            "error": False
        }

# Register the API handler
web_surfer.register_api_handler("example.com", CustomApiHandler())
```

## Architecture

The WebSurfer implementation is built on several key components:

1. **BrowserManager**: Singleton manager for browser instances
2. **BrowserController**: Controller for registering and executing browser actions
3. **WebSurfer**: Main class that combines BrowserManager and BrowserController
4. **API Handlers**: Optional handlers for API-based fallbacks

This architecture provides a more reliable and maintainable approach to browser automation, with better error handling and recovery mechanisms. 