# WebSurfer - Improved Browser Automation for AutoAgent

This is an improved implementation of browser automation for the AutoAgent framework, inspired by the Browser-Use framework and other similar tools. It addresses several key challenges in browser automation:

## Key Improvements

### 1. Better Browser Instance Management

The new implementation uses a singleton pattern for browser instance management, ensuring only one browser instance exists at any time. This helps prevent resource leaks and makes browser lifecycle management more robust.

```python
# BrowserManager singleton pattern
class BrowserManager:
    _instance = None
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = BrowserEnv(*args, **kwargs)
        return cls._instance
```

### 2. Structured Action System

The implementation introduces a controller pattern for registering and executing browser actions, making the code more modular and maintainable.

```python
# BrowserController action registration
controller = BrowserController()

@controller.action(name="visit_url", description="Navigate to a URL")
def visit_url(browser, url: str, timeout: float = 30):
    # Implementation
```

### 3. Resilient Navigation with Retries

The implementation adds proper retry mechanisms for browser actions, making navigation more reliable.

```python
# Retry mechanism for actions
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type(Exception)
)
def _visit_url_with_retry(browser, url, timeout=30):
    # Implementation
```

### 4. Improved Error Handling

The implementation adds explicit error recovery for common failure scenarios, making the browser automation more resilient.

```python
try:
    # Execute action
    return BrowserManager.safe_step(action_str)
except Exception as e:
    # Handle error
    self.last_error = str(e)
    self.logger.error(f"Error executing action: {e}")
    return {
        "error": True,
        "last_action_error": f"Failed to execute action: {str(e)}"
    }
```

### 5. Hybrid Approach with APIs

The implementation allows for API fallbacks when browser automation is not reliable, providing a more robust approach to web interaction.

```python
# Check if we have an API handler for this domain
if action_name == "visit_url" and "url" in params:
    domain = self._extract_domain(params["url"])
    if domain in self.api_handlers:
        return self.api_handlers[domain](**params)
```

## Architecture

### Components

The implementation consists of several key components:

1. **BrowserManager**: Singleton manager for browser instances
   - Ensures only one browser instance exists
   - Provides methods for browser lifecycle management
   - Implements retry logic for browser actions

2. **BrowserController**: Controller for registering and executing browser actions
   - Provides a decorator for registering actions
   - Validates parameters using Pydantic models
   - Executes actions with proper error handling

3. **WebSurfer**: Main class that combines BrowserManager and BrowserController
   - Provides a high-level interface for browser automation
   - Implements core browser actions
   - Supports API fallbacks for specific domains

4. **API Handlers**: Optional handlers for API-based fallbacks
   - Provide alternatives to browser automation for specific sites
   - Match the interface of browser actions for seamless integration

## Usage

### Basic Usage

```python
from autoagent.environment.web_surfer import WebSurfer

# Create WebSurfer instance
web_surfer = WebSurfer(
    browsergym_eval_env=None,
    local_root="/path/to/local/root",
    workplace_name="workplace_name"
)

# Execute an action
result = web_surfer.execute_action("visit_url", url="https://example.com")

# Clean up
web_surfer.close()
```

### Using Tool Wrappers

```python
from autoagent.tools.web_surfer_tools import wsurfer_visit_url

# Context variables should contain initialization parameters
context_variables = {
    "local_root": "/path/to/local/root",
    "workplace_name": "workplace_name"
}

# Use the tool wrapper
result = wsurfer_visit_url(context_variables, url="https://example.com")
```

### Adding API Handlers

```python
from autoagent.environment.api_handlers import register_api_handlers

# Register API handlers
register_api_handlers(web_surfer, google_api_key, google_cse_id)
```

## Integration with AutoAgent

The WebSurfer implementation is fully integrated with the AutoAgent framework. It can be used as a drop-in replacement for the existing browser automation functionality.

For example usage, see the `examples/websurfer_example.py` script and the `examples/README.md` file.

## Benefits

This implementation provides several benefits over the existing browser automation approach:

1. **More Reliable**: With better error handling and retry mechanisms
2. **More Maintainable**: With a modular and structured approach
3. **More Efficient**: With singleton browser management and API fallbacks
4. **More Extensible**: With a controller pattern for registering actions

## Future Improvements

Potential future improvements include:

1. **Cookie Management**: Better handling of cookies for persistent sessions
2. **More API Handlers**: Additional API handlers for popular websites
3. **Browser Context Reuse**: Support for reusing browser contexts for better performance
4. **Parallel Actions**: Support for executing multiple actions in parallel
5. **Action History**: Track and analyze action history for better debugging 