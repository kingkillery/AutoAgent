"""
Browser Controller for structured management of browser actions.
Inspired by the Browser-Use framework's controller pattern.
"""

import inspect
import functools
from typing import Dict, Callable, List, Any, Optional, Type, Union
from pydantic import BaseModel, create_model
from .browser_manager import BrowserManager
from .browser_env import BrowserEnv

class ActionMetadata:
    """Metadata for a browser action"""
    def __init__(self, 
                func: Callable, 
                name: str,
                description: str,
                requires_browser: bool = True,
                parameters: Optional[Type[BaseModel]] = None):
        self.func = func
        self.name = name
        self.description = description
        self.requires_browser = requires_browser
        self.parameters = parameters
        self.original_sig = inspect.signature(func)
    
    def __str__(self):
        return f"Action({self.name})"
    
    def __repr__(self):
        return self.__str__()

class BrowserController:
    """
    Controller for registering and executing browser actions.
    Provides a structured way to define and use browser actions.
    """
    
    def __init__(self):
        self.actions: Dict[str, ActionMetadata] = {}
    
    def action(self, name: str = None, 
              description: str = None,
              requires_browser: bool = True):
        """
        Decorator for registering browser actions.
        
        Args:
            name: Name of the action (defaults to function name)
            description: Description of the action (defaults to function docstring)
            requires_browser: Whether the action requires a browser instance
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            nonlocal name, description
            
            # Default name to function name if not provided
            if name is None:
                action_name = func.__name__
            else:
                action_name = name
                
            # Default description to function docstring if not provided
            if description is None:
                action_description = func.__doc__ or ""
            else:
                action_description = description
            
            # Extract parameters from function signature
            sig = inspect.signature(func)
            param_fields = {}
            
            for param_name, param in sig.parameters.items():
                # Skip 'self' and 'browser' parameters
                if param_name in ['self', 'browser']:
                    continue
                    
                # Get parameter type annotation
                param_type = param.annotation
                if param_type is inspect.Parameter.empty:
                    param_type = str
                    
                # Get default value if any
                has_default = param.default is not inspect.Parameter.empty
                default_value = param.default if has_default else ...
                
                # Add to parameter fields
                param_fields[param_name] = (param_type, default_value)
            
            # Create pydantic model for parameters
            if param_fields:
                param_model = create_model(
                    f"{action_name.title()}Params",
                    **param_fields
                )
            else:
                param_model = None
            
            # Create action metadata
            action_meta = ActionMetadata(
                func=func,
                name=action_name,
                description=action_description,
                requires_browser=requires_browser,
                parameters=param_model
            )
            
            # Register action
            self.actions[action_name] = action_meta
            
            # Return original function
            return func
            
        return decorator
    
    def execute(self, action_name: str, 
               params: Dict[str, Any] = None, 
               browser: BrowserEnv = None) -> Any:
        """
        Execute a registered action.
        
        Args:
            action_name: Name of the action to execute
            params: Parameters to pass to the action
            browser: Browser instance to use (if required)
            
        Returns:
            Result of the action
        """
        if params is None:
            params = {}
            
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' not registered")
            
        action = self.actions[action_name]
        
        # Validate parameters using pydantic model if available
        if action.parameters:
            validated_params = action.parameters(**params)
            params = validated_params.dict()
        
        # Check if browser instance is required
        if action.requires_browser:
            if browser is None:
                # Get browser instance from manager if not provided
                browser = BrowserManager.get_instance()
                
            # Add browser to parameters
            return action.func(browser=browser, **params)
        else:
            # Execute without browser
            return action.func(**params)
    
    def get_action_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered actions.
        
        Returns:
            List of action information dictionaries
        """
        action_info = []
        
        for name, meta in self.actions.items():
            # Extract parameter information if available
            if meta.parameters:
                param_schema = meta.parameters.schema()
                parameters = param_schema.get('properties', {})
                required = param_schema.get('required', [])
            else:
                parameters = {}
                required = []
                
            # Create action info dictionary
            info = {
                'name': name,
                'description': meta.description,
                'parameters': parameters,
                'required_parameters': required
            }
            
            action_info.append(info)
            
        return action_info
    
    def get_action_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for all registered actions.
        
        Returns:
            JSON schema dictionary
        """
        actions = []
        
        for name, meta in self.actions.items():
            # Extract parameter information if available
            if meta.parameters:
                param_schema = meta.parameters.schema()
                parameters = {
                    'type': 'object',
                    'properties': param_schema.get('properties', {}),
                    'required': param_schema.get('required', [])
                }
            else:
                parameters = {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
                
            # Create action schema
            action_schema = {
                'name': name,
                'description': meta.description,
                'parameters': parameters
            }
            
            actions.append(action_schema)
            
        return {
            'type': 'function',
            'functions': actions
        } 