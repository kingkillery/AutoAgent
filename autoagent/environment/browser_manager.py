"""
Browser Manager for maintaining a single instance of BrowserEnv.
This module implements the singleton pattern to ensure only one browser instance exists.
"""

import atexit
import tenacity
from typing import Optional, Dict, Any, List, Tuple
from .browser_env import BrowserEnv, BrowserInitException

class BrowserManager:
    """
    Singleton manager for browser instances.
    Ensures only one browser instance exists at any time and handles cleanup properly.
    """
    _instance: Optional[BrowserEnv] = None
    _init_params: Dict[str, Any] = {}
    
    @classmethod
    def get_instance(cls, browsergym_eval_env: str = None, 
                    local_root: str = None, 
                    workplace_name: str = None,
                    force_new: bool = False) -> BrowserEnv:
        """
        Get a singleton instance of BrowserEnv.
        
        Args:
            browsergym_eval_env: Evaluation environment name
            local_root: Local root directory
            workplace_name: Workplace name
            force_new: Force creation of a new instance even if one exists
        
        Returns:
            An instance of BrowserEnv
        """
        if force_new and cls._instance is not None:
            cls.close_instance()
            
        if cls._instance is None:
            cls._init_params = {
                "browsergym_eval_env": browsergym_eval_env,
                "local_root": local_root,
                "workplace_name": workplace_name
            }
            
            cls._instance = BrowserEnv(
                browsergym_eval_env=browsergym_eval_env,
                local_root=local_root,
                workplace_name=workplace_name
            )
            # Register cleanup on program exit
            atexit.register(cls.close_instance)
            
        return cls._instance
    
    @classmethod
    def close_instance(cls) -> None:
        """Close the browser instance and clean up resources."""
        if cls._instance:
            cls._instance.close()
            cls._instance = None
    
    @classmethod
    def restart_instance(cls) -> BrowserEnv:
        """Restart the browser instance with the same parameters."""
        cls.close_instance()
        return cls.get_instance(**cls._init_params)
    
    @classmethod
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    def safe_step(cls, action_str: str, timeout: float = 30) -> dict:
        """
        Execute a browser action with retry logic for reliability.
        
        Args:
            action_str: The action to execute
            timeout: Timeout in seconds
            
        Returns:
            Browser observation dictionary
        """
        if cls._instance is None:
            raise ValueError("Browser instance not initialized")
            
        try:
            return cls._instance.step(action_str, timeout)
        except Exception as e:
            # If a critical error occurs that suggests browser might be in a bad state
            if "Browser environment took too long to respond" in str(e):
                # Try restarting the browser instance
                cls.restart_instance()
                # And retry the action
                return cls._instance.step(action_str, timeout)
            # Otherwise, let the retry decorator handle it
            raise

    @classmethod
    def execute_with_fallback(cls, primary_action: str, 
                             fallback_actions: List[str], 
                             success_check=None,
                             timeout: float = 30) -> Tuple[dict, bool]:
        """
        Execute an action with fallbacks if it fails.
        
        Args:
            primary_action: The main action to try
            fallback_actions: List of fallback actions to try if the main one fails
            success_check: Function that checks if the action was successful
            timeout: Timeout for each action
            
        Returns:
            Tuple of (observation, success_flag)
        """
        # Try the primary action first
        try:
            obs = cls.safe_step(primary_action, timeout)
            if success_check is None or success_check(obs):
                return obs, True
        except Exception:
            pass  # Continue to fallbacks
            
        # Try fallback actions
        for fallback in fallback_actions:
            try:
                obs = cls.safe_step(fallback, timeout)
                if success_check is None or success_check(obs):
                    return obs, True
            except Exception:
                continue
                
        # If we got here, all actions failed
        return {
            "error": True,
            "last_action_error": "All actions failed, including fallbacks"
        }, False 