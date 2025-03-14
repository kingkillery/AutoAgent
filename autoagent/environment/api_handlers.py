"""
API Handlers for WebSurfer's hybrid approach.
These handlers provide API-based alternatives to browser automation for specific sites.
"""

import requests
import json
import logging
import time
from typing import Dict, Any, Optional, List
from urllib.parse import quote_plus

logger = logging.getLogger("ApiHandlers")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class GoogleSearchApiHandler:
    """Handler for Google Search using their Custom Search API."""
    
    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize with API credentials.
        
        Args:
            api_key: Google API key
            search_engine_id: Google Custom Search Engine ID
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def __call__(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Handle a visit_url action for Google Search.
        
        Args:
            url: The URL to visit (should be a Google search URL)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with search results in a format similar to browser observation
        """
        logger.info(f"Handling Google Search API request for URL: {url}")
        
        # Extract query from URL
        try:
            # Usually in the format https://www.google.com/search?q=query
            if "q=" in url:
                query = url.split("q=")[1].split("&")[0]
                query = quote_plus(query)
            else:
                # If we can't parse the URL, return an error
                return {
                    "error": True,
                    "last_action_error": f"Could not extract query from URL: {url}"
                }
        except Exception as e:
            logger.error(f"Error extracting query from URL {url}: {e}")
            return {
                "error": True,
                "last_action_error": f"Error extracting query: {str(e)}"
            }
        
        # Make API request
        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            
            # Format results to match browser observation
            formatted_results = self._format_results(search_results, url)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error making Google Search API request: {e}")
            return {
                "error": True,
                "last_action_error": f"API error: {str(e)}"
            }
    
    def _format_results(self, search_results: Dict[str, Any], original_url: str) -> Dict[str, Any]:
        """Format API results to match browser observation."""
        items = search_results.get("items", [])
        
        # Format the content as a readable text
        content = f"Search results for: {search_results.get('queries', {}).get('request', [{}])[0].get('searchTerms', '')}\n\n"
        
        for i, item in enumerate(items, 1):
            content += f"{i}. {item.get('title', 'No title')}\n"
            content += f"   URL: {item.get('link', 'No link')}\n"
            content += f"   {item.get('snippet', 'No description')}\n\n"
        
        return {
            "content": content,
            "url": original_url,
            "error": False,
            "screenshot": None  # No screenshot available from API
        }


class WikipediaApiHandler:
    """Handler for Wikipedia using their API."""
    
    def __init__(self):
        """Initialize the Wikipedia API handler."""
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    def __call__(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Handle a visit_url action for Wikipedia.
        
        Args:
            url: The URL to visit (should be a Wikipedia URL)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with Wikipedia content in a format similar to browser observation
        """
        logger.info(f"Handling Wikipedia API request for URL: {url}")
        
        # Extract title from URL
        try:
            # Usually in the format https://en.wikipedia.org/wiki/Title
            if "/wiki/" in url:
                title = url.split("/wiki/")[1]
                title = title.split("#")[0]  # Remove any anchor
                title = title.replace("_", " ")  # Replace underscores with spaces
            else:
                # If we can't parse the URL, return an error
                return {
                    "error": True,
                    "last_action_error": f"Could not extract title from URL: {url}"
                }
        except Exception as e:
            logger.error(f"Error extracting title from URL {url}: {e}")
            return {
                "error": True,
                "last_action_error": f"Error extracting title: {str(e)}"
            }
        
        # Make API request
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts|info",
                "inprop": "url",
                "explaintext": 1,
                "exsectionformat": "plain"
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format results to match browser observation
            formatted_results = self._format_results(data, url, title)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error making Wikipedia API request: {e}")
            return {
                "error": True,
                "last_action_error": f"API error: {str(e)}"
            }
    
    def _format_results(self, data: Dict[str, Any], original_url: str, title: str) -> Dict[str, Any]:
        """Format API results to match browser observation."""
        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            return {
                "error": True,
                "last_action_error": "No pages found in API response"
            }
        
        # Get the first (and usually only) page
        page_id = next(iter(pages))
        page = pages[page_id]
        
        # Check for missing page
        if "missing" in page:
            return {
                "error": True,
                "last_action_error": f"Wikipedia page not found: {title}"
            }
        
        # Format the content
        content = f"# {page.get('title', title)}\n\n"
        content += page.get("extract", "No content available")
        
        return {
            "content": content,
            "url": original_url,
            "error": False,
            "screenshot": None  # No screenshot available from API
        }


def register_api_handlers(web_surfer, api_key: str = None, search_engine_id: str = None):
    """
    Register API handlers with the WebSurfer instance.
    
    Args:
        web_surfer: WebSurfer instance
        api_key: Google API key (optional)
        search_engine_id: Google Custom Search Engine ID (optional)
    """
    # Register Wikipedia API handler
    web_surfer.register_api_handler("en.wikipedia.org", WikipediaApiHandler())
    
    # Register Google Search API handler if credentials are provided
    if api_key and search_engine_id:
        web_surfer.register_api_handler(
            "www.google.com", 
            GoogleSearchApiHandler(api_key, search_engine_id)
        )
        
    logger.info("API handlers registered successfully") 