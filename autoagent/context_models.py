from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class BaseContext(BaseModel):
    """Base context model that all other context models should inherit from."""
    
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional data that doesn't fit into the defined fields")


class FileSurferContext(BaseContext):
    """Context model for the FileSurfer agent."""
    
    file_env: Any = Field(None, description="The file environment object")
    current_directory: str = Field("", description="The current directory being browsed")
    history: List[str] = Field(default_factory=list, description="History of visited files/directories")


class WebSurferContext(BaseContext):
    """Context model for the WebSurfer agent."""
    
    browser: Any = Field(None, description="The browser object")
    current_url: str = Field("", description="The current URL being browsed")
    history: List[str] = Field(default_factory=list, description="History of visited URLs")
    cookies: Dict[str, str] = Field(default_factory=dict, description="Browser cookies")


class CodingAgentContext(BaseContext):
    """Context model for the Coding agent."""
    
    workspace_path: str = Field("", description="Path to the workspace")
    language: str = Field("", description="Programming language being used")
    files_modified: List[str] = Field(default_factory=list, description="List of files that have been modified")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Project dependencies")


class AssistantContext(BaseContext):
    """Context model for the OpenAI Assistant agent."""
    
    assistant_id: str = Field("", description="ID of the OpenAI Assistant")
    thread_id: Optional[str] = Field(None, description="ID of the thread, if one exists")
    user_id: Optional[str] = Field(None, description="ID of the user")
    
    
def convert_dict_to_context(context_dict: Dict[str, Any], context_type: type) -> BaseContext:
    """
    Convert a dictionary to a context model.
    
    Args:
        context_dict: Dictionary of context variables
        context_type: Type of context model to convert to
        
    Returns:
        An instance of the specified context model
    """
    # Extract known fields for the context type
    known_fields = {}
    additional_data = {}
    
    for key, value in context_dict.items():
        if key in context_type.__annotations__:
            known_fields[key] = value
        else:
            additional_data[key] = value
    
    # Create the context model
    context = context_type(**known_fields)
    context.additional_data = additional_data
    
    return context


def convert_context_to_dict(context: BaseContext) -> Dict[str, Any]:
    """
    Convert a context model to a dictionary.
    
    Args:
        context: Context model to convert
        
    Returns:
        Dictionary of context variables
    """
    context_dict = context.dict(exclude={"additional_data"})
    context_dict.update(context.additional_data)
    
    return context_dict 