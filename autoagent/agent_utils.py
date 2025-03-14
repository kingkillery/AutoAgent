import re
from typing import List, Dict, Any, Optional

def detect_agent_reference(content: str, tools: List[Dict[str, Any]], debug: bool = False) -> bool:
    """
    Detect if the response content mentions transferring to an agent.
    
    Args:
        content: The content to check for agent references
        tools: List of tool definitions
        debug: Whether to print debug information
        
    Returns:
        True if an agent reference is detected, False otherwise
    """
    # Check if content is None and handle it
    if content is None:
        return False
        
    print(f"DEBUG: Checking for agent reference in: '{content}'") if debug else None
    
    # Create a regex pattern for transfer-related words
    transfer_words = r"(transfer|connect|use|hand|pass|delegate|redirect|forward)"
    
    for tool in tools:
        if not isinstance(tool, dict) or 'function' not in tool:
            continue
            
        func_name = tool['function']['name']
        if not func_name.startswith('transfer_to_'):
            continue
            
        agent_name = func_name.replace('transfer_to_', '').replace('_agent', '')
        
        # Check for explicit transfer mentions using regex
        pattern = rf"{transfer_words}\s+(?:to(?:\s+the)?\s+)?{re.escape(agent_name)}"
        if re.search(pattern, content.lower()):
            print(f"DEBUG: Found transfer word with agent '{agent_name}'") if debug else None
            return True
            
        # Check if agent is mentioned with specific patterns
        agent_patterns = [
            rf"\b{re.escape(agent_name)}\b.*\bcan help\b",
            rf"\bask\b.*\b{re.escape(agent_name)}\b",
            rf"\b{re.escape(agent_name)}\b.*\bwould be better\b",
            rf"\blet\b.*\b{re.escape(agent_name)}\b",
            rf"\b{re.escape(agent_name)}\b.*\bspecializes\b",
            rf"\b{re.escape(agent_name)}\b.*\bis better suited\b",
            rf"\bI'll?\s+(?:get|bring|call|ask)\s+(?:the\s+)?{re.escape(agent_name)}\b"
        ]
        
        for pattern in agent_patterns:
            if re.search(pattern, content.lower()):
                print(f"DEBUG: Agent '{agent_name}' mentioned through pattern matching") if debug else None
                return True
                
    return False 