from typing import List, Dict, Any, Optional
from ..logger import get_logger
from ..validation import Plan, validate_plan
from ..registry import get_registered_tools

logger = get_logger(__name__)

def format_tools(tools: Dict[str, Any]) -> str:
    """Format available tools into a clear string representation."""
    tool_descriptions = []
    for name, tool in tools.items():
        desc = tool.get('description', 'No description available')
        args = tool.get('required_args', [])
        tool_descriptions.append(f"- {name}: {desc}\n  Arguments: {', '.join(args)}")
    
    return "\n".join(tool_descriptions)

def sanitize(step: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and validate a single execution step."""
    if not isinstance(step, dict):
        raise ValueError("Step must be a dictionary")
    
    required_keys = ['tool', 'args']
    for key in required_keys:
        if key not in step:
            raise ValueError(f"Missing required key: {key}")
    
    # Remove any unexpected keys
    return {
        'tool': step['tool'],
        'args': step['args']
    }

def execute(step: Dict[str, Any]) -> Any:
    """Execute a single validated step."""
    tools = get_registered_tools()
    tool_name = step['tool']
    
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool = tools[tool_name]
    return tool['function'](**step['args'])

def native_chain(prompt: str, tools: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Execute a workflow chain using native Python implementation.
    
    Args:
        prompt: The task description or goal
        tools: Optional dict of available tools (defaults to all registered tools)
    
    Returns:
        List of execution results
    """
    if tools is None:
        tools = get_registered_tools()
    
    results = []
    complete = False
    context = {
        'tools': format_tools(tools),
        'history': [],
        'prompt': prompt
    }
    
    while not complete:
        # Get next step from LLM
        step_prompt = """
        Task: {prompt}
        Available tools:
        {tools}
        
        Previous steps:
        {history}
        
        What is the next step to complete this task?
        If the task is complete, respond with 'COMPLETE'.
        """.format(**context)
        
        from ..core import get_llm  # Import here to avoid circular dependency
        llm = get_llm()
        next_step = llm(step_prompt)
        
        if next_step.strip().upper() == 'COMPLETE':
            complete = True
            continue
        
        try:
            # Validate and execute step
            step = sanitize(next_step)
            plan = Plan(steps=[step])
            
            if not validate_plan(plan):
                logger.error("Plan validation failed")
                continue
                
            result = execute(step)
            results.append(result)
            
            # Update context
            context['history'].append({
                'step': step,
                'result': str(result)
            })
            
        except Exception as e:
            logger.error(f"Error executing step: {str(e)}")
            context['history'].append({
                'error': str(e)
            })
    
    return results 