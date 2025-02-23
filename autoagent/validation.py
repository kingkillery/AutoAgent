from typing import Dict, Any, List
from dataclasses import dataclass
from .registry import get_registered_tools
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class Plan:
    """Represents an execution plan with steps and safety constraints."""
    steps: List[Dict[str, Any]]
    risk_level: float = 0.0
    safety_constraints: Dict[str, Any] = None

def check_tools(plan: Plan) -> bool:
    """Verify that all tools referenced in the plan exist."""
    registered_tools = get_registered_tools()
    
    for step in plan.steps:
        if 'tool' not in step:
            logger.warning(f"Step missing tool specification: {step}")
            return False
        
        tool_name = step['tool']
        if tool_name not in registered_tools:
            logger.warning(f"Unknown tool referenced: {tool_name}")
            return False
    
    return True

def check_args(plan: Plan) -> bool:
    """Validate argument schemas for each tool in the plan."""
    registered_tools = get_registered_tools()
    
    for step in plan.steps:
        tool_name = step['tool']
        tool_spec = registered_tools[tool_name]
        
        required_args = tool_spec.get('required_args', [])
        provided_args = step.get('args', {})
        
        # Check required arguments
        for arg in required_args:
            if arg not in provided_args:
                logger.warning(f"Missing required argument {arg} for tool {tool_name}")
                return False
    
    return True

def check_safety(plan: Plan) -> bool:
    """Check safety constraints and risk level."""
    if plan.risk_level > 0.7:  # High risk threshold
        logger.warning(f"Plan risk level too high: {plan.risk_level}")
        return False
        
    if plan.safety_constraints:
        # Check each defined safety constraint
        for constraint, value in plan.safety_constraints.items():
            if not value:
                logger.warning(f"Safety constraint violation: {constraint}")
                return False
    
    return True

def validate_plan(plan: Plan) -> bool:
    """Main validation function that checks all aspects of a plan."""
    validations = [
        (check_tools, "Tool validation"),
        (check_args, "Argument validation"),
        (check_safety, "Safety validation")
    ]
    
    for validation_fn, validation_name in validations:
        try:
            if not validation_fn(plan):
                logger.error(f"{validation_name} failed")
                return False
        except Exception as e:
            logger.error(f"Error during {validation_name}: {str(e)}")
            return False
    
    logger.info("Plan validation successful")
    return True 