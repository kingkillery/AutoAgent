from autoagent.types import Agent

# BaseAgent extends the Agent class for backward compatibility
class BaseAgent(Agent):
    """
    A base class for all agents in the AutoAgent framework.
    This class is a thin wrapper around the Agent class from autoagent.types.
    It exists primarily for backward compatibility.
    """
    pass 