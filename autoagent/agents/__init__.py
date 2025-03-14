# from autoagent.agents.programming_agent import get_programming_agent
# from autoagent.agents.tool_retriver_agent import get_tool_retriver_agent
# from autoagent.agents.agent_check_agent import get_agent_check_agent
# from autoagent.agents.tool_check_agent import get_tool_check_agent
# from autoagent.agents.github_agent import get_github_agent
# from autoagent.agents.programming_triage_agent import get_programming_triage_agent
# from autoagent.agents.plan_agent import get_plan_agent

# import os
# import importlib
# from autoagent.registry import registry

# # 获取当前目录下的所有 .py 文件
# current_dir = os.path.dirname(__file__)
# for file in os.listdir(current_dir):
#     if file.endswith('.py') and not file.startswith('__'):
#         module_name = file[:-3]
#         importlib.import_module(f'autoagent.agents.{module_name}')

# # 导出所有注册的 agent 创建函数
# globals().update(registry.agents)

# __all__ = list(registry.agents.keys())

import os
import importlib
from autoagent.registry import registry
from autoagent.agents.openai_assistant_agent import OpenAIAssistantAgent
from autoagent.agents.get_openai_assistant_agent import get_openai_assistant_agent

# Import TTSAgent separately to avoid circular imports
try:
    from autoagent.agents.tts_agent import TTSAgent
    from autoagent.agents.get_tts_agent import get_tts_agent
except ImportError as e:
    print(f"Warning: Failed to import TTS agent: {e}")

def import_agents_recursively(base_dir: str, base_package: str):
    """Recursively import all agents in .py files
    
    Args:
        base_dir: the root directory to start searching
        base_package: the base name of the Python package
    """
    for root, dirs, files in os.walk(base_dir):
        # get the relative path to the base directory
        rel_path = os.path.relpath(root, base_dir)
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__') and file not in ["tts_agent.py", "get_tts_agent.py"]:
                # build the module path
                if rel_path == '.':
                    # in the root directory
                    module_path = f"{base_package}.{file[:-3]}"
                else:
                    # in the subdirectory
                    package_path = rel_path.replace(os.path.sep, '.')
                    module_path = f"{base_package}.{package_path}.{file[:-3]}"
                
                try:
                    importlib.import_module(module_path)
                except Exception as e:
                    print(f"Warning: Failed to import {module_path}: {e}")

# get the current directory and import all agents
current_dir = os.path.dirname(__file__)
import_agents_recursively(current_dir, 'autoagent.agents')

# export all agent creation functions
globals().update(registry.agents)
globals().update(registry.plugin_agents)

# Add TTSAgent and get_tts_agent to __all__ if they were successfully imported
__all__ = list(registry.agents.keys()) + ["OpenAIAssistantAgent", "get_openai_assistant_agent"]
if "TTSAgent" in globals() and "get_tts_agent" in globals():
    __all__ += ["TTSAgent", "get_tts_agent"]