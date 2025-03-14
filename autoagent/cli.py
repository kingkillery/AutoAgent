import warnings
import os

# Redirect stderr to null only when importing pydub to suppress its warnings
import sys
import contextlib
import io

# Configure environment variables for pydub
os.environ["PYDUB_USE_FFMPEG"] = "False"
os.environ["PYDUB_NO_FFMPEG"] = "True"

# Add filter to ignore pydub warnings as a backup strategy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# Standard imports - keep these after the warning configuration
import click
import importlib
from autoagent import MetaChain
from autoagent.util import debug_print
import asyncio
from constant import DOCKER_WORKPLACE_NAME
from autoagent.io_utils import read_yaml_file, get_md5_hash_bytext, read_file
from autoagent.environment.utils import setup_metachain
from autoagent.types import Response
from autoagent import MetaChain
from autoagent.util import ask_text, single_select_menu, print_markdown, debug_print, UserCompleter
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
import argparse
from datetime import datetime
from autoagent.agents.meta_agent import tool_editor, agent_editor
from autoagent.tools.meta.edit_tools import list_tools
from autoagent.tools.meta.edit_agents import list_agents
from loop_utils.font_page import MC_LOGO, version_table, NOTES, GOODBYE_LOGO
from rich.live import Live
from autoagent.environment.docker_env import DockerEnv, DockerConfig, check_container_ports
from autoagent.environment.browser_env import BrowserEnv
from autoagent.environment.markdown_browser import RequestsMarkdownBrowser
from evaluation.utils import update_progress, check_port_available, run_evaluation, clean_msg
import os
import os.path as osp
from autoagent.agents import get_system_triage_agent
from autoagent.logger import LoggerManager, MetaChainLogger 
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.panel import Panel
import re
from autoagent.cli_utils.metachain_meta_agent import meta_agent
from autoagent.cli_utils.metachain_meta_workflow import meta_workflow
from autoagent.cli_utils.file_select import select_and_copy_files
from evaluation.utils import update_progress, check_port_available, run_evaluation, clean_msg
from constant import COMPLETION_MODEL
from typing import Optional
import subprocess

# Initialize console for rich output
console = Console()

# ASCII Art Logo
LOGO = """
    _         _        _                     _   
   / \\  _   _| |_ ___ / \\   __ _  ___ _ __ | |_ 
  / _ \\| | | | __/ _ \\ / \\ / _` |/ _ \\ '_ \\| __|
 / ___ \\ |_| | || (_) / _ \\ (_| |  __/ | | | |_ 
/_/   \\_\\__,_|\\__\\___/_/ \\_\\__, |\\___|_| |_|\\__|
                           |___/               
"""

VERSION = "1.0.0"

# Create version table
def get_version_table():
    table = Table(title="AutoAgent CLI", show_header=False, box=None)
    table.add_column("Key", style="bold blue")
    table.add_column("Value", style="green")
    table.add_row("Version", VERSION)
    table.add_row("Model", COMPLETION_MODEL)
    return table

# Important notes
NOTES = """
- Type 'exit' at any prompt to return to the main menu
- Use @agent_name to specify which agent to use
- For file uploads, use @Upload_files
"""

# Goodbye message
GOODBYE_LOGO = """
Thank you for using AutoAgent CLI!
See you next time!
"""

@click.group()
def cli():
    """AutoAgent CLI - A powerful framework for AI agents"""
    pass

@cli.command()
@click.option('--model', default='gpt-4o', help='The name of the model to use')
@click.option('--agent_func', default='get_dummy_agent', help='The function to get the agent')
@click.option('--query', default='...', help='The user query to the agent')
@click.argument('context_variables', nargs=-1)
def agent(model: str, agent_func: str, query: str, context_variables):
    """
    Run an agent with a given model, agent function, query, and context variables.
    
    Examples:
        python -m autoagent.cli agent --model=gpt-4o --agent_func=get_weather_agent --query="What is the weather in Tokyo?" city=Tokyo unit=C
    """
    # Parse context variables
    context_storage = {}
    for arg in context_variables:
        if '=' in arg:
            key, value = arg.split('=', 1)
            context_storage[key] = value
    
    # Import agent function
    try:
        agent_module = importlib.import_module('autoagent.agents')
        agent_func = getattr(agent_module, agent_func)
    except AttributeError:
        console.print(f"[bold red]Error:[/bold red] Agent function '{agent_func}' not found in 'autoagent.agents'")
        console.print("Available agent functions:")
        for name in dir(agent_module):
            if name.startswith('get_') and callable(getattr(agent_module, name)):
                console.print(f"  - {name}")
        return
    
    # Create agent and run
    try:
        agent = agent_func(model)
        mc = MetaChain()
        messages = [{"role": "user", "content": query}]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Running agent...", total=None)
            response = mc.run(agent, messages, context_storage, debug=True)
        
        debug_print(True, response.messages[-1]['content'], title=f'Result from {agent.name}', color='pink3')
        return response.messages[-1]['content']
    except Exception as e:
        console.print(f"[bold red]Error running agent:[/bold red] {str(e)}")
        return str(e)

@cli.command()
@click.option('--workflow_name', required=True, help='The name of the workflow')
@click.option('--system_input', default='...', help='The user query to the workflow')
def workflow(workflow_name: str, system_input: str):
    """
    Run a workflow with a given name and system input.
    
    Examples:
        python -m autoagent.cli workflow --workflow_name=research_workflow --system_input="Research quantum computing"
    """
    try:
        result = asyncio.run(async_workflow(workflow_name, system_input))
        return result
    except Exception as e:
        console.print(f"[bold red]Error running workflow:[/bold red] {str(e)}")
        return str(e)

async def async_workflow(workflow_name: str, system_input: str):
    """Asynchronous implementation of the workflow function"""
    try:
        workflow_module = importlib.import_module('autoagent.workflows')
        workflow_func = getattr(workflow_module, workflow_name)
    except AttributeError:
        available_workflows = [name for name in dir(workflow_module) 
                              if not name.startswith('_') and callable(getattr(workflow_module, name))]
        raise ValueError(f"Workflow '{workflow_name}' not found. Available workflows: {', '.join(available_workflows)}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Running workflow...", total=None)
        result = await workflow_func(system_input)
    
    debug_print(True, result, title=f'Result from {workflow_name}', color='pink3')
    return result

def clear_screen():
    console.print("[bold green]Coming soon...[/bold green]")
    print('\033[u\033[J\033[?25h', end='')  # Restore cursor and clear everything after it, show cursor

def get_config(container_name, port, test_pull_name="main", git_clone=False):
    container_name = container_name
    
    port_info = check_container_ports(container_name)
    if port_info:
        port = port_info[0]
    else:
        # while not check_port_available(port):
        #     port += 1
        # 使用文件锁来确保端口分配的原子性
        import filelock
        lock_file = os.path.join(os.getcwd(), ".port_lock")
        lock = filelock.FileLock(lock_file)
        
        with lock:
            port = port
            while not check_port_available(port):
                port += 1
                print(f'{port} is not available, trying {port+1}')
            # 立即标记该端口为已使用
            with open(os.path.join(os.getcwd(), f".port_{port}"), 'w') as f:
                f.write(container_name)
    local_root = os.path.join(os.getcwd(), f"workspace_meta_showcase", f"showcase_{container_name}")
    os.makedirs(local_root, exist_ok=True)
    docker_config = DockerConfig(
        workplace_name=DOCKER_WORKPLACE_NAME,
        container_name=container_name,
        communication_port=port,
        conda_path='/root/miniconda3',
        local_root=local_root,
        test_pull_name=test_pull_name,
        git_clone=git_clone
    )
    return docker_config

def create_environment(docker_config: DockerConfig):
    """
    1. create the code environment
    2. create the web environment
    3. create the file environment
    """
    code_env = DockerEnv(docker_config)
    code_env.init_container()
    
    web_env = BrowserEnv(browsergym_eval_env = None, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name)
    file_env = RequestsMarkdownBrowser(viewport_size=1024 * 5, local_root=docker_config.local_root, workplace_name=docker_config.workplace_name, downloads_folder=os.path.join(docker_config.local_root, docker_config.workplace_name, "downloads"))
    
    return code_env, web_env, file_env

def update_guidance(context_variables): 
    # print the logo
    logo_text = Text(MC_LOGO, justify="center")
    console.print(Panel(logo_text, style="bold salmon1", expand=True))
    console.print(version_table)
    console.print(Panel(NOTES,title="Important Notes", expand=True))

@cli.command(name='main')  # 修改这里，使用连字符
@click.option('--container_name', default='auto_agent', help='the function to get the agent')
@click.option('--port', default=12347, help='the port to run the container')
@click.option('--test_pull_name', default='autoagent_mirror', help='the name of the test pull')
@click.option('--git_clone', default=True, help='whether to clone a mirror of the repository')
def main(container_name: str, port: int, test_pull_name: str, git_clone: bool):
    """
    Start the main AutoAgent interactive shell with Docker environment
    """
    try:
        model = COMPLETION_MODEL
        print('\033[s\033[?25l', end='')  # Save cursor position and hide cursor
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True  # 这会让进度条完成后消失
        ) as progress:
            task = progress.add_task("[cyan]Initializing...", total=None)
            
            progress.update(task, description="[cyan]Initializing config...[/cyan]\n")
            docker_config = get_config(container_name, port, test_pull_name, git_clone)
            
            progress.update(task, description="[cyan]Setting up logger...[/cyan]\n")
            log_path = osp.join("casestudy_results", 'logs', f'agent_{container_name}_{model}.log')
            LoggerManager.set_logger(MetaChainLogger(log_path = None))
            
            progress.update(task, description="[cyan]Creating environment...[/cyan]\n")
            code_env, web_env, file_env = create_environment(docker_config)
            
            progress.update(task, description="[cyan]Setting up autoagent...[/cyan]\n")
        
        clear_screen()

        context_variables = {"working_dir": docker_config.workplace_name, "code_env": code_env, "web_env": web_env, "file_env": file_env}

        # select the mode
        while True:
            update_guidance(context_variables)
            mode = single_select_menu(['user mode', 'agent editor', 'workflow editor', 'exit'], "Please select the mode:")
            match mode:
                case 'user mode':
                    clear_screen()
                    user_mode(model, context_variables, False)
                case 'agent editor':
                    clear_screen()
                    meta_agent(model, context_variables, False)
                case 'workflow editor':
                    clear_screen()
                    meta_workflow(model, context_variables, False)
                case 'exit':
                    console = Console()
                    logo_text = Text(GOODBYE_LOGO, justify="center")
                    console.print(Panel(logo_text, style="bold salmon1", expand=True))
                    break
    except Exception as e:
        console.print(f"[bold red]Error in main mode:[/bold red] {str(e)}")
        console.print("[yellow]Tip:[/yellow] Make sure Docker is running and properly configured.")

def user_mode(model: str, context_variables: dict, debug: bool = False):
    """Interactive user mode with agent selection"""
    try:
        from autoagent.agents import get_system_triage_agent
        
        # Get logger
        logger = LoggerManager.get_logger()
        
        # Get system triage agent
        system_triage_agent = get_system_triage_agent(model)
        assert system_triage_agent.agent_teams != {}, "System Triage Agent must have agent teams"
        
        # Set up agents dictionary
        messages = []
        agent = system_triage_agent
        agents = {system_triage_agent.name.replace(' ', '_'): system_triage_agent}
        for agent_name in system_triage_agent.agent_teams.keys():
            # Fix: Get the agent directly instead of accessing .agent attribute
            agent_instance = system_triage_agent.agent_teams[agent_name](model)
            agents[agent_name.replace(' ', '_')] = agent_instance
        agents["Upload_files"] = "select"
        
        # Create session with autocompletion
        style = Style.from_dict({
            'bottom-toolbar': 'bg:#333333 #ffffff',
        })
        
        session = PromptSession(
            completer=UserCompleter(agents.keys()),
            complete_while_typing=True,
            style=style
        )
        
        # Initialize MetaChain
        client = MetaChain(log_path=logger)
        upload_infos = []
        
        # Main interaction loop
        while True:
            query = session.prompt(
                'Tell me what you want to do (type "exit" to quit): ',
                bottom_toolbar=HTML('<b>Prompt:</b> Enter <b>@</b> to mention Agents')
            )
            
            if query.strip().lower() == 'exit':
                console.print("[bold green]Exiting user mode...[/bold green]")
                break
            
            # Check for agent mentions
            words = query.split()
            console.print(f"[bold green]Your request: {query}[/bold green]")
            for word in words:
                if word.startswith('@') and word[1:] in agents.keys():
                    agent = agents[word.replace('@', '')]
            
            # Handle agent or file upload
            if hasattr(agent, "name"):
                agent_name = agent.name
                console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] will help you, be patient...[/bold green]")
                
                # Add uploaded files info if any
                if len(upload_infos) > 0:
                    query = "{}\n\nUser uploaded files:\n{}".format(query, "\n".join(upload_infos))
                
                # Run the agent
                messages.append({"role": "user", "content": query})
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True
                ) as progress:
                    task = progress.add_task(f"[cyan]Running {agent_name}...", total=None)
                    response = client.run(agent, messages, context_variables, debug=debug)
                
                # Process response
                messages.extend(response.messages)
                model_answer = response.messages[-1]['content']
                
                # Display response
                console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] has finished:[/bold green]")
                print_markdown(model_answer)
                
                # Update agent based on response
                agent = response.agent
            elif agent == "select":
                # Handle file upload
                code_env = context_variables["code_env"]
                local_workplace = code_env.local_workplace
                docker_workplace = code_env.docker_workplace
                files_dir = os.path.join(local_workplace, "files")
                docker_files_dir = os.path.join(docker_workplace, "files")
                os.makedirs(files_dir, exist_ok=True)
                upload_infos.extend(select_and_copy_files(files_dir, console, docker_files_dir))
                agent = agents["System_Triage_Agent"]
            else:
                console.print(f"[bold red]Unknown agent: {agent}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error in user mode:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

@cli.command(name='deep-research')  # 修改这里，使用连字符
@click.option('--container_name', default='deepresearch', help='the function to get the agent')
@click.option('--port', default=12346, help='the port to run the container')
def deep_research(container_name: str, port: int):
    """
    Run deep research with a given model, container name, port
    """ 
    model = COMPLETION_MODEL
    print('\033[s\033[?25l', end='')  # Save cursor position and hide cursor
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True  # 这会让进度条完成后消失
    ) as progress:
        task = progress.add_task("[cyan]Initializing...", total=None)
        
        progress.update(task, description="[cyan]Initializing config...[/cyan]\n")
        docker_config = get_config(container_name, port)
        
        progress.update(task, description="[cyan]Setting up logger...[/cyan]\n")
        log_path = osp.join("casestudy_results", 'logs', f'agent_{container_name}_{model}.log')
        LoggerManager.set_logger(MetaChainLogger(log_path = None))
        
        progress.update(task, description="[cyan]Creating environment...[/cyan]\n")
        code_env, web_env, file_env = create_environment(docker_config)
        
        progress.update(task, description="[cyan]Setting up autoagent...[/cyan]\n")
    
    clear_screen()

    context_variables = {"working_dir": docker_config.workplace_name, "code_env": code_env, "web_env": web_env, "file_env": file_env}

    update_guidance(context_variables)

    logger = LoggerManager.get_logger()
    console = Console()
    system_triage_agent = get_system_triage_agent(model)
    assert system_triage_agent.agent_teams != {}, "System Triage Agent must have agent teams"
    messages = []
    agent = system_triage_agent
    agents = {system_triage_agent.name.replace(' ', '_'): system_triage_agent}
    for agent_name in system_triage_agent.agent_teams.keys():
        agents[agent_name.replace(' ', '_')] = system_triage_agent.agent_teams[agent_name]("placeholder").agent
    agents["Upload_files"] = "select"
    style = Style.from_dict({
        'bottom-toolbar': 'bg:#333333 #ffffff',
    })

    # 创建会话
    session = PromptSession(
        completer=UserCompleter(agents.keys()),
        complete_while_typing=True,
        style=style
    )
    client = MetaChain(log_path=logger)
    while True: 
        # query = ask_text("Tell me what you want to do:")
        query = session.prompt(
            'Tell me what you want to do (type "exit" to quit): ',
            bottom_toolbar=HTML('<b>Prompt:</b> Enter <b>@</b> to mention Agents')
        )
        if query.strip().lower() == 'exit':
            # logger.info('User mode completed.  See you next time! :waving_hand:', color='green', title='EXIT')
            
            logo_text = "See you next time! :waving_hand:"
            console.print(Panel(logo_text, style="bold salmon1", expand=True))
            break
        words = query.split()
        console.print(f"[bold green]Your request: {query}[/bold green]", end=" ")
        for word in words:
            if word.startswith('@') and word[1:] in agents.keys():
                # print(f"[bold magenta]{word}[bold magenta]", end=' ') 
                agent = agents[word.replace('@', '')]
            else:
                # print(word, end=' ')
                pass
        print()
        
        if hasattr(agent, "name"): 
            agent_name = agent.name
            console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] will help you, be patient...[/bold green]")
            messages.append({"role": "user", "content": query})
            response = client.run(agent, messages, context_variables, debug=False)
            messages.extend(response.messages)
            model_answer_raw = response.messages[-1]['content']

            # attempt to parse model_answer
            if model_answer_raw.startswith('Case resolved'):
                model_answer = re.findall(r'<solution>(.*?)</solution>', model_answer_raw, re.DOTALL)
                if len(model_answer) == 0:
                    model_answer = model_answer_raw
                else:
                    model_answer = model_answer[0]
            else: 
                model_answer = model_answer_raw
            console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] has finished with the response:\n[/bold green] [bold blue]{model_answer}[/bold blue]")
            agent = response.agent
        elif agent == "select": 
            code_env: DockerEnv = context_variables["code_env"]
            local_workplace = code_env.local_workplace
            files_dir = os.path.join(local_workplace, "files")
            os.makedirs(files_dir, exist_ok=True)
            select_and_copy_files(files_dir, console)
        else: 
            console.print(f"[bold red]Unknown agent: {agent}[/bold red]")
    
# List available agents
@cli.command()
def list_agents():
    """List all available agents in the system"""
    try:
        from autoagent.registry import Registry
        registry = Registry()
        agents = registry._registry.get("agents", {})
        
        if not agents:
            console.print("[yellow]No agents found in the registry[/yellow]")
            return
        
        table = Table(title="Available Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        
        for name, func in agents.items():
            description = func.__doc__ or "No description available"
            table.add_row(name, description.strip())
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error listing agents:[/bold red] {str(e)}")

# List available tools
@cli.command()
def list_tools():
    """List all available tools in the system"""
    try:
        from autoagent.registry import Registry
        registry = Registry()
        tools = registry._registry.get("tools", {})
        
        if not tools:
            console.print("[yellow]No tools found in the registry[/yellow]")
            return
        
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        
        for name, func in tools.items():
            description = func.__doc__ or "No description available"
            table.add_row(name, description.strip())
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error listing tools:[/bold red] {str(e)}")

# List available workflows
@cli.command()
def list_workflows():
    """List all available workflows in the system"""
    try:
        from autoagent.registry import Registry
        registry = Registry()
        workflows = registry._registry.get("workflows", {})
        
        if not workflows:
            console.print("[yellow]No workflows found in the registry[/yellow]")
            return
        
        table = Table(title="Available Workflows")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        
        for name, func in workflows.items():
            description = func.__doc__ or "No description available"
            table.add_row(name, description.strip())
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error listing workflows:[/bold red] {str(e)}")

# Interactive mode
@cli.command()
@click.option('--model', default=COMPLETION_MODEL, help='The model to use')
def interactive(model: str):
    """Start an interactive session with AutoAgent"""
    # Display welcome message
    logo_text = Text(LOGO, justify="center")
    console.print(Panel(logo_text, style="bold green", expand=False))
    console.print(get_version_table())
    console.print(Panel(NOTES, title="Important Notes", expand=False))
    
    # Set up logger
    log_path = os.path.join("logs", f'interactive_{model}.log')
    LoggerManager.set_logger(MetaChainLogger(log_path=log_path))
    
    # Get system triage agent
    try:
        from autoagent.agents import get_system_triage_agent
        system_triage_agent = get_system_triage_agent(model)
        
        # Set up agents dictionary
        agents = {system_triage_agent.name.replace(' ', '_'): system_triage_agent}
        if hasattr(system_triage_agent, 'agent_teams'):
            for agent_name in system_triage_agent.agent_teams.keys():
                # Fix: Get the agent directly instead of accessing .agent attribute
                agent_instance = system_triage_agent.agent_teams[agent_name](model)
                agents[agent_name.replace(' ', '_')] = agent_instance
        agents["Upload_files"] = "select"
        
        # Create session with autocompletion
        style = Style.from_dict({
            'bottom-toolbar': 'bg:#333333 #ffffff',
        })
        
        session = PromptSession(
            completer=UserCompleter(agents.keys()),
            complete_while_typing=True,
            style=style
        )
        
        # Initialize MetaChain and variables
        client = MetaChain(log_path=LoggerManager.get_logger())
        messages = []
        agent = system_triage_agent
        upload_infos = []
        
        # Main interaction loop
        while True:
            query = session.prompt(
                'Tell me what you want to do (type "exit" to quit): ',
                bottom_toolbar=HTML('<b>Prompt:</b> Enter <b>@</b> to mention Agents')
            )
            
            if query.strip().lower() == 'exit':
                logo_text = Text(GOODBYE_LOGO, justify="center")
                console.print(Panel(logo_text, style="bold green", expand=False))
                break
            
            # Check for agent mentions
            words = query.split()
            console.print(f"[bold green]Your request: {query}[/bold green]")
            for word in words:
                if word.startswith('@') and word[1:] in agents.keys():
                    agent = agents[word.replace('@', '')]
            
            # Handle agent or file upload
            if hasattr(agent, "name"):
                agent_name = agent.name
                console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] will help you, be patient...[/bold green]")
                
                # Add uploaded files info if any
                if len(upload_infos) > 0:
                    query = "{}\n\nUser uploaded files:\n{}".format(query, "\n".join(upload_infos))
                
                # Run the agent
                messages.append({"role": "user", "content": query})
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True
                ) as progress:
                    task = progress.add_task(f"[cyan]Running {agent_name}...", total=None)
                    response = client.run(agent, messages, {}, debug=False)
                
                # Process response
                messages.extend(response.messages)
                model_answer = response.messages[-1]['content']
                
                # Display response
                console.print(f"[bold green][bold magenta]@{agent_name}[/bold magenta] has finished:[/bold green]")
                print_markdown(model_answer)
                
                # Update agent based on response
                agent = response.agent
            elif agent == "select":
                # Handle file upload
                console.print("[bold blue]File Upload Mode[/bold blue]")
                upload_infos.extend(select_and_copy_files("files", console))
                agent = agents["System_Triage_Agent"]
            else:
                console.print(f"[bold red]Unknown agent: {agent}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error in interactive mode:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())

# TTS command
@cli.command()
@click.option('--model', default='gpt-4o', help='The model to use')
@click.option('--text', required=True, help='The text to convert to speech')
@click.option('--output_dir', default='audio_output', help='Directory to save the audio file')
@click.option('--speaker_id', default=None, help='Optional speaker ID for voice selection')
def tts(model: str, text: str, output_dir: str, speaker_id: Optional[str]):
    """
    Convert text to speech using the TTS agent
    
    Examples:
        python -m autoagent.cli tts --text="Hello, how are you today?" --output_dir=audio_output
    """
    try:
        # Import TTS agent
        from autoagent.agents.get_tts_agent import get_tts_agent
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create TTS agent
        tts_agent = get_tts_agent(model, output_dir=output_dir, speaker_id=speaker_id)
        
        # Run TTS agent
        mc = MetaChain()
        messages = [{"role": "user", "content": text}]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Converting text to speech...", total=None)
            response = mc.run(tts_agent, messages, {}, debug=True)
        
        # Get the audio file path from the response
        result = response.messages[-1]['content']
        console.print(f"[bold green]Text-to-Speech completed:[/bold green] {result}")
        
        # Suggest playing the audio
        console.print("[bold blue]To play the generated audio, use:[/bold blue]")
        console.print(f"python play_audio.py {output_dir}")
        
        return result
    except ImportError:
        console.print("[bold red]Error:[/bold red] TTS agent not found. Make sure you have set up the TTS functionality.")
        console.print("See TTS_README.md for setup instructions.")
    except Exception as e:
        console.print(f"[bold red]Error converting text to speech:[/bold red] {str(e)}")
        return str(e)

@cli.command()
def ui_assistant():
    """Launch the UI Assistant web interface."""
    console = Console()
    console.print("[bold green]Launching UI Assistant web interface...[/bold green]")
    console.print("[bold blue]Access the UI at http://localhost:5000 once the server starts.[/bold blue]")
    
    # Get the path to the ui_assistant_web.py file
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui_assistant_web.py")
    
    # Check if the file exists
    if not os.path.exists(script_path):
        console.print("[bold red]Error: UI Assistant web interface script not found.[/bold red]")
        console.print(f"Expected path: {script_path}")
        return
    
    # Launch the web interface
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error launching UI Assistant: {str(e)}[/bold red]")
    except KeyboardInterrupt:
        console.print("[yellow]UI Assistant web interface stopped.[/yellow]")

# Run the CLI when the module is executed directly
if __name__ == "__main__":
    # Print a welcome message
    console.print("[bold green]Welcome to AutoAgent CLI![/bold green]")
    console.print("[bold blue]Type 'python -m autoagent.cli --help' for available commands.[/bold blue]")
    
    # Register all commands
    cli.add_command(agent)
    cli.add_command(workflow)
    cli.add_command(main)
    cli.add_command(deep_research)
    cli.add_command(list_agents)
    cli.add_command(list_tools)
    cli.add_command(list_workflows)
    cli.add_command(interactive)
    cli.add_command(tts)
    cli.add_command(ui_assistant)
    
    # Run the CLI
    cli()
    