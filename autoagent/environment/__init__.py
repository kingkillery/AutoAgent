from autoagent.environment.docker_env import DockerEnv, DockerConfig
from autoagent.environment.local_env import LocalEnv
from autoagent.environment.browser_env import BrowserEnv, BrowserInitException
from autoagent.environment.markdown_browser import RequestsMarkdownBrowser
from autoagent.environment.browser_manager import BrowserManager
from autoagent.environment.browser_controller import BrowserController
from autoagent.environment.web_surfer import WebSurfer
from autoagent.environment.api_handlers import register_api_handlers, GoogleSearchApiHandler, WikipediaApiHandler

# Constants
VIEWPORT = {'width': 1280, 'height': 720}

from .utils import setup_metachain