# UI Assistant for AutoAgent

The UI Assistant is a specialized agent within the AutoAgent framework that helps users perform web-based tasks through a simple chat interface. It leverages the WebSurfer functionality to navigate websites, interact with web elements, and extract information.

## Features

- **Web Navigation**: Visit websites, navigate between pages, and go back in history
- **Web Interaction**: Click on elements, type text into forms, and scroll pages
- **Web Search**: Perform web searches and analyze results
- **Content Extraction**: Get the content of webpages in a readable format
- **Web Analysis**: Analyze webpages to identify interactive elements

## Getting Started

### Prerequisites

- Python 3.9+
- AutoAgent framework installed
- Required Python packages: Flask, rich, and dependencies for WebSurfer

### Installation

The UI Assistant is included with the AutoAgent framework. No additional installation is required beyond the standard AutoAgent setup.

### Running the UI Assistant

There are two ways to run the UI Assistant:

#### 1. Using the AutoAgent CLI

```bash
python -m autoagent.cli ui_assistant
```

This will start the UI Assistant web interface and provide a URL (typically http://localhost:5000) that you can open in your web browser.

#### 2. Running the Web Interface Directly

```bash
python ui_assistant_web.py
```

This will start the UI Assistant web interface directly.

## Using the UI Assistant

1. Open the UI Assistant web interface in your browser
2. You'll see a chat interface with a welcome message
3. Type your request in the input field and press Enter or click Send
4. The UI Assistant will process your request and respond in the chat

### Example Tasks

Here are some examples of tasks you can ask the UI Assistant to perform:

- "Visit wikipedia.org and search for artificial intelligence"
- "Go to github.com and show me the trending repositories"
- "Search for the latest news about machine learning"
- "Visit weather.com and tell me the forecast for New York"
- "Go to amazon.com and search for wireless headphones"

## How It Works

The UI Assistant uses the WebSurfer agent to interact with web browsers. When you send a message, the following happens:

1. Your message is sent to the server
2. The server processes your message using the UI Assistant agent
3. The agent uses WebSurfer to perform the requested web tasks
4. The results are sent back to your browser and displayed in the chat

## Troubleshooting

### Common Issues

- **Web Interface Not Starting**: Make sure you have Flask installed (`pip install flask`)
- **Agent Not Responding**: Check that the WebSurfer dependencies are installed correctly
- **Slow Performance**: Complex web tasks may take time to complete, especially on slower connections

### Logs

The UI Assistant logs information to the console. Check these logs for any error messages or warnings that might help diagnose issues.

## Contributing

Contributions to improve the UI Assistant are welcome! Please follow the standard AutoAgent contribution guidelines.

## License

The UI Assistant is part of the AutoAgent framework and is subject to the same license terms. 