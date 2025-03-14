import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autoagent import MetaChain
from autoagent.agents.ui_assistant_agent import get_ui_assistant_agent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UIAssistantWeb")

app = Flask(__name__)

# Global variables
messages = []
context_storage = {}
agent = None
client = None
active_task = None
task_result = None
task_error = None

def initialize_agent():
    """Initialize the UI Assistant agent and MetaChain client"""
    global agent, client
    try:
        # Create the UI Assistant agent
        agent = get_ui_assistant_agent("gpt-4o")
        
        # Create MetaChain client
        client = MetaChain()
        
        logger.info("UI Assistant agent and MetaChain client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        return False

def run_agent_task(user_message):
    """Run the agent task in a separate thread"""
    global active_task, task_result, task_error, messages, context_storage
    
    try:
        # Add user message to messages
        messages.append({"role": "user", "content": user_message})
        
        # Run the agent
        response = client.run(agent, messages, context_storage, debug=False)
        
        # Process response
        messages.extend(response.messages)
        task_result = response.messages[-1]['content']
        active_task = None
    except Exception as e:
        logger.error(f"Error running agent task: {str(e)}")
        task_error = str(e)
        active_task = None

@app.route('/')
def index():
    """Render the main UI page"""
    return render_template('ui_assistant.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle sending a message to the UI Assistant"""
    global active_task
    
    # Get the message from the request
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Check if there's already an active task
    if active_task and active_task.is_alive():
        return jsonify({"error": "There's already an active task running"}), 429
    
    # Start a new task
    active_task = threading.Thread(target=run_agent_task, args=(user_message,))
    active_task.start()
    
    return jsonify({"status": "Message received, processing started"}), 202

@app.route('/check_status', methods=['GET'])
def check_status():
    """Check the status of the current task"""
    global active_task, task_result, task_error
    
    if active_task and active_task.is_alive():
        return jsonify({"status": "processing"}), 200
    
    if task_error:
        error = task_error
        task_error = None
        return jsonify({"status": "error", "error": error}), 500
    
    if task_result:
        result = task_result
        task_result = None
        return jsonify({"status": "complete", "response": result}), 200
    
    return jsonify({"status": "idle"}), 200

@app.route('/get_messages', methods=['GET'])
def get_messages():
    """Get all messages in the conversation"""
    global messages
    
    formatted_messages = []
    for msg in messages:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_messages.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": timestamp
        })
    
    return jsonify({"messages": formatted_messages}), 200

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation"""
    global messages, context_storage, active_task, task_result, task_error
    
    # Check if there's an active task
    if active_task and active_task.is_alive():
        return jsonify({"error": "Cannot reset while a task is running"}), 429
    
    # Reset all variables
    messages = []
    context_storage = {}
    task_result = None
    task_error = None
    
    return jsonify({"status": "Conversation reset successfully"}), 200

def create_templates_directory():
    """Create the templates directory if it doesn't exist"""
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UI Assistant Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #001529;
            color: #fff;
        }
        .chat-container {
            max-width: 500px;
            margin: 20px auto;
            border: 1px solid #444;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        .chat-header {
            background-color: #002140;
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid #444;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .chat-header img {
            height: 40px;
            margin-right: 10px;
        }
        .chat-messages {
            flex: 1;
            padding: 10px;
            overflow-y: auto;
            background-color: #001529;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 4px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .bot {
            background-color: #003366;
            align-self: flex-start;
            margin-right: auto;
            color: #4CAF50;
        }
        .user {
            background-color: #1a1a1a;
            align-self: flex-end;
            margin-left: auto;
            color: #ffffff;
        }
        .timestamp {
            font-size: 0.7em;
            color: #888;
            margin-top: 4px;
        }
        .chat-input {
            display: flex;
            padding: 10px;
            background-color: #002140;
            border-top: 1px solid #444;
        }
        .chat-input input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #444;
            border-radius: 4px;
            background-color: #001529;
            color: #fff;
        }
        .chat-input button {
            margin-left: 8px;
            padding: 8px 16px;
            background-color: #1890ff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .chat-input button:hover {
            background-color: #40a9ff;
        }
        .chat-input button:disabled {
            background-color: #888;
            cursor: not-allowed;
        }
        .loading {
            text-align: center;
            padding: 10px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <img src="/static/ui_assistant_logo.svg" alt="UI Assistant Logo">
            <h2>UI Assistant Chat</h2>
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="message bot">
                <div class="content">Hi! I am your UI assistant. I can perform web tasks for you. What can I help you with?</div>
                <div class="timestamp" id="initial-timestamp"></div>
            </div>
        </div>
        <div class="chat-input">
            <input type="text" id="message-input" placeholder="Type your message...">
            <button id="send-button">Send</button>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const messagesContainer = document.getElementById('chat-messages');
            const messageInput = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');
            const initialTimestamp = document.getElementById('initial-timestamp');
            
            // Set initial timestamp
            const now = new Date();
            initialTimestamp.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')} - BOT`;
            
            let isProcessing = false;
            
            // Function to add a message to the chat
            function addMessage(role, content, timestamp) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role === 'user' ? 'user' : 'bot'}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'content';
                contentDiv.textContent = content;
                
                const timestampDiv = document.createElement('div');
                timestampDiv.className = 'timestamp';
                timestampDiv.textContent = `${timestamp} - ${role.toUpperCase()}`;
                
                messageDiv.appendChild(contentDiv);
                messageDiv.appendChild(timestampDiv);
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            // Function to send a message
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message || isProcessing) return;
                
                // Add user message to chat
                const now = new Date();
                const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
                addMessage('user', message, timestamp);
                
                // Clear input
                messageInput.value = '';
                
                // Set processing state
                isProcessing = true;
                sendButton.disabled = true;
                
                // Add loading indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'loading';
                loadingDiv.textContent = 'Processing...';
                messagesContainer.appendChild(loadingDiv);
                
                try {
                    // Send message to server
                    const response = await fetch('/send_message', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ message })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to send message');
                    }
                    
                    // Poll for response
                    await pollForResponse();
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('bot', `Error: ${error.message}`, timestamp);
                } finally {
                    // Remove loading indicator
                    messagesContainer.removeChild(loadingDiv);
                    
                    // Reset processing state
                    isProcessing = false;
                    sendButton.disabled = false;
                }
            }
            
            // Function to poll for response
            async function pollForResponse() {
                let attempts = 0;
                const maxAttempts = 60; // 30 seconds (500ms * 60)
                
                while (attempts < maxAttempts) {
                    try {
                        const response = await fetch('/check_status');
                        const data = await response.json();
                        
                        if (data.status === 'complete') {
                            const now = new Date();
                            const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
                            addMessage('bot', data.response, timestamp);
                            return;
                        } else if (data.status === 'error') {
                            const now = new Date();
                            const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
                            addMessage('bot', `Error: ${data.error}`, timestamp);
                            return;
                        } else if (data.status === 'idle') {
                            // No active task, stop polling
                            return;
                        }
                        
                        // Still processing, wait and try again
                        await new Promise(resolve => setTimeout(resolve, 500));
                        attempts++;
                    } catch (error) {
                        console.error('Error polling for response:', error);
                        throw error;
                    }
                }
                
                throw new Error('Timed out waiting for response');
            }
            
            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            
            messageInput.addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            });
        });
    </script>
</body>
</html>
"""
    
    with open('templates/ui_assistant.html', 'w') as f:
        f.write(html_template)

if __name__ == '__main__':
    # Create templates directory and HTML file
    create_templates_directory()
    
    # Initialize the agent
    if initialize_agent():
        # Start the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize agent. Exiting.")
        sys.exit(1) 