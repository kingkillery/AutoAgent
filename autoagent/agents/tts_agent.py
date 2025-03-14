from typing import Dict, List, Any, Optional, Union
from pydantic import Field
import os
import json
import traceback

from autoagent.agents.base_agent import BaseAgent
from autoagent.types import Response, Message

class TTSAgent(BaseAgent):
    """
    Text-to-Speech Agent using the CSM-1B model from Sesame AI.
    This agent converts text to speech using a state-of-the-art neural TTS model.
    """
    name: str = "Text-to-Speech Agent"
    description: str = "Converts text to speech using the CSM-1B model from Sesame AI"
    
    # Configuration for the TTS model
    output_dir: str = Field(default="tts_outputs")
    speaker_id: int = Field(default=0)
    
    # Conversation history for context
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"TTSAgent initialized with output_dir: {os.path.abspath(self.output_dir)}")
    
    async def _arun(self, messages: List[Message], context_variables: Dict[str, Any] = None) -> Response:
        """
        Asynchronous run method for the TTS agent.
        
        Args:
            messages: List of messages from the conversation
            context_variables: Additional context information
            
        Returns:
            Response: Agent response with TTS output information
        """
        return await self.run(messages, context_variables)
    
    def run(self, messages: List[Message], context_variables: Dict[str, Any] = None) -> Response:
        """
        Run the TTS agent to convert text to speech.
        
        Args:
            messages: List of messages from the conversation
            context_variables: Additional context information
            
        Returns:
            Response: Agent response with TTS output information
        """
        print(f"TTSAgent.run called with {len(messages)} messages")
        
        try:
            # Import the TTS tools
            from autoagent.tools.tts_tools import text_to_speech, text_to_speech_with_context
            
            # Get the latest user message
            user_message = messages[-1]["content"]
            print(f"Processing user message: {user_message[:50]}...")
            
            # Update conversation history
            self.conversation_history.append({
                "text": user_message,
                "speaker": 0  # User is always speaker 0
            })
            
            # Generate a unique filename
            import uuid
            output_filename = f"{uuid.uuid4()}.wav"
            output_path = os.path.join(self.output_dir, output_filename)
            print(f"Output path: {output_path}")
            
            # Check if we have enough context for contextual TTS
            if len(self.conversation_history) > 1:
                print(f"Using contextual TTS with {len(self.conversation_history)-1} previous messages")
                # Extract context texts and speakers
                context_texts = [item["text"] for item in self.conversation_history[:-1]]
                context_speakers = [item["speaker"] for item in self.conversation_history[:-1]]
                
                # Generate speech with context
                print("Calling text_to_speech_with_context...")
                result_path = text_to_speech_with_context(
                    text=user_message,
                    context_texts=context_texts,
                    context_speakers=context_speakers,
                    output_path=output_path,
                    speaker=self.speaker_id
                )
                print(f"text_to_speech_with_context returned: {result_path}")
            else:
                print("Using non-contextual TTS")
                # Generate speech without context
                print("Calling text_to_speech...")
                result_path = text_to_speech(
                    text=user_message,
                    output_path=output_path,
                    speaker=self.speaker_id
                )
                print(f"text_to_speech returned: {result_path}")
            
            # Add agent response to conversation history
            response_text = f"I've converted your text to speech. The audio file is saved at: {result_path}"
            self.conversation_history.append({
                "text": response_text,
                "speaker": 1  # Agent is always speaker 1
            })
            
            # Create response
            response = Response(
                messages=messages + [{"role": "assistant", "content": response_text}],
                context_variables=context_variables or {}
            )
            
            # Add the audio file path to the response context
            response.context_variables["audio_file"] = result_path
            print(f"Added audio_file to context_variables: {result_path}")
            
            return response
        except Exception as e:
            print(f"Error in TTSAgent.run: {e}")
            print(traceback.format_exc())
            
            # Create error response
            error_message = f"I encountered an error while trying to convert your text to speech: {str(e)}"
            response = Response(
                messages=messages + [{"role": "assistant", "content": error_message}],
                context_variables=context_variables or {}
            )
            return response 