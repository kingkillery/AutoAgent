import os
import sys
import torch
from autoagent import MetaChain
from autoagent.agents import get_tts_agent

def test_tts_agent():
    """
    Test the TTS agent with a simple text input.
    """
    print("Testing TTS Agent with CSM-1B model...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Create output directory
    output_dir = "tts_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {os.path.abspath(output_dir)}")
    
    # Initialize the TTS agent
    print("Initializing TTS agent...")
    tts_agent = get_tts_agent(output_dir=output_dir)
    print(f"TTS agent initialized: {tts_agent}")
    
    # Create a MetaChain instance
    mc = MetaChain()
    
    # Test with a simple message
    messages = [
        {"role": "user", "content": "Hello, this is a test of the CSM-1B text-to-speech model. It should convert this text to natural-sounding speech."}
    ]
    
    # Run the agent
    print("\nConverting text to speech...")
    try:
        # Add direct call to TTS function for debugging
        from autoagent.tools.tts_tools import text_to_speech
        debug_output = text_to_speech(
            text=messages[0]["content"],
            output_path=os.path.join(output_dir, "debug_test.wav"),
            speaker=0
        )
        print(f"Debug TTS output: {debug_output}")
    except Exception as e:
        print(f"Debug TTS error: {e}")
    
    # Now run through the agent
    response = mc.run(tts_agent, messages)
    
    # Print the response
    print("\nAgent Response:")
    print(response.messages[-1]["content"])
    
    # Print the audio file path
    audio_file = response.context_variables.get("audio_file")
    if audio_file:
        print(f"\nAudio file saved at: {audio_file}")
        print(f"Absolute path: {os.path.abspath(audio_file)}")
        
        # Check if the file exists
        if os.path.exists(audio_file):
            print(f"File exists and is {os.path.getsize(audio_file)} bytes")
        else:
            print(f"File does not exist at {audio_file}")
    else:
        print("\nNo audio file was generated!")
        print("Response context variables:", response.context_variables)
    
    # Test with a follow-up message to test context
    print("\nTesting with a follow-up message to test contextual TTS...")
    messages.append(response.messages[-1])
    messages.append({"role": "user", "content": "This is a follow-up message to test the contextual capabilities of the CSM model. It should sound more natural with context."})
    
    # Run the agent again
    response = mc.run(tts_agent, messages)
    
    # Print the response
    print("\nAgent Response (with context):")
    print(response.messages[-1]["content"])
    
    # Print the audio file path
    audio_file = response.context_variables.get("audio_file")
    if audio_file:
        print(f"\nAudio file saved at: {audio_file}")
        print(f"Absolute path: {os.path.abspath(audio_file)}")
        
        # Check if the file exists
        if os.path.exists(audio_file):
            print(f"File exists and is {os.path.getsize(audio_file)} bytes")
        else:
            print(f"File does not exist at {audio_file}")
    else:
        print("\nNo audio file was generated!")
        print("Response context variables:", response.context_variables)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_tts_agent() 