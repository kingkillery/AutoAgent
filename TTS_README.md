# CSM-1B Text-to-Speech Integration for AutoAgent

This integration adds text-to-speech capabilities to AutoAgent using the CSM-1B model from Sesame AI.

## Prerequisites

1. Python 3.9+ with pip
2. A Hugging Face account with access to the Llama-3.2-1B model
3. Required Python packages (installed via `pip install -r csm/requirements.txt`)

## Setup Instructions

### 1. Request Access to Required Models

The CSM-1B model depends on the Llama-3.2-1B tokenizer, which is a gated model on Hugging Face. You need to:

1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B and request access
2. Wait for approval (this may take some time)
3. Once approved, authenticate with Hugging Face using your token

### 2. Authenticate with Hugging Face

Run the provided authentication script:

```bash
python huggingface_login.py
```

This will:
- Prompt you for your Hugging Face token
- Log you in to Hugging Face
- Optionally save your token to the `.env` file for future use

### 3. Test the TTS Agent

Run the test script to verify that the TTS agent is working correctly:

```bash
python test_tts_agent.py
```

This will:
- Initialize the TTS agent
- Convert a test message to speech
- Save the audio file to the `tts_test_outputs` directory

### 4. Play the Generated Audio

Use the provided audio player to listen to the generated audio files:

```bash
python play_audio.py
```

This will:
- List all WAV files in the `tts_test_outputs` directory
- Allow you to select and play any of the files

## Usage in Your Code

To use the TTS agent in your own code:

```python
from autoagent import MetaChain
from autoagent.agents import get_tts_agent

# Initialize the TTS agent
tts_agent = get_tts_agent(output_dir="your_output_dir")

# Create a MetaChain instance
mc = MetaChain()

# Define your messages
messages = [
    {"role": "user", "content": "Text to convert to speech"}
]

# Run the agent
response = mc.run(tts_agent, messages)

# Get the audio file path
audio_file = response.context_variables.get("audio_file")
print(f"Audio file saved at: {audio_file}")
```

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:
- Make sure you have requested and been granted access to the Llama-3.2-1B model
- Verify that you're using the correct Hugging Face token
- Try running `huggingface_login.py` again

### Missing Dependencies

If you encounter missing dependencies:
- Make sure you've installed all required packages: `pip install -r csm/requirements.txt`
- For Windows users, you may need to install additional dependencies for audio playback

### Audio Playback Issues

If you have trouble playing the generated audio:
- Make sure you have installed pygame: `pip install pygame`
- Check that the audio files exist in the output directory
- Try playing the audio files with another media player 