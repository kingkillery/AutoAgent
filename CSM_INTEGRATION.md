# CSM-1B Text-to-Speech Integration

This document describes the integration of the CSM-1B text-to-speech model from Sesame AI into the AutoAgent framework.

## Overview

CSM (Conversational Speech Model) is a speech generation model from Sesame AI that generates high-quality, natural-sounding speech from text. The model architecture employs a Llama backbone and a smaller audio decoder that produces Mimi audio codes.

The integration includes:
- A TTS tool module (`autoagent/tools/tts_tools.py`) that provides functions for text-to-speech conversion
- A TTS agent (`autoagent/agents/tts_agent.py`) that uses the CSM-1B model
- A getter function (`autoagent/agents/get_tts_agent.py`) to easily create TTS agent instances
- A test script (`test_tts_agent.py`) to verify the integration

## Requirements

The integration requires the following dependencies:
- Python 3.10 or higher
- PyTorch 2.0.0 or higher
- torchaudio 2.0.0 or higher
- huggingface_hub
- The CSM repository from GitHub

These dependencies are automatically added to the `setup.cfg` file.

## Usage

### Basic Usage

```python
from autoagent import MetaChain
from autoagent.agents import get_tts_agent

# Initialize the TTS agent
tts_agent = get_tts_agent(output_dir="tts_outputs")

# Create a MetaChain instance
mc = MetaChain()

# Create a message
messages = [
    {"role": "user", "content": "Hello, this is a test of the CSM-1B text-to-speech model."}
]

# Run the agent
response = mc.run(tts_agent, messages)

# Get the audio file path
audio_file = response.context_storage.get("audio_file")
print(f"Audio file saved at: {audio_file}")
```

### Contextual TTS

The CSM-1B model supports contextual TTS, which means it can generate more natural-sounding speech when provided with conversation context. The TTS agent automatically maintains conversation history and uses it for contextual TTS.

```python
# Add a follow-up message
messages.append(response.messages[-1])
messages.append({"role": "user", "content": "This is a follow-up message."})

# Run the agent again
response = mc.run(tts_agent, messages)

# Get the audio file path
audio_file = response.context_storage.get("audio_file")
print(f"Audio file saved at: {audio_file}")
```

## Testing

To test the TTS agent, run the provided test script:

```bash
python test_tts_agent.py
```

This script will:
1. Initialize the TTS agent
2. Convert a sample text to speech
3. Test the contextual TTS capabilities with a follow-up message
4. Save the generated audio files to the `tts_test_outputs` directory

## Notes

- The CSM-1B model requires a GPU for efficient inference. The integration will automatically use CUDA if available.
- The model checkpoint is automatically downloaded from the Hugging Face Hub the first time you use the TTS agent.
- The CSM repository is automatically cloned if it doesn't exist.

## Ethical Considerations

As noted by Sesame AI, this model should be used responsibly. The following uses are explicitly prohibited:
- Impersonation or fraud
- Misinformation or deception
- Illegal or harmful activities

By using this integration, you agree to comply with all applicable laws and ethical guidelines. 