import os
import sys
import traceback
from typing import List, Optional, Dict, Any, Union

# Define a Segment class for handling context in CSM model
class Segment:
    def __init__(self, text: str, speaker: int, audio=None):
        self.text = text
        self.speaker = speaker
        self.audio = audio

class CSMGenerator:
    """
    A wrapper class for the CSM-1B text-to-speech model from Sesame AI.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        try:
            print(f"Initializing CSMGenerator with model_path: {model_path}, device: {device}")
            # Import torch and torchaudio here to avoid circular imports
            import torch
            import torchaudio
            
            # Get the CSM directory path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            csm_dir = os.path.join(base_dir, "csm")
            print(f"CSM directory: {csm_dir}")
            
            # Add the CSM directory to the Python path
            if csm_dir not in sys.path:
                sys.path.append(csm_dir)
                print(f"Added {csm_dir} to sys.path")
                
            # Import the generator module
            print("Importing generator module...")
            from generator import load_csm_1b
            
            # Load the model
            print(f"Loading CSM-1B model from {model_path}...")
            self.generator = load_csm_1b(model_path, device)
            self.sample_rate = self.generator.sample_rate
            self.device = device
            self._initialized = True
            print(f"CSMGenerator initialized successfully. Sample rate: {self.sample_rate}")
        except Exception as e:
            print(f"Error initializing CSM Generator: {e}")
            print(traceback.format_exc())
            self._initialized = False
    
    def generate(self, text: str, speaker: int = 0, context: List[Segment] = None, max_audio_length_ms: int = 10000) -> Any:
        """
        Generate speech from text using the CSM-1B model.
        
        Args:
            text: The text to convert to speech
            speaker: Speaker ID (0 or 1)
            context: Optional list of Segment objects for context
            max_audio_length_ms: Maximum audio length in milliseconds
            
        Returns:
            torch.Tensor: Audio waveform
        """
        if not self._initialized:
            raise RuntimeError("CSM Generator not properly initialized")
            
        context = context or []
        print(f"Generating speech for text: {text[:50]}... with speaker: {speaker}, context length: {len(context)}")
        
        try:
            audio = self.generator.generate(
                text=text,
                speaker=speaker,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
            )
            print(f"Speech generated successfully. Audio shape: {audio.shape}")
            return audio
        except Exception as e:
            print(f"Error generating speech: {e}")
            print(traceback.format_exc())
            raise

def text_to_speech(text: str, output_path: str = "output.wav", speaker: int = 0) -> str:
    """
    Convert text to speech using the CSM-1B model and save to a file.
    
    Args:
        text: The text to convert to speech
        output_path: Path to save the audio file
        speaker: Speaker ID (0 or 1)
        
    Returns:
        str: Path to the saved audio file
    """
    try:
        print(f"text_to_speech called with text: {text[:50]}..., output_path: {output_path}, speaker: {speaker}")
        # Import dependencies inside function to avoid circular imports
        import torch
        import torchaudio
        from huggingface_hub import hf_hub_download
        
        # Download the model if it doesn't exist
        print("Downloading CSM-1B model from Hugging Face...")
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        print(f"Model downloaded to: {model_path}")
        
        # Initialize the generator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        generator = CSMGenerator(model_path, device)
        
        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=10000,
        )
        
        # Save the audio
        print(f"Saving audio to {output_path}...")
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Audio saved successfully to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        print(traceback.format_exc())
        return f"Error generating speech: {str(e)}"

def text_to_speech_with_context(
    text: str, 
    context_texts: List[str], 
    context_speakers: List[int], 
    output_path: str = "output.wav", 
    speaker: int = 0
) -> str:
    """
    Convert text to speech with conversation context using the CSM-1B model.
    
    Args:
        text: The text to convert to speech
        context_texts: List of previous utterances for context
        context_speakers: List of speaker IDs for context utterances
        output_path: Path to save the audio file
        speaker: Speaker ID for the current utterance
        
    Returns:
        str: Path to the saved audio file
    """
    try:
        print(f"text_to_speech_with_context called with text: {text[:50]}..., context_texts: {len(context_texts)}, output_path: {output_path}")
        # Import dependencies inside function to avoid circular imports
        import torch
        import torchaudio
        from huggingface_hub import hf_hub_download
        
        # Download the model if it doesn't exist
        print("Downloading CSM-1B model from Hugging Face...")
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        print(f"Model downloaded to: {model_path}")
        
        # Initialize the generator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        generator = CSMGenerator(model_path, device)
        
        # Create context segments
        segments = [
            Segment(text=t, speaker=s)
            for t, s in zip(context_texts, context_speakers)
        ]
        print(f"Created {len(segments)} context segments")
        
        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=segments,
            max_audio_length_ms=10000,
        )
        
        # Save the audio
        print(f"Saving audio to {output_path}...")
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Audio saved successfully to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error in text_to_speech_with_context: {e}")
        print(traceback.format_exc())
        return f"Error generating speech with context: {str(e)}" 