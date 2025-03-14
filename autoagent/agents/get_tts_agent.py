from typing import Optional
from autoagent.agents.tts_agent import TTSAgent
from autoagent.registry import register_agent

@register_agent(name="get_tts_agent")
def get_tts_agent(output_dir: str = "tts_outputs", speaker_id: int = 0) -> TTSAgent:
    """
    Factory function to create a Text-to-Speech Agent instance.
    
    Args:
        output_dir: Directory to save audio files
        speaker_id: ID of the speaker to use for TTS (0 or 1)
        
    Returns:
        TTSAgent: An instance of the Text-to-Speech Agent
    """
    return TTSAgent(
        output_dir=output_dir,
        speaker_id=speaker_id,
    ) 