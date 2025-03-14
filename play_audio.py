import os
import sys
import glob
import pygame
from pygame import mixer

def list_audio_files(directory):
    """List all WAV files in the specified directory."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    return wav_files

def play_audio_file(file_path):
    """Play the specified audio file."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return False
    
    try:
        # Initialize pygame mixer
        pygame.init()
        mixer.init()
        
        # Load and play the sound
        sound = mixer.Sound(file_path)
        print(f"Playing {file_path}...")
        print(f"Duration: {sound.get_length():.2f} seconds")
        sound.play()
        
        # Wait for the sound to finish playing
        pygame.time.wait(int(sound.get_length() * 1000))
        
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False
    finally:
        # Clean up
        mixer.quit()
        pygame.quit()

def main():
    print("Audio Player for TTS Files")
    print("==========================")
    
    # Default directory
    default_dir = "tts_test_outputs"
    
    # Check if directory is provided as command line argument
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = default_dir
    
    # List audio files
    audio_files = list_audio_files(directory)
    
    if not audio_files:
        print(f"No audio files found in {directory}.")
        return
    
    print(f"Found {len(audio_files)} audio files in {directory}:")
    for i, file in enumerate(audio_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Ask which file to play
    while True:
        choice = input("\nEnter the number of the file to play (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            break
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(audio_files):
                play_audio_file(audio_files[index])
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

if __name__ == "__main__":
    main() 