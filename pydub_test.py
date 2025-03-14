"""
Helper script to test pydub imports and warning suppression
"""
import os
# Configure environment variables
os.environ["PYDUB_USE_FFMPEG"] = "False"
os.environ["PYDUB_NO_FFMPEG"] = "True"

import warnings
# Configure warning filters
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# Standard imports after warning configuration
import sys

print("Python version:", sys.version)
print("Environment variables:")
print(f"  PYDUB_USE_FFMPEG: {os.environ.get('PYDUB_USE_FFMPEG', 'Not set')}")
print(f"  PYDUB_NO_FFMPEG: {os.environ.get('PYDUB_NO_FFMPEG', 'Not set')}")

try:
    import pydub
    print("\nSuccessfully imported pydub")
    print(f"Pydub version: {pydub.__version__}")
    print(f"Pydub path: {pydub.__file__}")
except Exception as e:
    print(f"\nError importing pydub: {e}")

print("\nTest complete. If you see no ffmpeg warnings above, suppression is working!")