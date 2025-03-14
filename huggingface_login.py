from huggingface_hub import login
import os

def main():
    print("Hugging Face Login Helper")
    print("=========================")
    print("This script will help you authenticate with Hugging Face to access gated models.")
    print("You need to have a Hugging Face account and have requested access to the Llama-3.2-1B model.")
    print("Visit https://huggingface.co/meta-llama/Llama-3.2-1B to request access if you haven't already.")
    print()
    
    # Check if token is already set in environment
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"Hugging Face token found in environment variable HF_TOKEN")
        use_env = input("Use this token? (y/n): ").lower() == 'y'
        if use_env:
            login(token=token)
            print("Logged in successfully with token from environment variable.")
            return
    
    # Ask for token
    print("Please enter your Hugging Face token.")
    print("You can find your token at https://huggingface.co/settings/tokens")
    token = input("Token: ")
    
    if not token:
        print("No token provided. Exiting.")
        return
    
    # Login with token
    login(token=token)
    print("Logged in successfully!")
    
    # Ask if user wants to save token to .env file
    save_to_env = input("Would you like to save your token to the .env file? (y/n): ").lower() == 'y'
    if save_to_env:
        env_path = ".env"
        
        # Read existing .env file if it exists
        env_vars = {}
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        env_vars[key] = value
        
        # Add or update HF_TOKEN
        env_vars["HF_TOKEN"] = token
        
        # Write back to .env file
        with open(env_path, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"Token saved to {env_path}")
    
    print("\nYou should now be able to access gated models on Hugging Face.")
    print("Try running the test_tts_agent.py script again.")

if __name__ == "__main__":
    main() 