from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the HF_TOKEN environment variable
hf_token = os.getenv("HF_TOKEN")

# Use the HF_TOKEN
if hf_token:
    print("HF_TOKEN found:", hf_token)
    # Your code that uses the HF_TOKEN goes here...
else:
    print("HF_TOKEN not found in .env file.")