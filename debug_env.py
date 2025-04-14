# from dotenv import load_dotenv
# import os

# # Load environment variables from .env file
# load_dotenv()

# # Check if HF_TOKEN is loaded
# hf_token = os.getenv("HF_TOKEN")

# if hf_token:
#     print("✅ HF_TOKEN Loaded Successfully!")
# else:
#     print("❌ ERROR: HF_TOKEN not found. Check .env file location and formatting.")

# # Debug: Print specific environment variables (for testing)
# print("🔍 Loaded Hugging Face API Token:", hf_token[:5] + "*****" if hf_token else "None")

# # Print all environment variables (for deeper debugging)
# print("🔎 All Env Vars:", dict(os.environ))



from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

api = HfApi()

try:
    user = api.whoami(hf_token)
    print("✅ Connected to Hugging Face API as:", user)
except Exception as e:
    print("❌ Failed to connect to Hugging Face API:", e)
