
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv(override=True)

api_key = os.getenv("NVIDIA_API_KEY", "").strip()

if not api_key:
    print("API key not found!")
    exit()

print("API key loaded:", bool(api_key))

try:
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=100,
    )

    response = llm.invoke("Reply only: API is working")

    print("Success!")
    print(response.content)

except Exception as e:
    print("API Error:")
    print(e)
