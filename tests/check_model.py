import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

try:
    models = client.models.list()
    print("Available Models on your Key:")
    for model in models.data:
        print(f" - {model.id}")
except Exception as e:
    print(f"Failed to fetch models: {e}")