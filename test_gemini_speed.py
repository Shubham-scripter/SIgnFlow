import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

models = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

for model in models:

    print("\n" + "=" * 50)
    print("Testing:", model)

    start = time.time()

    try:
        response = client.models.generate_content(
            model=model,
            contents="Convert YOU NAME WHAT into one natural English sentence."
        )

        elapsed = time.time() - start

        print("Response:", response.text)
        print(f"Time: {elapsed:.2f} seconds")

    except Exception as e:

        elapsed = time.time() - start

        print("ERROR:", e)
        print(f"Time before error: {elapsed:.2f} seconds")