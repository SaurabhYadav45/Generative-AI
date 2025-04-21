from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = openai_api_key)

#                     ************* Zero Shot Prompting ************

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "what is greater 9.8 or 9.11"}
    ]
)

print(response.choices[0].message.content)
