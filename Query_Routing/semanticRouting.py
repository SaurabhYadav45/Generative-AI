from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

system_prompts = {
    "overview": "You are an expert assistant. Give a brief overview of the given topic.",
    "detailed": "You are a subject matter expert. Explain the topic in detailed, technical terms.",
    "simple": "You are a friendly teacher. Explain the topic in the simplest possible way for a beginner.",
    "summary": "You are a helpful assistant. Summarize the given content in concise bullet points."
}

def semantic_routing(query):
    query = query.lower()
    if any(word in query for word in ["overview", "intro", "introduction"]):
        return "overview"
    elif any(word in query for word in ["detailed", "in-depth", "technically"]):
        return "detailed"
    elif any(word in query for word in ["explain simply", "in simple terms", "easy explanation", "like a child"]):
        return "simple"
    elif any(word in query for word in ["summary", "summarize", "tl;dr"]):
        return "summary"
    else:
        return "simple"

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

query = input("Ask Your Query: ")
selected_prompts = semantic_routing(query)

messages = [
    {"role": "user", "content": query},
    {"role":"system", "content": selected_prompts}
]

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=messages
)

print(response.choices[0].message.content)
