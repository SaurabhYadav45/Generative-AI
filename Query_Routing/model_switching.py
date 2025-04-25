import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Setup clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# === TASK TYPE DETECTION ===
def get_task_type(query):
    q = query.lower()
    if any(word in q for word in ["solve", "calculate", "math", "probability", "integral", "equation"]):
        return "math"
    elif any(word in q for word in ["code", "program", "debug", "bug", "python", "javascript"]):
        return "code"
    elif any(word in q for word in ["explain", "reason", "why", "how does", "logic", "compare"]):
        return "reasoning"
    elif any(word in q for word in ["paper", "research", "study", "analyze", "findings"]):
        return "research"
    else:
        return "general"

# === MODEL MAP ===
model_map = {
    "math": "claude-3-opus",
    "code": "deepseek-coder",
    "reasoning": "gpt-4o",
    "research": "claude-3-opus",
    "general": "gpt-4o"
}

# === LLM CALLERS ===

def call_openai(model, user_query):
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def call_claude(user_query):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": user_query}]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["content"][0]["text"]

def call_deepseek(user_query):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-coder",
        "messages": [{"role": "user", "content": user_query}]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# === MAIN ROUTING LOGIC ===

query = "Can you write a Python function to check if a number is prime?"

task_type = get_task_type(query)
selected_model = model_map[task_type]

print(f" Task type: {task_type} → Model selected: {selected_model}")

# Call the appropriate model
if selected_model == "gpt-4o":
    result = call_openai("gpt-4o", query)
elif selected_model == "claude-3-opus":
    result = call_claude(query)
elif selected_model == "deepseek-coder":
    result = call_deepseek(query)
else:
    result = "Model not supported or missing API setup."

print("\n Response:\n", result)
