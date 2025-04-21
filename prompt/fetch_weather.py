import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_weather(city:str):
    print("🔨 Tool Called: get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if(response.status_code == 200):
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"

def run_command(command):
    result = os.system(command=command)
    return result

def add(x, y):
    print("🔨 Tool Called: add", x, y)
    return x + y

available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Take the city name as an input and return the current weather for the city"
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns ouput"
    }
}

system_prompt = """
You're an helpfull AI assistant who is specialized in resolving user query.
You work on start, plan, action and observe mode.

For the given user query and available tools, plan the step by step execution based on planning.
Select the relevant tool from  the available tool and based on the selected tool perform an action to call the tool.
Wait for the observation and based on observation from the tool call resolve the user query.

Rules:
1. Follow the output JSON format
2. Always perform one step at a time and wait for the next input
3. Carefully analyze the user query

Output JSON Format:
{{
    "step": "string",
    "content":"string",
    "function":The name of the function if the step is action",
    "input": "The input parameter of the function"
}}

Availabe tools:
- get_weather : Take a city name as an input and return the current weather for the city
- run_command: Takes a command as input to execute on system and returns ouput

Example:
Query : What is the weather of New York
output: {{"step": "plan", "content": "The user is interested in weather data of new york"}}
output: {{"step": "plan", "content": "From teh avalable s i should call get weather"}}
output: {{"step": "action", "function": "get_weather", "input": "new york"}}
output: {{"step": "observe", "output": "12 Degree Celcius"}}
output: {{"step": "output", "content": "The weather for new york seems to be 12 degrees."}}
"""

messages = [
    {"role":"system","content":system_prompt}
]

query=""
while (query != "bye"):
    query = input("> ")
    messages.append({"role":"user", "content": query})

    while True:
        result = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type":"json_object"},
            messages=messages
        )

        parsed_result = json.loads(result.choices[0].message.content)
        messages.append({"role":"assistant", "content": json.dumps(parsed_result)})

        if parsed_result.get("step") == "plan":
            print(f"👽: {parsed_result.get("content")}")
            continue

        if parsed_result.get("step") == "action":
            tool_name = parsed_result.get("function")
            tool_input = parsed_result.get("input")
            
            if available_tools.get(tool_name, False) != False:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append({"role":"assistant","content": json.dumps({"step": "observe", "output": output})})
                continue

        if parsed_result.get("step") == "output":
            print(f"🤖: {parsed_result.get("content")}")
            break
