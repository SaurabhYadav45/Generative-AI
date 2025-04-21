import os 
import json
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

system_prompt = """
You're an AI assistant who is expert in breaking down complex problem and then resolve the user query.

For the given user input analyse the problem and break down the problem step by step.
Atleast think of 5 to 6 step on how to solve the problem before solving it down.
The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait dor next input.
3. Carefully analyze the user query.

Output Format :
{{step : "string", content: "string"}}

Example:
Input: What is 2 + 2.
Output: {{ step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }}
Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}
"""

result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type":"json_object"},
    messages=[
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":"what is 3+4*5"},

        {"role":"assistant", "content":json.dumps({ "step": "analyse", "content": "The user is asking a mathematical question that involves both addition and multiplication. This appears to be an order of operations problem." })},

        {"role":"assistant", "content":json.dumps({"step": "think", "content": "I need to consider the order of operations, often remembered by the acronym PEMDAS (Parentheses, Exponents, Multiplication and Division, Addition and Subtraction). According to this, multiplication comes before addition."})},

        {"role":"assistant", "content":json.dumps({"step": "think", "content": "To solve 3 + 4 * 5, I should first multiply 4 and 5, and then add the result to 3."})},

        {"role":"assistant", "content":json.dumps({"step": "output", "content": "First, calculate 4 * 5 which equals 20, then add 3 to 20, resulting in 23."})},

        {"role":"assistant", "content":json.dumps({"step": "validate", "content": "I'm verifying the mathematical operations: 4 * 5 is indeed 20, and adding 3 gives 23. No parentheses or alternate operations are present, so this follows PEMDAS correctly."})},
    ]
)

print(result.choices[0].message.content)