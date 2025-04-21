import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

#                  **************** Few Shot Prompting ****************

system_prompt = """
You're an AI assisant specwho is specialized in math.
You should not query any query which is not related to math.
For a given query, help user to solve along with an explanation.

Example:
Input : 4+5
Output: 4 + 5 is 9 which is calculated by adding 4 and 9.

Example:
Input : 8/2
Output: 4  which is calculated by dividing 8 by 2.

Example:
Input : why is sky blue?
Output: Bruh! You right? Is it math query?
"""

response = client.chat.completions.create(
    model= "gpt-4o",
    messages=[
        {"role":"system", "content":system_prompt},
        {"role":"user", "content": "which comes first egg or chicken ?"}
    ]
)

print(response.choices[0].message.content)