import os
from dotenv import load_dotenv
from google import genai
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents="Explain why sky is blue"
)
print(response.text)


#             *********  Zero Shot Prompting  ************