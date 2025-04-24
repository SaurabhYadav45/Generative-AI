from pathlib import Path
import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
# from qdrant_client import QdrantClient

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# pdf_path  = Path(__file__).parent/"nodejs.pdf"

# loader = PyPDFLoader(pdf_path)
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# split_docs = text_splitter.split_documents(documents=docs)

# open_api_key = os.getenv("OPENAI_API_KEY")

# embeddings = OpenAIEmbeddings(
#   model="text-embedding-3-large",
#   api_key=open_api_key
#   )


gemini_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_key

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004"
# )

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     embedding=embeddings,
#     url="http://localhost:6333",
#     collection_name="parallel_query",
# )

# vector_store.add_documents(documents=split_docs)
print("Document is loaded in Vector DB")

# Creating a retrival function

def retrieve(query: str)->str:
    
    os.environ["GOOGLE_API_KEY"] = gemini_key
    
    embedding = GoogleGenerativeAIEmbeddings(
       model="models/text-embedding-004"
    )

    retriever = QdrantVectorStore.from_existing_collection(
       collection_name="parallel_query",
       embedding=embedding,
       url="http://localhost:6333"
    )

    relevant_chunk = retriever.similarity_search(
       query=query
    )

    set_ = set() 
    unique = []

    for doc in relevant_chunk:
       page = doc.metadata.get('page')
       content = doc.page_content.strip()
       key = (page, content)
       if key not in set_:
          set_.add(key)
          unique.append(doc)

    formatted = []
    for doc in unique:
       snippet = f"[Page {doc.metadata.get('page')}]\n {doc.page_content}"
       formatted.append(snippet)

    context = "\n\n".join(formatted)
    return context


# Funtion to get final answer
def answer_AI(query, assistant):
    
    client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    system_prompt = """
    You are an helpfull AI Assistant who is specialized in resolving user query.

    Note:
    Response should be in well structured JSON format
    Answer should be in detail .
    You recive a question and you give answer based on the assistant content and 
    also Mention the page number from where did you pick all the information and
    If you add something from you then tell where did you added something
    """
    message =[
      {"role":"system","content":system_prompt},
      {"role":"user","content":query},
      {"role":"assistant","content":assistant}
    ]
    response=client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=message,
        response_format={"type":"json_object"}

    )

    return response.choices[0].message.content


gemini_api_key = os.getenv("GEMINI_API_KEY")
client = OpenAI(
   api_key=gemini_api_key,
   base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def ai(messages):
    response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=messages,
    response_format={"type":"json_object"}
    )
    # print(response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)

system_prompt = """
You're an helpful AI assistant, who is specialized in solving user query.
You break the user query into three or five different query.

Rules:
- Return ONLY a JSON object with questions as values and unique keys.
- Do not include any explanation or extra text.
- Ensure the output is strictly valid JSON.

Example:
Question: "What is FS module?"
You Break this question into different question.
Output:
{
  "question_1": "What does 'fs' stand for?",
  "question_2": "What is a module in Node.js?",
  "question_3": "What functionalities does the fs module provide in Node.js?"
}

"""

query = input("> ")
messages = [
   {"role":"user", "content":query},
   {"role":"system", "content": system_prompt}
]

# response = client.chat.completions.create(
#    model="gemini-2.0-flash",
#    response_format={"type":"json_object"},
#    messages=messages,
# )

questions = ai(messages)

print("\nQuestions: ")
print(questions)

array = []
for key, que in questions.items():
   answer = retrieve(que)
   array.append(answer)

# print("Array :", array)


output = answer_AI(query, json.dumps(array))

print("\n---------------------------")
print("Answer: ")
print(output)