import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

env_path = Path(__file__).resolve().parent.parent/".env"
load_dotenv(dotenv_path=env_path)

file_path = Path(__file__).parent/"nodejs.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_docs = text_splitter.split_documents(documents=docs)

gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    embedding=embedding,
    url="http://localhost:6333",
    collection_name="step_back_prompting"
)

vector_store.add_documents(documents=split_docs)
print("Document is loaded in Vector DB")


#                ************** Retrive Function ************

def retrieve(query)->str:
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    retrieved = QdrantVectorStore.from_existing_collection(
        collection_name="step_back_prompting",
        url="http://localhost:6333",
        embedding=embedding
    )

    relevant_chunk = retrieved.similarity_search(
        query=query
    )

    formatted = []
    for doc in relevant_chunk:
        snippet = f"[Page {doc.metadata.get('page')}]\n{doc.page_content}"
        formatted.append(snippet)

    context = "\n\n".join(formatted)
    return context


#                *************** Answer_AI Function ***********

def answer_AI(query, assistant):

    client = OpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    system_Prompt = """
    You're an helpful AI assistant who is specialized in resolving user query.

    "You're an AI that answer questions based on the retrieved knowledge. Mention page numbers for each fact."

    Note:
        -Answer should be in detail.
        -You recive a question and you give answer based on the assistant content and 
        also Mention the page number from where did you pick all the information.
        -If you add something from you then tell where did you added something.
    """

    message =[
        {"role":"system","content":system_Prompt},
        {"role":"user","content":query},
        {"role":"assistant","content":assistant}
    ]
    response=client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=message,
        response_format={"type":"json_object"}
    )
    return response.choices[0].message.content


#                  ************* Main Function **************

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)
system_prompt = f"""
You are an expert at node js knowledge. 
Your task is to step back and paraphrase a question to a more generic 
step-back question, which is easier to answer. 

RESPONSE FORMAT (Strictly follow this structure):
User Question: <repeat the original question here>
<write the generalized, step-back question here>

Here are a few examples:
Original Question: Which version of Node.js introduced the `fs.promises` API?
Stepback Question: What are the key features introduced in different versions of Node.js?

Original Question: How do you handle streams in the `fs` module for large file uploads?
Stepback Question: How does the `fs` module handle file streams in Node.js?

Original Question: Which core module in Node.js supports HTTP server creation?
Stepback Question: What are the core modules provided by Node.js and their functionalities?

Original Question: How does the `cluster` module help improve performance on multi-core systems?
Stepback Question: What is the purpose of the `cluster` module in Node.js?

Response Formate should be like this
User Question: Which core module in Node.js supports HTTP server creation?
Output: What are the core modules provided by Node.js and their functionalities? 
"""
query = input("> ")
message=[{"role":"system","content":system_prompt},{"role":"user","content":query}]

response=client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=message,
        response_format={"type":"json_object"}
    )
question = response.choices[0].message.content

print("Step Back Question: ")
print(question)

relevant_chunk = retrieve(question)

output = answer_AI(query, relevant_chunk)

print("\n------------------")
print("Answer: ")
print(output)
