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
    collection_name="reciprocate_rank_fusion"
)

vector_store.add_documents(documents=split_docs)
print("Document is loaded in Vector DB")


# rank fusion function
def reciprocal_rank_fusion(rankings, k = 15):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Retrieval function
def retrieve(queries, k=15):
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    embedding = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    relevant_chunks = QdrantVectorStore.from_existing_collection(
        embedding=embedding,
        url="http://localhost:6333",
        collection_name="reciprocate_rank_fusion"
    )

    # 3) run each sub‑query, collect rankings of IDs + keep lookup
    rankings=[]
    lookup={}
    for query in queries:
        docs = relevant_chunks.similarity_search(query=query, k=k)
        ids=[]
        # assume each Doc has a unique metadata["id"]
        for d in docs:
            doc_id = d.metadata.get("id") or f"{d.metadata.get('page')}#{hash(d.page_content)}"
            ids.append(doc_id)
            lookup[doc_id] = d
        rankings.append(ids)

    # 4) fuse the ranked ID lists
    fused = reciprocal_rank_fusion(rankings)

    # 5) map fused IDs back to Doc objects, preserving order
    fused_docs=[]
    for doc_id, score in fused:
        if doc_id in lookup:
            fused_docs.append(lookup[doc_id])

    # 6) format into your final context string
    formatted=[]
    for doc in fused_docs:
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        formatted.append(f"[Page {page}]\n{text}")

    context = "\n\n".join(formatted)
    return context


# client = OpenAI(
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# Anser_AI Function
def answer_AI(query, assistant):
    system_prompt = """
    You are an helpfull AI Assistant who is specialized in resolving user query.

    Note:
    Answer should be in detail
    Response should be in well structured JSON format
    You recive a question and you give answer based on the assistant content and 
    also Mention the page number from where did you pick all the information and
    If you add something from you then tell where did you added something
    """
    message =[{"role":"system","content":system_prompt},{"role":"user","content":query},{"role":"assistant","content":assistant}]
    response=client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=message,
        response_format={"type":"json_object"}

    )

    return response.choices[0].message.content


client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

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
    {"role":"system", "content":system_prompt},
    {"role":"user", "content":query},
]

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    response_format={"type":"json_object"},
    messages=messages
)

questions = response.choices[0].message.content

print("\nQuestions :")
print(questions)

relevant_chunks = retrieve(questions)

output = answer_AI(query, relevant_chunks)

print("\n---------------------------")
print("Answers :---")
print(output)
