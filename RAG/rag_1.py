from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path  = Path(__file__).parent.parent/"nodejs.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
# print(docs[0])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents=docs)
# print(split_document[25])
# print("Docs", len(docs))
# print("Chunks", len(split_docs))

my_api_key = os.getenv("OPENAI_API_KEY")

embedder = OpenAIEmbeddings(
  model="text-embedding-3-large",
  api_key=my_api_key
  )

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    embedding = embedder,
    url="http://localhost:6333",
    collection_name="my_documents",
)

vector_store.add_documents(documents=split_docs)
print("Injection Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="my_documents",
    embedding=embedder,
)


search_result = retriver.similarity_search(
    query="What is FS Module?"
)

print("Relevant Chunks", search_result)