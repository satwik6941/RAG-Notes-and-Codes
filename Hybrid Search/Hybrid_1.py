'''This is the implementation of a hybrid search system that combines keyword search and vector search. For keyword search, it uses BM25, and for vector search, it uses a chroma vector store with embeddings.
The system loads a PDF document, extracts text, chunks it, and then performs searches based on user queries. The results from both searches are combined to provide a comprehensive answer'''

import fitz
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.schema import Document
import os
import dotenv as env

env.load_dotenv()  # Load environment variables from .env file

def extract_text_from_pdf(pdf_path):
    load_pdf = fitz.open(pdf_path)  # Load the PDF document
    documents = [page.get_text() for page in load_pdf]  # Extract text from each page
    return documents

pdf_path = "metal_casting.pdf"
docs = extract_text_from_pdf(pdf_path)

documents = [Document(page_content=page) for page in docs]      # Loading the documents

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)      # Chunking and loading the documents

tokens = [doc.page_content for doc in chunks]  # Extracting tokens from the chunks
bm25 = BM25Okapi(tokens)  # Initializing BM25 with the tokens

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Loading the embedding model
vector_store = Chroma.from_documents(chunks, embedding_model)  # Creating a vector store from the chunks

def hybrid_search(query, top_k=5):
    # Performs the keyword search using BM25 and retrieves the top_k results
    bm25_scores = bm25.get_scores(query.split())
    bm25_results = sorted(zip(chunks, bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
    bm25_docs = [doc[0] for doc in bm25_results]

    vector_docs = vector_store.similarity_search(query, k=top_k)    #Performs vector search using the vector store

    all_docs = list({doc.page_content: doc for doc in bm25_docs + vector_docs}.values()) # Combine and removes duplicate results from both searches
    
    return all_docs

llm = ChatGroq(
    model="gemma2-9b-it",  # or "llama3-8b-8192", etc.
    api_key = os.getenv('GROQ_API_KEY')  # or set it as an environment variable
)

def generate_answer(query):     # Generates an answer to the query using the hybrid search results
    docs = hybrid_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"
    return llm([HumanMessage(content=prompt)]).content

query = input("Enter your question: ")
print(generate_answer(query))  # Generate and print the answer based on the query