'''Complete RAG implementation using vector search for semantic retrieval and Groq LLM for answer generation.
The PDF is loaded, converted into embeddings using sentence transformer, and indexed with FAISS for fast vector search.
User query is converted to embeddings, similar documents are retrieved, and then Groq LLM generates a comprehensive answer.
This demonstrates the full RAG pipeline: Retrieval (Vector Search) + Augmentation (context) + Generation (Groq LLM).'''

import fitz  # PyMuPDF for PDF handling
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from groq import Groq
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    load_pdf = fitz.open(pdf_path)  # Load the PDF document
    documents = [page.get_text() for page in load_pdf]  # Extract text from each page
    return documents

pdf_path = "data/1 Unit 1-Metal Casting.pdf"
docs = extract_text_from_pdf(pdf_path)

# Initialize Groq LLM client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

load_embeddeing_model = SentenceTransformer('all-MiniLM-L6-v2')  #Loading of the embedding model
docs_embeddings = load_embeddeing_model.encode(docs) # Convert documents to embeddings

docs_embeddings = np.array(docs_embeddings).astype('float32')  # Converted to a numpy array for indexing
index = faiss.IndexFlatL2(docs_embeddings.shape[1])  # Create a FAISS index
index.add(docs_embeddings)  # Add document embeddings to the index

input = input("Enter a word for a vector search: ")
input_embedding = load_embeddeing_model.encode([input])  # Convert the input query to an embedding
input_embedding = np.array(input_embedding).astype('float32')  # Convert to numpy array for FAISS

N = 3 # The top N results to be returned

distance, indice = index.search(input_embedding, N)  # Perform the search
print("\n" + "="*80)
print("RETRIEVED CONTEXT (Top relevant documents):")
print("="*80)

# Collect retrieved context
retrieved_context = []
for i in range(N):
    doc_text = docs[indice[0][i]]
    retrieved_context.append(doc_text)
    print(f"\n[Document {i+1}] (Distance: {distance[0][i]:.4f})")
    print(f"{doc_text[:300]}...")  # Show first 300 characters

# Combine retrieved documents into context
context = "\n\n".join(retrieved_context)

# Create prompt for Groq LLM
prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context from the documents.

Context:
{context}

Question: {input}

Instructions:
- Provide a comprehensive and accurate answer based on the context provided
- If the context doesn't contain enough information, mention what's available and what's missing
- Be clear and concise in your response

Answer:"""

print("\n" + "="*80)
print("GENERATING ANSWER WITH GROQ LLM...")
print("="*80)

# Generate answer using Groq
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",  # Using Groq's Llama model
    temperature=0.3,
    max_tokens=1024,
)

# Extract and display the answer
answer = chat_completion.choices[0].message.content

print("\n" + "="*80)
print("FINAL ANSWER:")
print("="*80)
print(f"\n{answer}\n")
print("="*80)