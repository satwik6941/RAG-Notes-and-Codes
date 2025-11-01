'''Complete RAG implementation using BM25 for keyword-based retrieval and Groq LLM for answer generation.
I made a sample dataset of documents in a list and a user input query.
The text is preprocessed, BM25 ranks documents, and then Groq LLM generates a comprehensive answer based on the retrieved context.
This demonstrates the full RAG pipeline: Retrieval (BM25) + Augmentation (context) + Generation (Groq LLM).'''

from rank_bm25 import BM25Okapi
import re
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sample documents for the keyword search
documents = [
    'The mould, into which the metal is poured, is made of some heat resisting material', 
    'Sand is most often used as it resists the high temperature of the molten metal. Permanent moulds of metal can also be used to cast products.',
    'The metal casting industry plays a key role in all the major sectors of our economy.',
    'There are castings in locomotives, cars, trucks, aircraft, office buildings, factories, schools, and homes.'
]

# Initialize Groq LLM client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Input query from the user
query = input("Enter your search query: ")

# Preprocessing function to clean and normalize text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()  # BM25 requires tokenized text (list of words)

# Preprocess documents and query
preprocessed_docs = [preprocess_text(doc) for doc in documents]
preprocessed_query = preprocess_text(query)

# print(preprocessed_docs)
# print(preprocessed_query)

# Initialize BM25
bm25 = BM25Okapi(preprocessed_docs)

# Get BM25 scores for the query
scores = bm25.get_scores(preprocessed_query)
print(f"BM25 scores: {scores}")

# Get document rankings based on scores
ranks = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
print(f"Ranks of the statements in the documents: {ranks}")

print("\n--- RETRIEVAL PHASE ---")
print(f"This is our query: {' '.join(preprocessed_query)} and below are the documents ranked based on their relevance to the query:")
for i, rank in enumerate(ranks):
    print(f"Rank {i+1} (Doc {rank}): {' '.join(preprocessed_docs[rank])} (Score: {scores[rank]:.4f})")

# --- AUGMENTATION & GENERATION PHASE ---
print("\n--- GENERATION PHASE ---")
print("Generating answer using Groq LLM based on retrieved context...")

# Get top 3 most relevant documents as context
top_k = 3
retrieved_context = "\n\n".join([documents[rank] for rank in ranks[:top_k]])

# Create prompt with retrieved context
prompt = f"""You are a helpful AI assistant. Answer the user's question based on the following context.

Context:
{retrieved_context}

Question: {query}

Answer: Provide a clear and concise answer based on the context above. If the context doesn't contain relevant information, say so."""

# Call Groq LLM for generation
try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # You can also use "mixtral-8x7b-32768" or other models
        temperature=0.3,
        max_tokens=500
    )
    
    generated_answer = chat_completion.choices[0].message.content
    
    print("\n--- FINAL ANSWER (RAG) ---")
    print(generated_answer)
    
    print("\n--- SOURCES USED ---")
    for i, rank in enumerate(ranks[:top_k], 1):
        print(f"Source {i}: {documents[rank][:100]}...")
        
except Exception as e:
    print(f"Error generating answer: {e}")
    print("Make sure you have set GROQ_API_KEY in your .env file")
