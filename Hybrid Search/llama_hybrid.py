'''This is a complete RAG implementation using LlamaIndex and groqcloud LLM. The database is 4 pdfs in the data directory.
It combines vector search, keyword search, and BM25 retrieval methods (Retrieval) and uses Groq LLM for answer generation (Generation).'''

import os
from llama_index.core import (
    VectorStoreIndex,           
    SimpleKeywordTableIndex,    
    SimpleDirectoryReader,      
    Settings,                   
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

print("Configuring LLM and embedding model...")
llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.environ.get("GROQ_API_KEY"))
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", cache_folder="./cache")

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model

print("Loading documents...")
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()
print(f"Loaded {len(documents)} documents.")

print("Creating vector store index...")
vector_index = VectorStoreIndex.from_documents(documents)

print("Creating keyword table index...")
keyword_index = SimpleKeywordTableIndex.from_documents(documents)

print("Creating retrievers...")
vector_retriever = vector_index.as_retriever(similarity_top_k=5)
keyword_retriever = keyword_index.as_retriever(similarity_top_k=5)

# FIX: Convert documents to nodes for BM25Retriever
nodes = list(vector_index.docstore.docs.values())
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

# Define a custom retriever class that combines results from multiple retrievers
class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self._retrievers = retrievers
        super().__init__()

    def _retrieve(self, query_bundle):
        all_results = []
        for retriever in self._retrievers:
            try:
                results = retriever.retrieve(query_bundle)
                all_results.extend(results)
            except Exception as e:
                print(f"Warning: Retriever failed with error: {e}")
                continue

        # Create a dictionary to store unique nodes and their highest scores
        unique_nodes = {}
        for res in all_results:
            node_id = res.node.node_id
            if node_id not in unique_nodes:
                unique_nodes[node_id] = res
            else:
                # Keep the result with higher score
                if res.score and res.score > (unique_nodes[node_id].score or 0):
                    unique_nodes[node_id] = res
        
        # Sort by score and return top results
        sorted_results = sorted(
            unique_nodes.values(), 
            key=lambda x: x.score or 0, 
            reverse=True
        )
        return sorted_results[:10]

# Instantiate the hybrid retriever
hybrid_retriever = HybridRetriever([vector_retriever, keyword_retriever, bm25_retriever])

print("Creating query engine with LLM for generation...")
# Add response_mode to ensure LLM generates comprehensive answers
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=llm,  # Explicitly pass the LLM
    response_mode="compact",  # LLM synthesizes answer from retrieved chunks
    verbose=True  # Shows retrieval and generation process
)

# Interactive query loop for complete RAG
while True:
    query = input("\nEnter your query (or 'quit' to exit): ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    try:
        print("\n--- RETRIEVAL PHASE ---")
        print("Retrieving relevant documents...")
        
        print("\n--- GENERATION PHASE ---")
        print("Generating answer using LLM...")
        response = query_engine.query(query)
        
        print(f"\n--- FINAL ANSWER ---")
        print(response)

    except Exception as e:
        print(f"Error processing query: {e}")
