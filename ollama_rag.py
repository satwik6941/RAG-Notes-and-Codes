'''This is an implementation of a hybrid search system using LlamaIndex, Ollama LLM and Ollama Embedding Model. The database is 4 pdfs in the data directory.
It combines vector search, keyword search, and BM25 retrieval methods to provide a comprehensive search experience.

The main objective of this code is to demonstrate a hybrid RAG where the document processing to the LLM is "Gemma3:4b" and the embedding model is "embeddinggemma:300m", which is launched by Google DeepMind recently is completely executed in the "Local Environment".
'''

import os
from llama_index.core import (
    VectorStoreIndex,           
    SimpleKeywordTableIndex,    
    SimpleDirectoryReader,      
    Settings,                   
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Ollama LLM and Ollama Embedding Model configuration
print("Configuring LLM and embedding model...")
llm = Ollama(model="gemma3:4b", request_timeout=3000)  # Updated model name
embed_model = OllamaEmbedding(model_name="embeddinggemma:300m", request_timeout=3000)  # Better embedding model

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

# Fix: Convert documents to nodes for BM25Retriever
nodes = vector_index.docstore.docs.values()
bm25_retriever = BM25Retriever.from_defaults(nodes=list(nodes), similarity_top_k=5)

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
        
        # Sort by score (descending) and return top results
        sorted_results = sorted(unique_nodes.values(), 
                              key=lambda x: x.score or 0, 
                              reverse=True)
        return sorted_results[:10]  # Return top 10 results

# Instantiate the hybrid retriever
hybrid_retriever = HybridRetriever([vector_retriever, keyword_retriever, bm25_retriever])

print("Creating query engine...")
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    response_mode="compact"  # Better response formatting
)

# Interactive query loop
while True:
    query = input("\nEnter your query (or 'quit' to exit): ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    try:
        response = query_engine.query(query)
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"Error processing query: {e}")
