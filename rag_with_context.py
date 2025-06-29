'''This script implements a hybrid RAG system using LlamaIndex and Groq LLM which has the ability to remember conversation context and stores the indexes in local PC'''

from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
import asyncio
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Validate API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is required")

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5", 
    cache_folder="./cache"
)
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=360.0,
)

# Try to load existing storage, otherwise create new indexes
storage_dir = "storage"
try:
    print("Trying to load existing indexes from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    vector_index = load_index_from_storage(storage_context, index_id="vector")
    keyword_index = load_index_from_storage(storage_context, index_id="keyword")
    print("Loaded existing indexes from storage.")
    
    # Still need to load documents for BM25 (not stored)
    documents = SimpleDirectoryReader("data").load_data()
    parser = SimpleNodeParser.from_defaults(
        chunk_size=1000,
        chunk_overlap=200,
    )
    nodes = parser.get_nodes_from_documents(documents)
    
except Exception as e:
    print(f"Could not load from storage ({e}), creating new indexes...")
    
    print("Loading documents...")
    try:
        documents = SimpleDirectoryReader("data").load_data()
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        raise

    # Parse documents into nodes for BM25 with proper chunking
    print("Parsing documents into nodes with chunking...")
    parser = SimpleNodeParser.from_defaults(
        chunk_size=1000,  # Smaller chunks for better retrieval
        chunk_overlap=200,  # Some overlap to maintain context
    )
    nodes = parser.get_nodes_from_documents(documents)

    print("Creating indexes...")
    # Create vector store index from nodes (with chunking)
    vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
    
    # Create keyword table index from nodes
    keyword_index = SimpleKeywordTableIndex(nodes)
    
    # Save indexes to storage
    print("Saving indexes to storage...")
    vector_index.set_index_id("vector")
    keyword_index.set_index_id("keyword")
    vector_index.storage_context.persist(persist_dir=storage_dir)
    keyword_index.storage_context.persist(persist_dir=storage_dir)

print("Creating retrievers...")
# Create retrievers with smaller top_k to reduce context size
vector_retriever = vector_index.as_retriever(similarity_top_k=3)
keyword_retriever = keyword_index.as_retriever(similarity_top_k=3)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

# Define a custom hybrid retriever class
class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self._retrievers = retrievers
        super().__init__()

    def _retrieve(self, query_bundle):
        # Retrieve results from each retriever
        all_results = []
        for retriever in self._retrievers:
            try:
                results = retriever.retrieve(query_bundle)
                all_results.extend(results)
            except Exception as e:
                print(f"Warning: Retriever failed: {e}")
                continue

        # Create a dictionary to store unique nodes and their highest scores
        unique_nodes = {}
        for res in all_results:
            node_id = res.node.node_id
            if node_id not in unique_nodes:
                unique_nodes[node_id] = res
            else:
                # If node already exists, update with the higher score
                if hasattr(res, 'score') and hasattr(unique_nodes[node_id], 'score'):
                    if res.score and (not unique_nodes[node_id].score or res.score > unique_nodes[node_id].score):
                        unique_nodes[node_id] = res
        
        # Return the unique nodes as a list
        return list(unique_nodes.values())

# Instantiate the hybrid retriever
hybrid_retriever = HybridRetriever([vector_retriever, keyword_retriever, bm25_retriever])

# Create hybrid query engine
hybrid_query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=Settings.llm,
)

print("Creating query engine...")

# Conversation context to maintain chat history
conversation_history = []

async def search_documents_with_context(query: str) -> str:
    """Search through documents using hybrid retrieval with conversation context
    
    Args:
        query (str): The search query to find relevant documents
        
    Returns:
        str: The response from the document search
    """
    try:
        # Build context-aware query
        if conversation_history:
            # Include recent conversation history for context
            recent_history = conversation_history[-4:]  # Last 4 exchanges
            context_prompt = f"""
Previous conversation:
{chr(10).join([f"User: {h['user']}" + chr(10) + f"Assistant: {h['assistant']}" for h in recent_history])}

Current question: {query}

Please answer the current question, considering the conversation context above.
"""
        else:
            context_prompt = query
            
        response = await hybrid_query_engine.aquery(context_prompt)
        
        # Add to conversation history
        conversation_history.append({
            "user": query,
            "assistant": str(response)
        })
        
        # Keep only last 10 exchanges to prevent context overflow
        if len(conversation_history) > 10:
            conversation_history.pop(0)
            
        return str(response)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

async def main():
    print("Document search ready! You can ask questions about documents.")
    print("The system will remember our conversation context.")
    
    while True:
        try:
            user_query = input("\nEnter your query (or 'quit' to exit): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("Processing...")
            
            # Use direct search with context
            result = await search_documents_with_context(user_query)
            print(f"\nResponse: {result}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            # Final fallback
            try:
                print("Trying fallback search...")
                # Try sync version as final fallback
                response = hybrid_query_engine.query(user_query)
                print(f"Fallback result: {response}")
            except Exception as fallback_error:
                print(f"All methods failed: {fallback_error}")

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())