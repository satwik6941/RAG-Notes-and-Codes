import os
from llama_index.core import (
    VectorStoreIndex,           # For creating a vector index
    SimpleKeywordTableIndex,    # For creating a keyword index
    SimpleDirectoryReader,      # For reading documents from a directory
    Settings,                   # To configure LlamaIndex settings
)
from llama_index.retrievers.bm25 import BM25Retriever # For sparse retrieval
from llama_index.core.query_engine import RetrieverQueryEngine # To create a query engine from a retriever
from llama_index.core.postprocessor import SentenceTransformerRerank # For reranking results
from llama_index.llms.groq import Groq # The Groq LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever # HuggingFace embedding model

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

print("Configuring LLM and embedding model...")
Settings.llm = Groq(model="llama3-8b-8192", api_key=os.environ.get("GROQ_API_KEY"))
# Set up a HuggingFace embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2", cache_folder="./cache")

print("Loading documents...")
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()
print(f"Loaded {len(documents)} documents.")

print("Creating vector store index...")
vector_index = VectorStoreIndex.from_documents(documents)

print("Creating keyword table index...")
keyword_index = SimpleKeywordTableIndex.from_documents(documents)   # Create a keyword table index from the documents

print("Creating retrievers...")
vector_retriever = vector_index.as_retriever(similarity_top_k=5)    # Create a retriever for the vector index

keyword_retriever = keyword_index.as_retriever(similarity_top_k=5)  # Create a retriever for the keyword index

bm25_retriever = BM25Retriever.from_defaults(nodes=documents, similarity_top_k=5)   # BM25 is a ranking function based on term frequency and inverse document frequency

# Define a custom retriever class that combines results from multiple retrievers
class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers):
        # Initialize with a list of retrievers
        self._retrievers = retrievers
        super().__init__()

    def _retrieve(self, query_bundle):
        # Retrieve results from each retriever
        all_results = []
        for retriever in self._retrievers:
            all_results.extend(retriever.retrieve(query_bundle))

        # Create a dictionary to store unique nodes and their highest scores
        unique_nodes = {}
        for res in all_results:
            if res.node.node_id not in unique_nodes:
                unique_nodes[res.node.node_id] = res
            else:
                # If node already exists, update with the higher score
                if res.score > unique_nodes[res.node.node_id].score:
                    unique_nodes[res.node.node_id] = res
        
        # Return the unique nodes as a list
        return list(unique_nodes.values())

# Instantiate the custom hybrid retriever with the vector and keyword retrievers
hybrid_retriever = HybridRetriever([vector_retriever, keyword_retriever, bm25_retriever])

# --- 7. Create Query Engine ---

print("Creating query engine...")
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,         # Create a query engine that uses the hybrid retriever and reranker
)

query = input("Enter your query: ")
response = query_engine.query(query)
print(response)
