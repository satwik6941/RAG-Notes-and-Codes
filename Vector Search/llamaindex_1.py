'''This is a vector search implementation using LlamaIndex, which is a framework for RAG applications to build with LLMs. 
We will create a folder with some PDFs (my college study material) and the LLM we will use a Local running LLM "Llama 3.1 8B" from Ollama'''

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
import asyncio
import os

huggingface_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")  # Load the embedding model
Ollama_LLM = Ollama(
    model="llama3.1:8B",
    request_timeout=360,
    context_window=8000
    )  # Load the LLM from Ollama

docs = SimpleDirectoryReader("data").load_data()  # Load documents from the "data" directory
index = VectorStoreIndex.from_documents(docs, embed_model=huggingface_model)    #Convert the documents into a vector store index (Automatically embeddings are created in the backend thanks to llamaindex)
query_engine = index.as_query_engine(llm = Ollama_LLM)   # Create a query engine from the index (Automatically embeddings are created in the backend thanks to llamaindex)

def multiply(a, b):  # Some basic arithmetic operations
    return a * b

def add(a, b):  # Some basic arithmetic operations
    return a + b

async def search_docs(query):   # Function to search documents using the query engine
    response = await query_engine.aquery(query)
    return str(response)

agent = AgentWorkflow.from_tools_or_functions(          #We create an agent workflow integrated with the LLM and the tools (multiply, add, search_docs) and this LLM does the vector search
    [multiply, add, search_docs],
    llm = Ollama_LLM,
    system_prompt="You are a helpful assistant that can perform vector search on documents and also help in basic arthmetic operations like addition and multiplication. You can also search for documents based on a query.",
)

async def main():
    while True:
        user_input = input("Enter a query or 'exit' to quit: ")     # User input
        if user_input.lower() == 'exit':
            break
        response = await agent.run(user_input)          #Search and display the result
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())  # Run the main function to start the agent workflow