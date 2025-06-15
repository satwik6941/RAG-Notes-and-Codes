import fitz  # PyMuPDF for PDF handling
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def extract_text_from_pdf(pdf_path):
    load_pdf = fitz.open(pdf_path)  # Load the PDF document
    documents = [page.get_text() for page in load_pdf]  # Extract text from each page
    return documents

pdf_path = "metal_casting.pdf"
docs = extract_text_from_pdf(pdf_path)

load_embeddeing_model = SentenceTransformer('all-MiniLM-L6-v2')  #Loading of the embedding model
docs_embeddings = load_embeddeing_model.encode(docs) # Convert documents to embeddings

docs_embeddings = np.array(docs_embeddings).astype('float32')  # Converted to a numpy array for indexing
index = faiss.IndexFlatL2(docs_embeddings.shape[1])  # Create a FAISS index
index.add(docs_embeddings)  # Add document embeddings to the index

input = input("Enter a word for a vector search: ")
input_embedding = load_embeddeing_model.encode([input])  # Convert the input query to an embedding
input_embedding = np.array(input_embedding).astype('float32')  # Convert to numpy array for FAISS

N = 5 # The top N results to be returned

distance, indice = index.search(input_embedding, N)  # Perform the search
print("The top results are: ")
for i in range(N):
    print(f"Document: {docs[indice[0][i]]}")