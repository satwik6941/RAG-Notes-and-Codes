'''Implementation of a basic keyword search using BM25 algorithm. I made a sample dataset of documents extracted from PDF and a user input query.
The text is extracted from the pdf and then the text is preprocessed such that symbols are removed and everything is converted to lowercase. Then we use BM25 to rank documents based on their relevance to the query.
BM25 is a probabilistic ranking function that considers term frequency, document frequency, and document length normalization.'''

from rank_bm25 import BM25Okapi
import re
import fitz

load_pdf = fitz.open("metal_casting.pdf")  # Load the PDF document

documents = [page.get_text() for page in load_pdf]  # Extract text from each page

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

print(f"This is our query: {' '.join(preprocessed_query)} and below are the documents ranked based on their relevance to the query:")
for i, rank in enumerate(ranks):
    print(f"Rank {i+1} (Page {rank+1}): {' '.join(preprocessed_docs[rank])} (Score: {scores[rank]:.4f})")
