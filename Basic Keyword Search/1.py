'''Implementation of a basic keyword search using scikit-learn's TfidfVectorizer and cosine similarity. I made a sample dataset of documents in a list and a user input query.'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Sample documents for the keyword search
documents = [
    'The mould, into which the metal is poured, is made of some heat resisting material', 
    'Sand is most often used as it resists the high temperature of the molten metal. Permanent moulds of metal can also be used to cast products.',
    'The metal casting industry plays a key role in all the major sectors of our economy.',
    'There are castings in locomotives, cars, trucks, aircraft, office buildings, factories, schools, and homes.'
]

# Input query from the user
query = input("Enter your search query: ")

# Preprocessing function to clean and normalize text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

preprocessed_docs = [preprocess_text(doc) for doc in documents]
preprocess_query = preprocess_text(query)

# print(preprocessed_docs)
# print(preprocess_query)

vector = TfidfVectorizer()
doc_vector = vector.fit_transform(preprocessed_docs)
query_vector = vector.transform([preprocess_query])

# print(doc_vector.toarray()[1])        To get a particular vector from the vectors of the preprocessed documents
# print(query_vector.toarray())         To get the vector of the preprocessed query

similarity = cosine_similarity(query_vector, doc_vector) #Finds the cosine similarity between the query vector and document vectors
print(F"Similarity vectors: {similarity}")

ranks = np.argsort(similarity[0],axis=0)[::-1].flatten()
print(f"Ranks of the statments in the documents: {ranks}")  # Sort indices based on similarity scores in descending order

print(f"This is our query: {preprocess_query} and below are the documents ranked based on their relevance to the query:")
for rank in ranks:
    print(f"Rank {rank}: {preprocessed_docs[rank]}")
