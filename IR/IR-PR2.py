# Document Retrieval using Inverted File Index

from collections import defaultdict

# Step 1: Sample documents
documents = {
    1: "machine learning is amazing",
    2: "deep learning drives artificial intelligence",
    3: "machine learning and deep learning are related fields",
    4: "artificial intelligence is the future"
}

# Step 2: Create an inverted index
inverted_index = defaultdict(list)

for doc_id, text in documents.items():
    for word in text.lower().split():
        if doc_id not in inverted_index[word]:
            inverted_index[word].append(doc_id)

# Step 3: Function to retrieve documents
def retrieve_docs(query):
    words = query.lower().split()
    result_sets = []
    for word in words:
        result_sets.append(set(inverted_index.get(word, [])))
    if not result_sets:
        return []
    # Intersection of all sets to get common docs
    result = set.intersection(*result_sets)
    return list(result)

# Step 4: Display inverted index
print("Inverted Index:")
for word, doc_list in inverted_index.items():
    print(f"{word} → {doc_list}")

# Step 5: Test retrieval
while True:
    query = input("\nEnter query (or 'exit' to stop): ")
    if query.lower() == "exit":
        break
    docs = retrieve_docs(query)
    if docs:
        print(f"Documents matching '{query}': {docs}")
    else:
        print(f"No documents found for '{query}'")
