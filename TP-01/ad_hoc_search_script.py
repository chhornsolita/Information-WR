from elasticsearch import Elasticsearch

# Connect to Elasticsearch (HTTP mode)
es = Elasticsearch("http://localhost:9200")

# Test connection
if not es.ping():
    raise Exception("Connection failed.")
print("Connected to Elasticsearch!\n")

# --- Step 1: Index sample documents ---
docs = [
    {"title": "Machine Learning Basics", "content": "Introduction to ML concepts."},
    {"title": "Deep Learning Guide", "content": "Neural networks and more."},
    {"title": "Python Programming", "content": "Python for data analysis."},
    {"title": "Information Retrieval", "content": "Elasticsearch and ad-hoc search."},
    {"title": "Data Science Overview", "content": "Statistics and ML applications."}
]

index_name = "books"

# Index documents
for i, doc in enumerate(docs, 1):
    es.index(index=index_name, id=i, document=doc)
print("Documents indexed.\n")

# --- Step 2: Perform ad-hoc search ---
query_text = "neural networks"
query = {
    "query": {
        "match": {
            "content": query_text
        }
    }
}

res = es.search(index=index_name, body=query)

# --- Step 3: Display results with ranking scores ---
print(f"Search results for: '{query_text}'\n")
for hit in res['hits']['hits']:
    print(f"Score: {hit['_score']:.4f}, Title: {hit['_source']['title']}")
 
 
 
 