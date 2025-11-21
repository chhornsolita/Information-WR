from elasticsearch import Elasticsearch

# For HTTP (security disabled)
es = Elasticsearch("http://localhost:9200")

# Test connection
if es.ping():
    print("Elasticsearch is connected!")
else:
    print("Connection failed.")

