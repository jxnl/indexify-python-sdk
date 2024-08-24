# This requires the server to be running with MiniLM extractor.

import uuid

from indexify import Document, ExtractionGraph, IndexifyClient

# Create a simple extraction graph.
client = IndexifyClient()

graph_name = uuid.uuid4().hex
graph = f"""
name: "{graph_name}"
extraction_policies:
  - extractor: "tensorlake/minilm-l6"
    name: "embeddings"
"""

extraction_graph = ExtractionGraph.from_yaml(graph)
client.create_extraction_graph(extraction_graph)

label1 = {"fruit": "apple", "year": 2020, "is_good": True}

label2 = {"fruit": "apple", "year": 2022, "is_good": False}

label3 = {"fruit": "banana", "year": 2024, "is_good": False}

documents = [
    Document("I love apples", label1, None),
    Document("I hate apple pie", label2, None),
    Document("I don't like rotten banana", label3, None),
]

content_ids = client.add_documents(graph_name, documents)

for content_id in content_ids:
    client.wait_for_extraction(content_id)

indexes = client.indexes()

for index in indexes:
    if index["name"].startswith(graph_name):
        index_name = index["name"]
        break

query = "love fruit"
k = 3

# Filter by fruit
res = client.search_index(index_name, query, k, ["fruit=apple"])
assert len(res) == 2

# Filter not equal to fruit
res = client.search_index(index_name, query, k, ["fruit!=apple"])
assert len(res) == 1

# Filter by year
res = client.search_index(index_name, "love", 3, ["year>=2020", "year<2024"])
assert len(res) == 2

# Filter by is_good and fruit
res = client.search_index(index_name, "love", 3, ["is_good=true", "fruit=apple"])
assert len(res) == 1

# Filter by is_good and year
res = client.search_index(index_name, "love", 3, ["is_good=false", "year>2022"])
assert len(res) == 1
