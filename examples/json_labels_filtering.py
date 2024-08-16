# This requires the server to be running with MiniLM extractor.

import uuid
from indexify import IndexifyClient, ExtractionGraph, Document

# Create a simple extraction graph.
client = IndexifyClient()

graph_name = uuid.uuid4().hex
graph = f"""
name: "{graph_name}"
extraction_policies:
  - extractor: "tensorlake/minilm-l6"
    name: 'embeddings'
"""

extraction_graph = ExtractionGraph.from_yaml(graph)
client.create_extraction_graph(extraction_graph)

documents = [
    Document("I love apples", {"year": 2020, "fruit": "apple"}, None),
    Document("I love bananas", {"year": 2020, "fruit": "banana"}, None),
    Document("I love deep fried banana", {"year": 2021, "fruit": "banana"}, None),
]

content_ids = client.add_documents(graph_name, documents)

for content_id in content_ids:
    client.wait_for_extraction(content_id)

indexes = client.indexes()

for index in indexes:
    if index["name"].startswith(graph_name):
        index_name = index["name"]
        break

# Search for "love" in the index
all_docs = client.search_index(index_name, "love", 3)
assert len(all_docs) == 3

# Search with label filter string.
banana_docs = client.search_index(index_name, "love", 3, ["fruit=banana"])
assert len(banana_docs) == 2

# Search with label filter number.
docs_2020 = client.search_index(index_name, "love", 3, ["year=2020"])
assert len(docs_2020) == 2

# Search with multiple label filters.
docs_2020_banana = client.search_index(
    index_name, "love", 3, ["year=2020", "fruit=banana"]
)

assert len(docs_2020_banana) == 1
