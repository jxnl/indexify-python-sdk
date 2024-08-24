import time

from indexify import ExtractionGraph, IndexifyClient

client = IndexifyClient()
extraction_graph_spec = """
name: 'eg'
extraction_policies:
   - extractor: 'tensorlake/minilm-l6'
     name: 'embeddings'
"""
extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
client.create_extraction_graph(extraction_graph)

# add two documents to database, this will trigger extraction
content_ids = client.add_documents("eg", ["Hello, world!", "Goodbye, world!"])
print(content_ids)

# wait for extraction to finish
client.wait_for_extraction(content_ids[0])
client.wait_for_extraction(content_ids[1])

indexes = client.indexes()

# search with max results 2, this will return both documents since we only have 2
res = client.search_index(indexes[0]["name"], "word", 2)

print(res)

# update labels of the first document to have label "value"
client.update_labels(content_ids[0], {"label": "value"})

# wait for update to propagate to vector index
time.sleep(2)

# search with max results 2 and label filter, this will return only the first document
res = client.search_index(indexes[0]["name"], "word", 2, ["label=value"])

print(res)
