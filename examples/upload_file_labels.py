from indexify import Document, ExtractionGraph, IndexifyClient

client = IndexifyClient()

# setup extraction graph
clip_graph = ExtractionGraph.from_yaml(
    """
name: clipkb
extraction_policies:
  - extractor: 'tensorlake/clip-extractor'
    name: clipgif
"""
)
client.create_extraction_graph(clip_graph)

# download file from https://picsum.photos/id/1/200/300
import requests

file_path = "tmp.jpg"
res = requests.get("https://picsum.photos/id/1/200/300")
f = open(file_path, "wb")
f.write(res.content)
f.close()

# upload file with labels
content_id = client.upload_file("clipkb", file_path, labels={"type": "image"})
print(client.get_content_metadata(content_id))
