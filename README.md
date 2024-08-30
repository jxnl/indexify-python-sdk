# Indexify Python SDK

[![PyPI version](https://badge.fury.io/py/indexify.svg)](https://badge.fury.io/py/indexify)

This is the Python SDK to build real-time continuously running unstructured data processing pipelines with Indexify.

Start by writing and testing your pipelines locally using your data, then deploy them into the Indexify service to process data in real-time at scale.

## Installation

```shell
pip install indexify
```

## Quick Start
Write compute functions, and build a compute graph that process your data. The output of each functions is automatically passed to the next function in the graph by Indexify. 

If a function returns a list, Indexify will automatically invoke the next function with each item in the list in **parallel**.

The input of the first function becomes the input to the HTTP endpoint of the Graph.

```python
from pydantic import BaseModel
from indexify import indexify_function

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any]

class TextChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@indexify_function()
def split_text(doc: Document) -> List[TextChunk]:
    midpoint = len(doc.text) // 2
    first_half = TextChunk(text=doc.text[:midpoint], metadata=doc.metadata)
    second_half = TextChunk(text=doc.text[midpoint:], metadata=doc.metadata)
    return [first_half, second_half]

@indexify_function()
def embed_text(chunk: TextChunk) -> TextChunk:
    chunk.embedding = [0.1, 0.2, 0.3]
    return chunk

@indexify_function()
def write_to_db(chunk: TextChunk):
    # Write to your favorite vector database
    print(chunk)

## Create a graph
```python
from indexify import Graph

g = Graph(name="my_graph", start_node=split_text)
g.add_edge(split_text, embed_text)
g.add_edge(embed_text, write_to_db)
```

## Run the Graph Locally
```python
from indexify import IndexifyClient

client = IndexifyClient(local=True)
client.register_graph(g)
output_id = client.invoke_graph_with_object(g.name, Document(text="Hello, world!", metadata={"source": "test"}))
graph_outputs = client.graph_outputs(g.name, output_id)
```

## Deploy the Graph to Indexify Server for Production
```python
from indexify import IndexifyClient

client = IndexifyClient(service_url="http://localhost:8000")
client.register_graph(g)
```

#### Ingestion into the Service
Extraction Graphs continuously run on the Indexify Service like any other web service. Indexify Server runs the extraction graphs in parallel and in real-time when new data is ingested into the service.

```python
output_id = client.invoke_graph_with_object(g.name, Document(text="Hello, world!", metadata={"source": "test"}))
```

#### Retrieve Graph Outputs for a given ingestion object
```python
graph_outputs = client.graph_outputs(g.name, output_id)
```

#### Retrieve All Graph Inputs 
```python
graph_inputs = client.graph_inputs(g.name)
```

## Examples 
** [PDF Document Extraction](./examples/pdf_document_extraction/workflow.py) **
1. Extracts text, tables and images from an ingested PDF file
2. Indexes the text using MiniLM-L6-v2, the images with CLIP
3. Writes the results into a vector database.

** [Meeting Minutes Extraction](./examples/meeting_minutes_extraction/workflow.py) **
1. Extracts transcriptions 
