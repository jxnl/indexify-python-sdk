from indexify.extractor_sdk.extractor import extractor
from indexify.extractor_sdk.data import File
from indexify import Graph
from pydantic import BaseModel
from typing import List, Mapping
import unittest

@extractor()
def extractor_a(url: str) -> File:
    return File(data=b"hello", mime_type="text/plain")

class FileChunk(BaseModel):
    data: bytes
    start: int
    end: int

@extractor()
def extractor_b(file: File) -> List[FileChunk]:
    return [FileChunk(data=file.data, start=0, end=len(5)), FileChunk(data=file.data, start=5, end=len(file.data))]

class SomeMetadata(BaseModel):
    metadata: Mapping[str, str]

def extractor_c(file_chunk: FileChunk) -> SomeMetadata:
    return SomeMetadata(metadata={"a": "b", "c": "d"})


def create_graph_a():
    graph = Graph(name="test", input=str, start_node=extractor_a, runner=True)
    graph.add_edge(extractor_a, extractor_b)
    graph.add_edge(extractor_b, extractor_c)
    return graph

class TestGraphA(unittest.TestCase):

    def test_run_graph(self):
        graph = create_graph_a()
        results = graph.run(graph, "https://example.com", local=True)
        self.assertEqual(results, {"extractor_a": [File(data=b"hello", mime_type="text/plain")], "extractor_b": [FileChunk(data=b"hello", start=0, end=5), FileChunk(data=b"hello", start=5, end=5)], "extractor_c": [SomeMetadata(metadata={"a": "b", "c": "d"})]})


if __name__ == "__main__":
    unittest.main()
