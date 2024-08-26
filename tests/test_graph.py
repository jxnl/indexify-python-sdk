import unittest
from typing import List, Mapping

from pydantic import BaseModel

from indexify import Graph
from indexify.extractor_sdk.data import BaseData, File
from indexify.extractor_sdk.extractor import extractor
from indexify.local_runner import LocalRunner


@extractor()
def extractor_a(url: str) -> File:
    """
    Random description of extractor_a
    """
    return File(data=bytes(b"hello"), mime_type="text/plain")


class FileChunk(BaseModel):
    data: bytes
    start: int
    end: int


@extractor()
def extractor_b(file: File) -> List[FileChunk]:
    return [
        FileChunk(data=file.data, start=0, end=5),
        FileChunk(data=file.data, start=5, end=len(file.data)),
    ]


class SomeMetadata(BaseModel):
    metadata: Mapping[str, str]


@extractor()
def extractor_c(file_chunk: FileChunk) -> SomeMetadata:
    return SomeMetadata(metadata={"a": "b", "c": "d"})


def create_graph_a():
    graph = Graph(name="test", start_node=extractor_a)
    graph = graph.add_edge(extractor_a, extractor_b)
    graph = graph.add_edge(extractor_b, extractor_c)
    return graph


class TestGraphA(unittest.TestCase):
    def test_run_graph(self):
        graph = create_graph_a()

        runner = LocalRunner()
        wf_input = BaseData.from_data(url="https://example.com")
        runner.run(graph, wf_input)


if __name__ == "__main__":
    unittest.main()
