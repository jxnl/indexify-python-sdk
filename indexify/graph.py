from collections import defaultdict
from typing import Dict, List, Optional, Type

import cloudpickle

from .extractor_sdk import Extractor
from .extractor_sdk.data import BaseData
from .extractor_sdk.extractor import ExtractorWrapper


def load_graph(graph: bytes) -> "Graph":
    return cloudpickle.loads(graph)


class Graph:
    def __init__(self, name: str, start_node: Extractor):
        self.name = name

        self.nodes: Dict[str, ExtractorWrapper] = {}

        self.edges: Dict[str, List[(str, str)]] = defaultdict(list)

        self._start_node: str = start_node.name
        self.add_node(start_node)

    def get_extractor(self, name: str) -> ExtractorWrapper:
        return self.nodes[name]

    def add_node(self, extractor: Extractor) -> "Graph":
        name = extractor.name
        if name in self.nodes:
            return

        self.nodes[name] = ExtractorWrapper(extractor)

        return self

    def serialize(self):
        return cloudpickle.dumps(self)

    @staticmethod
    def deserialize(graph: bytes) -> "Graph":
        return cloudpickle.loads(graph)

    def add_edge(
        self,
        from_node: Type[Extractor],
        to_node: Type[Extractor],
        pre_filter_predicates: Optional[str] = None,
    ) -> "Graph":

        self.add_node(from_node)
        self.add_node(to_node)

        from_node_name = from_node.name
        to_node_name = to_node.name

        self.edges[from_node_name].append((to_node_name, pre_filter_predicates))
        return self
