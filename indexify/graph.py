from collections import defaultdict
from typing import Dict, List, Optional, Type, Callable

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
        self.routers: Dict[str, Callable[[BaseData], Optional[str]]] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)

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
    ) -> "Graph":

        self.add_node(from_node)
        self.add_node(to_node)

        from_node_name = from_node.name
        to_node_name = to_node.name

        self.edges[from_node_name].append(to_node_name)
        return self

    def route(
        self, from_node: Type[Extractor], router: Callable[[BaseData], Optional[str]]
    ) -> "Graph":
        from_node_name = from_node.name
        self.routers[from_node_name] = router
        return self
