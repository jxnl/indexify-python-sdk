from collections import defaultdict
from typing import Callable, Dict, List, Optional, Type

import cloudpickle

from .functions_sdk.indexify_functions import IndexifyFunction, IndexifyFunctionWrapper
from .functions_sdk.data_objects import BaseData


def load_graph(graph: bytes) -> "Graph":
    return cloudpickle.loads(graph)


class Graph:
    def __init__(self, name: str, start_node: IndexifyFunction):
        self.name = name

        self.nodes: Dict[str, IndexifyFunctionWrapper] = {}
        self.routers: Dict[str, Callable[[BaseData], Optional[str]]] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)

        self._start_node: str = start_node.name
        self.add_node(start_node)

    def get_extractor(self, name: str) -> IndexifyFunctionWrapper:
        return self.nodes[name]

    def add_node(self, indexify_fn: IndexifyFunction) -> "Graph":
        name = indexify_fn.name
        if name in self.nodes:
            return

        self.nodes[name] = IndexifyFunctionWrapper(indexify_fn)

        return self

    def serialize(self):
        return cloudpickle.dumps(self)

    @staticmethod
    def deserialize(graph: bytes) -> "Graph":
        return cloudpickle.loads(graph)

    def add_edge(
        self,
        from_node: Type[IndexifyFunction],
        to_node: Type[IndexifyFunction],
    ) -> "Graph":

        self.add_node(from_node)
        self.add_node(to_node)

        from_node_name = from_node.name
        to_node_name = to_node.name

        self.edges[from_node_name].append(to_node_name)
        return self

    def route(
        self, from_node: Type[IndexifyFunction], router: Callable[[BaseData], Optional[str]]
    ) -> "Graph":
        from_node_name = from_node.name
        self.routers[from_node_name] = router
        return self
