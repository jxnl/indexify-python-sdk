import itertools
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

import cloudpickle
from pydantic import BaseModel

from .extractor_sdk import Content, Extractor, extractor
from .runner import Runner


@extractor(description="id function")
def _id(content: Content) -> List[Content]:
    return [content]


def load_graph(graph: bytes) -> "Graph":
    return cloudpickle.loads(graph)


class Graph:
    def __init__(
        self, name: str, input: Type[BaseModel], start_node: extractor, runner: Runner
    ):
        # TODO check for cycles
        self.name = name

        self.nodes: Dict[str, Union[extractor, Extractor]] = {}
        self.params: Dict[str, Any] = {}

        self.edges: Dict[str, List[(str, str)]] = defaultdict(list)

        self.nodes["start"] = _id
        self.nodes["end"] = _id

        self._topo_counter = defaultdict(int)

        self._start_node = None
        self._input = input

        self.runner = runner

    def get_extractor(self, name: str) -> Extractor:
        return self.nodes[name]

    def _node(self, extractor: Extractor, params: Any = None) -> "Graph":
        name = extractor.name

        # if you've already inserted a node just ignore the new insertion.
        if name in self.nodes:
            return

        self.nodes[name] = extractor
        self.params[name] = extractor.__dict__.get("params", None)

        # assign each node a rank of 1 to init the graph
        self._topo_counter[name] = 1

        return self

    def serialize(self):
        return cloudpickle.dumps(self)

    def add_edge(
        self,
        from_node: Type[Extractor],
        to_node: Type[Extractor],
        prefilter_predicates: Optional[str] = None,
    ) -> "Graph":

        self._node(from_node)
        self._node(to_node)

        from_node_name = from_node.name
        to_node_name = to_node.name

        self.edges[from_node_name].append((to_node_name, prefilter_predicates))

        self._topo_counter[to_node_name] += 1

        return self

    """
    Connect nodes as a fan out from one `from_node` to multiple `to_nodes` and respective `prefilter_predicates`.
    Note: The user has to match the sizes of the lists to make sure they line up otherwise a None is used as a default.
    """

    def steps(
        self,
        from_node: extractor,
        to_nodes: List[extractor],
        prefilter_predicates: List[str] = [],
    ) -> "Graph":
        print(f"{to_nodes}, {prefilter_predicates}, {prefilter_predicates}")
        for t_n, p in itertools.zip_longest(
            to_nodes, prefilter_predicates, fillvalue=None
        ):
            self.step(from_node=from_node, to_node=t_n, prefilter_predicates=p)

        return self

    def add_param(self, node: extractor, params: Dict[str, Any]):
        try:
            # check if the params can be serialized since the server needs this
            json.dumps(params)
        except Exception:
            raise Exception(f"For node {node.name}, cannot serialize params as json.")

        self.params[node.name] = params

    def run(self, wf_input, local):
        self._assign_start_node()
        self.runner.run(self, wf_input=wf_input)
        pass

    def clear_cache_for_node(self, node: Union[extractor, Extractor]):
        if node.name not in self.nodes.keys():
            raise Exception(f"Node with name {node.name} not found in graph")

        self.runner.deleted_from_memo(node.name)

    def clear_cache_for_all_nodes(self):
        for node_name in self.nodes:
            self.runner.deleted_from_memo(node_name=node_name)

    def get_result(self, node: Union[extractor, Extractor]) -> Any:
        return self.runner.results[node.name]

    def _assign_start_node(self):
        # this method should be called before a graph can be run
        nodes = sorted(self._topo_counter.items(), key=lambda x: x[1])
        self._start_node = nodes[0][0]
