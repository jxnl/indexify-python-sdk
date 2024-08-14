from indexify import Content, extractor
from indexify.extractor import Extractor

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Self

import itertools


@extractor(description="id function")
def _id(content: Content) -> List[Content]:
    return [content]

class Graph:
    def __init__(self, name: str):
        # TODO check for cycles
        self.name = name

        self.nodes: Dict[str, Callable] = {}
        self.params: Dict[str, Any] = {}

        self.edges: Dict[str, List[(str, str)]] = defaultdict(list)

        self.results: Dict[str, Any] = defaultdict(list) # TODO should the Any be Content?

        self.nodes["start"] = _id
        self.nodes["end"] = _id

        self._topo_counter = defaultdict(int)

        self._start_node = None

    def _node(self, extractor: Extractor, params: Any = None) -> Self:
        name = extractor._extractor_name

        # if you've already inserted a node just ignore the new insertion.
        if name in self.nodes:
            return

        self.nodes[name] = extractor
        self.params[name] = extractor.__dict__.get('params', None)

        # assign each node a rank of 1 to init the graph
        self._topo_counter[name] = 1

        return self

    def step(self,
             from_node: extractor,
             to_node: extractor,
             prefilter_predicates: Optional[str] = None
    ) -> Self:

        self._node(from_node)
        self._node(to_node)

        from_node_name = from_node._extractor_name
        to_node_name = to_node._extractor_name

        self.edges[from_node_name].append((to_node_name, prefilter_predicates))

        self._topo_counter[to_node_name] += 1

        return self

    """
    Connect nodes as a fan out from one `from_node` to multiple `to_nodes` and respective `prefilter_predicates`.
    Note: The user has to match the sizes of the lists to make sure they line up otherwise a None is used as a default.
    """
    def steps(self, from_node: extractor, to_nodes: List[extractor], prefilter_predicates: List[str] = []) -> Self:
        print(f'{to_nodes}, {prefilter_predicates}, {prefilter_predicates}')
        for t_n, p in itertools.zip_longest(to_nodes, prefilter_predicates, fillvalue=None):
            self.step(from_node=from_node, to_node=t_n, prefilter_predicates=p)

        return self

    def _assign_start_node(self):
        # this method should be called before a graph can be run
        nodes = sorted(self._topo_counter.items(), key=lambda x: x[1])
        self._start_node = nodes[0][0]