from collections import defaultdict
from queue import deque
from typing import Any, Dict, Optional, Type, Union

from rich import print

from indexify.extractor_sdk.data import BaseData
from indexify.extractor_sdk.extractor import Extractor, extractor
from indexify.extractor_sdk.local_cache import CacheAwareExtractorWrapper
from indexify.graph import Graph
from indexify.runner import Runner


class LocalRunner(Runner):
    def __init__(self, cache_dir: str = "./indexify_local_runner_cache"):
        self.results: Dict[str, Type[BaseData]] = defaultdict(list)
        self._extractors: Dict[str, CacheAwareExtractorWrapper] = {}
        self._cache_dir = cache_dir

    def run_from_serialized_code(self, code: bytes, **kwargs):
        g = Graph.deserialize(graph=code)
        self.run(g, **kwargs)

    def run(self, g: Graph, **kwargs):
        input = BaseData.from_data(**kwargs)
        print(f"[bold] Invoking {g._start_node}[/bold]")
        self._run(g, input)

    def _run(self, g: Graph, initial_input: Type[BaseData]):
        queue = deque([(g._start_node, initial_input)])

        while queue:
            node_name, input_data = queue.popleft()

            extractor_construct = self._extractors.setdefault(
                node_name,
                CacheAwareExtractorWrapper(
                    self._cache_dir, g.name, g.get_extractor(node_name)
                ),
            )
            extractor_results = extractor_construct.extract(node_name, input_data)
            self.results[node_name].extend(extractor_results)

            out_edges = g.edges.get(node_name, [])
            # Figure out if there are any routers for this node
            if node_name in g.routers:
                for output in extractor_results:
                    out_edge = self._route(g, node_name, output)
                    if out_edge is not None and out_edge in g.nodes:
                        print(f"[bold]dynamic router returned node: {out_edge}[/bold]")
                        out_edges.append(out_edge)

            for out_edge in out_edges:
                print(
                    f"invoking {out_edge} with {len(extractor_results)} outputs from {node_name}"
                )
                for output in extractor_results:
                    queue.append((out_edge, output))

    def _route(self, g: Graph, node_name: str, input: Type[BaseData]) -> Optional[str]:
        if str(type(input)) == "<class 'indexify.extractor_sdk.data.DynamicModel'>":
            return g.routers[node_name](**input.model_dump())
        return g.routers[node_name](input.payload)

    def get_result(self, node: Union[extractor, Extractor]) -> Any:
        node_name = node.name
        return self.results[node_name]
