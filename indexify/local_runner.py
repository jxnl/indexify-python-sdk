from collections import defaultdict
from queue import deque
from typing import Any, Dict, List, Optional, Type, Union

import magic
from nanoid import generate
from pydantic import BaseModel, Json
from rich import print

from indexify.base_client import BaseClient
from indexify.functions_sdk.data_objects import BaseData, File
from indexify.functions_sdk.local_cache import CacheAwareFunctionWrapper
from indexify.graph import Graph


# Holds the outputs of a
class ContentTree(BaseModel):
    id: str
    outputs: Dict[str, List[BaseData]]


class LocalRunner(BaseClient):
    def __init__(self, cache_dir: str = "./indexify_local_runner_cache"):
        self._extractors: Dict[str, CacheAwareFunctionWrapper] = {}
        self._cache_dir = cache_dir
        self._graphs: Dict[str, Graph] = {}
        self._results: Dict[str, Dict[str, List[BaseData]]] = {}

    def register_extraction_graph(self, graph: Graph):
        self._graphs[graph.name] = graph

    def run_from_serialized_code(self, code: bytes, **kwargs):
        g = Graph.deserialize(graph=code)
        self.run(g, **kwargs)

    def run(self, g: Graph, **kwargs):
        input = BaseData.from_data(**kwargs)
        input.content_id = generate()
        print(f"[bold] Invoking {g._start_node}[/bold]")
        outputs = defaultdict(list)
        self._results[input.content_id] = outputs
        self._run(g, input, outputs)
        return input.content_id

    def _run(
        self,
        g: Graph,
        initial_input: Type[BaseData],
        outputs: Dict[str, List[BaseData]],
    ):
        queue = deque([(g._start_node, initial_input)])
        while queue:
            node_name, input_data = queue.popleft()
            extractor_construct = self._extractors.setdefault(
                node_name,
                CacheAwareFunctionWrapper(
                    self._cache_dir, g.name, g.get_extractor(node_name)
                ),
            )
            extractor_results = extractor_construct.run(node_name, input_data)
            for result in extractor_results:
                result.content_id = generate()
            outputs[node_name].extend(extractor_results)

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
        if str(type(input)) == "<class 'indexify.functions_sdk.data_objects.DynamicModel'>":
            return g.routers[node_name](**input.model_dump())
        return g.routers[node_name](input.payload)

    def register_graph(self, graph: Graph):
        self._graphs[graph.name] = graph

    def graphs(self) -> str:
        return list(self._graphs.keys())

    def namespaces(self) -> str:
        return "local"

    def create_namespace(self, namespace: str):
        pass

    def invoke_graph_with_object(self, graph: str, **kwargs) -> str:
        graph = self._graphs[graph]
        return self.run(graph, **kwargs)

    def invoke_graph_with_file(
        self, graph: str, path: str, metadata: Optional[Dict[str, Json]] = None
    ) -> str:
        graph = self._graphs[graph]
        with open(path, "rb") as f:
            data = f.read()
            file = File(
                data, mime_type=magic.from_file(path, mime=True), metadata=metadata
            )
        return self.run(graph, file=file)

    def graph_outputs(
        self,
        graph: str,
        ingested_object_id: str,
        extractor_name: Optional[str] = None,
        block_until_done: bool = True,
    ) -> Union[Dict[str, List[Any]], List[Any]]:
        if ingested_object_id not in self._results:
            raise ValueError(
                f"No results found for ingested object {ingested_object_id}"
            )
        if extractor_name is not None:
            if extractor_name not in self._results[ingested_object_id]:
                raise ValueError(
                    f"No results found for extractor {extractor_name} on ingested object {ingested_object_id}"
                )
            return self._results[ingested_object_id][extractor_name]
        return self._results[ingested_object_id]
