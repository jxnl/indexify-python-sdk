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
            res = extractor_construct.extract(node_name, input_data)
            self.results[node_name].extend(res)

            for out_edge, pre_filter_predicate in g.edges[node_name]:
                outputs = [
                    output_of_node
                    for output_of_node in self.results[node_name]
                    if not self._pre_filter_content(
                        content=output_of_node,
                        pre_filter_predicate=pre_filter_predicate,
                    )
                ]
                print(f"[bold] invoking {out_edge} with {len(outputs)}[/bold]")
                for output in outputs:
                    queue.append((out_edge, output))

    def _pre_filter_content(
        self, content: BaseData, pre_filter_predicate: Optional[str]
    ) -> bool:
        if pre_filter_predicate is None:
            return False

        atoms = pre_filter_predicate.split("and")
        if len(atoms) == 0:
            return False

        # TODO For now only support `and` and `=` and `string values`
        bools = []
        metadata = content.get_features()["metadata"]
        for atom in atoms:
            l, r = atom.split("=")
            if l in metadata:
                bools.append(metadata[l] != r)

        return all(bools)

    def get_result(self, node: Union[extractor, Extractor]) -> Any:
        node_name = node.name
        return self.results[node_name]
