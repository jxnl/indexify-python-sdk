import hashlib
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from indexify.extractor_sdk.data import BaseData, Feature
from indexify.extractor_sdk.extractor import (
    Extractor,
    ExtractorWrapper,
    extractor,
)
from indexify.extractor_sdk.local_cache import CacheAwareExtractorWrapper
from indexify.graph import Graph
from indexify.runner import Runner


class LocalRunner(Runner):
    def __init__(self, cache_dir: str = "./indexify_local_runner_cache"):
        self.results: Dict[str, Type[BaseData]] = defaultdict(list)
        self._extractors: Dict[str, CacheAwareExtractorWrapper] = {}
        self._cache_dir = cache_dir

    def run(self, g: Graph, input: Type[BaseData]):
        self._run(g, input, g._start_node)

    def _run(self, g: Graph, _input: Type[BaseData], node_name: str):
        print(f"---- Starting node {node_name}")

        extractor_construct: CacheAwareExtractorWrapper = self._extractors.setdefault(
            node_name,
            CacheAwareExtractorWrapper(
                self._cache_dir, g.name, g.get_extractor(node_name)
            ),
        )

        res = extractor_construct.extract(node_name, _input.payload)
        print(f"---- Finished node {node_name}")
        self.results[node_name].extend(res)

        for out_edge, pre_filter_predicate in g.edges[node_name]:
            for output_of_node in self.results[node_name]:
                if self._pre_filter_content(
                    content=output_of_node, pre_filter_predicate=pre_filter_predicate
                ):
                    continue
                self._run(g, _input=output_of_node, node_name=out_edge)

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
