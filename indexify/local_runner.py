from indexify import Content, Extractor

from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import json

class LocalRunner:
    def __init__(self):
        self.results: Dict[str, Any] = defaultdict(list) # TODO should the Any be Content?

    def run(self, g, content: Content):
        g._assign_start_node()
        return self._run(g, content=content, node_name=g._start_node)

    def _run(self, g, content: Content, node_name: str):
        extractor_construct: Callable = g.nodes[node_name]
        params = g.params.get(node_name, None)

        print(f"----Starting {node_name}")

        res = extractor_construct().extract(content=content, params=params)

        self.results[node_name].extend(res)

        for out_edge, pre_filter_predicate in g.edges[node_name]:
            # TODO there are no reductions yet, each recursion finishes it's path and returns
            for r in res:
                if self._prefilter_content(content=r, prefilter_predicate=pre_filter_predicate):
                    continue

                self._run(g, content=r, node_name=out_edge)

    """
    Returns True if content should be filtered
    """
    def _prefilter_content(self, content: Content, prefilter_predicate: Optional[str]) -> bool:
        if prefilter_predicate is None:
            return False

        atoms = prefilter_predicate.split('and')
        if len(atoms) == 0:
            return False

        # TODO For now only support `and` and `=` and `string values`
        bools = []
        for feature in content.features:
            if feature.feature_type == 'metadata':
                predicates = json.loads(feature.value)

                print(f"predicates {predicates}")

                for atom in atoms:
                    l, r = atom.split('=')
                    if l in predicates:
                        print(f'predicates[l], r: {predicates[l], r}')
                        bools.append(predicates[l] != r)

        print(bools)

        return all(bools)

    def get_result(self, node: Extractor) -> Content:
        node_name = node._extractor_name
        return self.results[node_name]
