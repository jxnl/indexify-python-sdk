from indexify.extractor_sdk.data import BaseData, Feature
from indexify.extractor_sdk.extractor import extractor

from collections import defaultdict
from typing import Any, Callable, Dict, Optional


class LocalRunner:
    def __init__(self):
        self.results: Dict[str, Any] = defaultdict(
            list
        )  # TODO should the Any be Content?

    def run(self, g, wf_input: BaseData):
        g._assign_start_node()
        return self._run(g, _input=wf_input, node_name=g._start_node)


    # graph is getting some files which are files, some lables and the MIME type of the bytes
    # those bytes have to be a python type

    # _input needs to be serializable into python object (ie json for ex) and Feature
    def _run(self, g, _input: BaseData, node_name: str):
        print(f"---- Starting node {node_name}")

        extractor_construct: Callable = g.nodes[node_name]
        params = g.params.get(node_name, None)

        res = extractor_construct().extract(_input=_input, params=params)
        if not isinstance(res, list):
            res = [res]


        res_data = [i for i in res if not isinstance(i, Feature)]
        res_features = [i for i in res if isinstance(i, Feature)]

        self.results[node_name].extend(res_data)

        for f in res_features:
            _input.meta[f.name] = f.value

        # this assume that if an extractor emits features then the next edge will always process
        # the edges
        data_to_process = res_data
        if len(res_features) > 0:
            data_to_process.append(_input)

        for out_edge, pre_filter_predicate in g.edges[node_name]:
            # TODO there are no reductions yet, each recursion finishes it's path and returns
            for r in data_to_process:
                if self._prefilter_content(content=r, prefilter_predicate=pre_filter_predicate):
                    continue

                self._run(g, _input=r, node_name=out_edge)

    """
    Returns True if content should be filtered
    """
    def _prefilter_content(self, content: BaseData, prefilter_predicate: Optional[str]) -> bool:
        if prefilter_predicate is None:
            return False

        atoms = prefilter_predicate.split("and")
        if len(atoms) == 0:
            return False

        # TODO For now only support `and` and `=` and `string values`
        bools = []
        metadata = content.get_features()['metadata']
        for atom in atoms:
            l, r = atom.split('=')
            if l in metadata:
                bools.append(metadata[l] != r)

        return all(bools)

    def get_result(self, node: extractor) -> Any:
        node_name = node._extractor_name
        return self.results[node_name]
