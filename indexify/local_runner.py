import hashlib
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from indexify.extractor_sdk.data import BaseData, Feature
from indexify.extractor_sdk.extractor import Extractor, extractor
from indexify.graph import Graph
from indexify.runner import Runner


class LocalRunner(Runner):
    def __init__(self):
        self.results: Dict[str, Any] = defaultdict(
            list
        )  # TODO should the Any be Content?

    def run(self, g, wf_input: BaseData):
        return self._run(g, _input=wf_input, node_name=g._start_node)

    # graph is getting some files which are files, some lables and the MIME type of the bytes
    # those bytes have to be a python type

    # _input needs to be serializable into python object (ie json for ex) and Feature
    def _run(self, g: Graph, _input: BaseData, node_name: str):
        print(f"---- Starting node {node_name}")
        print(f"node_name {node_name}")

        extractor_construct: Callable = g.nodes[node_name]
        params = g.params.get(node_name, None)

        # NOTE: User should clear cache for nodes they would like to re-rerun
        input_hash = hashlib.sha256(str(_input).encode()).hexdigest()
        memo_output = self.get_from_memo(node_name, input_hash)
        if memo_output is None:
            print("=== FYI Writing output to cache")
            res = extractor_construct().extract(input=_input, params=params)
            self.put_into_memo(node_name, input_hash, pickle.dumps(res))
        else:
            print("=== Reading output from cache")
            res = pickle.loads(memo_output)

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
                if self._prefilter_content(
                    content=r, prefilter_predicate=pre_filter_predicate
                ):
                    continue

                self._run(g, _input=r, node_name=out_edge)

    """
    Returns True if content should be filtered
    """

    def _prefilter_content(
        self, content: BaseData, prefilter_predicate: Optional[str]
    ) -> bool:
        if prefilter_predicate is None:
            return False

        atoms = prefilter_predicate.split("and")
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

    def deleted_from_memo(self, node_name):
        path_prefix = f"./indexify_local_runner_cache/{node_name}"

        if os.path.exists(path_prefix) and os.path.isdir(path_prefix):
            shutil.rmtree(path_prefix)

    def get_from_memo(self, node_name, input_hash):
        path_prefix = f"./indexify_local_runner_cache/{node_name}"
        file_name = f"{input_hash}"
        file_path = f"{path_prefix}/{file_name}"

        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            return f.read()

    def put_into_memo(self, node_name, input_hash, output):
        path_prefix = f"./indexify_local_runner_cache/{node_name}"
        file_name = f"{input_hash}"
        file_path = f"{path_prefix}/{file_name}"

        os.makedirs(path_prefix, exist_ok=True)

        Path(file_path).touch()

        with open(file_path, "wb") as f:
            return f.write(output)
