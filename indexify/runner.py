from abc import ABC

from indexify.extractor_sdk.data import BaseData
from indexify.extractor_sdk.extractor import extractor, Extractor

from typing import Any, Union

class Runner(ABC):
    def run(self, g, wf_input: BaseData):
        raise NotImplementedError()

    def get_result(self, node: Union[extractor, Extractor]) -> Any:
        raise NotImplementedError()

    def deleted_from_memo(self, node_name):
        raise NotImplementedError()

    def get_from_memo(self, node_name, input_hash):
        raise NotImplementedError()

    def put_into_memo(self, node_name, input_hash, output):
        raise NotImplementedError()
