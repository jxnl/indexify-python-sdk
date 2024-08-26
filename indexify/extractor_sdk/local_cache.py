import hashlib
import os
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel, RootModel
from typing_extensions import get_args, get_origin

from .data import BaseData
from .extractor import ExtractorWrapper
from .output_serializer import CachedOutput, PayloadSerializer


def get_list_inner_type(type_hint: Type) -> Union[Type, None]:
    """
    Detects and returns the inner type of a List type annotation.

    Args:
    type_hint: A type hint, potentially a List type.

    Returns:
    Type: The inner type of the List, or None if not a List or can't be determined.
    """
    # Check if the type_hint is a List
    if get_origin(type_hint) is list:
        # Get the arguments of the generic type
        args = get_args(type_hint)
        if args:
            # Return the first argument as the inner type
            return args[0]
        else:
            # If no arguments, it's List[Any]
            return Any
    elif isinstance(type_hint, type) and issubclass(type_hint, List):
        # Handle cases where List is used without brackets
        return Any
    else:
        # Not a List type
        return None


class CacheAwareExtractorWrapper:
    def __init__(self, cache_dir: str, graph: str, extractor_wrapper: ExtractorWrapper):
        self._cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self._graph = graph
        self._extractor_wrapper = extractor_wrapper
        self._output_model = extractor_wrapper.get_output_model()

    def extract(self, node_name: str, input: BaseData) -> List[BaseData]:
        cached_result = self.get(self._graph, node_name, input)
        if cached_result:
            return cached_result

        output = self._extractor_wrapper.extract(input)
        self.set(self._graph, node_name, input, output)
        return output

    def get(self, graph: str, node_name: str, input: BaseData) -> List[BaseData]:
        dir_path = os.path.join(self._cache_dir, graph)
        file_path = os.path.join(
            dir_path, f"{node_name}_{input.md5_payload_checksum}.json"
        )

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                output_json = f.read()
                serializer = PayloadSerializer()
                output_model = self._output_model
                if get_origin(self._output_model) is list:
                    output_model = get_list_inner_type(self._output_model)

                class BaseData(BaseModel):
                    content_id: Optional[str] = None
                    md5_payload_checksum: Optional[str] = None
                    payload: output_model

                output = serializer.deserialize(output_json, BaseData)
                return output
        return None

    def set(
        self,
        graph: str,
        node_name: str,
        input: BaseData,
        output: List[BaseData],
    ):
        serializer = PayloadSerializer()
        node_name_with_hex = f"{node_name}_{input.md5_payload_checksum}"
        dir_path = os.path.join(self._cache_dir, graph)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        cached_output = serializer.serialize(CachedOutput(output))
        file_path = os.path.join(dir_path, f"{node_name_with_hex}.json")
        with open(file_path, "w") as f:
            f.write(cached_output)
