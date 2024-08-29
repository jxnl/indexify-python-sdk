from abc import ABC, abstractmethod
from functools import update_wrapper
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel
from typing_extensions import get_type_hints

from .data_objects import BaseData


class EmbeddingIndexes(BaseModel):
    dim: int
    distance: Optional[str] = "cosine"
    database_url: Optional[str] = None


class PlacementConstraints(BaseModel):
    min_python_version: Optional[str] = "3.9"
    max_python_version: Optional[str] = None
    platform: Optional[str] = None
    image_name: Optional[str] = None


class IndexifyFunction(ABC):
    name: str = ""
    version: str = "0.0.0"
    base_image: Optional[str] = None
    system_dependencies: List[str] = []
    python_dependencies: List[str] = []
    description: str = ""
    input_mime_types = ["text/plain"]
    embedding_indexes: Dict[str, EmbeddingIndexes] = {}
    placement_constraints: Optional[PlacementConstraints] = None

    @abstractmethod
    def run(self, *args, **kwargs) -> Union[List[Any], Any]:
        pass

    def partial(self, **kwargs) -> Callable:
        from functools import partial

        return partial(self.run, **kwargs)


def indexify_function(
    name: Optional[str] = None,
    description: Optional[str] = "",
    version: Optional[str] = "",
    base_image: Optional[str] = "ubuntu:22.04",
    python_dependencies: Optional[List[str]] = None,
    system_dependencies: Optional[List[str]] = None,
    input_mime_types: Optional[List[str]] = None,
    embedding_indexes: Optional[Dict[str, EmbeddingIndexes]] = None,
):
    def construct(fn):
        args = locals().copy()
        args["name"] = args["name"] if args.get("name", None) else fn.__name__
        args["description"] = (
            args["description"] if args.get("description", None) else fn.__doc__
        )

        class IndexifyFn(IndexifyFunction):
            def run(self, *args, **kwargs) -> Union[List[Any], Any]:
                return fn(*args, **kwargs)

            update_wrapper(run, fn)

        for key, value in args.items():
            if key != "fn" and key != "self":
                setattr(IndexifyFn, key, value)

        return IndexifyFn

    return construct


class IndexifyFunctionWrapper:
    def __init__(self, indexify_function: IndexifyFunction):
        self.indexify_function: IndexifyFunction = indexify_function()

    def get_output_model(self) -> Any:
        if not isinstance(self.indexify_function, IndexifyFunction):
            raise TypeError("Input must be an instance of IndexifyFunction")

        extract_method = self.indexify_function.run
        type_hints = get_type_hints(extract_method)
        return_type = type_hints.get("return", Any)
        return return_type

    def run(self, input: BaseData) -> List[BaseData]:
        input = input.payload
        print(type(input))
        if str(type(input)) == "<class 'indexify.functions_sdk.data_objects.DynamicModel'>":
            extractor_input = {}
            for field_name in input.model_fields:
                extractor_input[field_name] = getattr(input, field_name)
            extracted_data = self.indexify_function.run(**extractor_input)
        else:
            extracted_data = self.indexify_function.run(input)

        if not isinstance(extracted_data, list):
            return [BaseData(payload=extracted_data)]

        outputs = []
        for data in extracted_data:
            outputs.append(BaseData(payload=data))
        return outputs
