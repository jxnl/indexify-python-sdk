from abc import ABC, abstractmethod
from functools import update_wrapper
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel
from typing_extensions import get_type_hints

from .data import BaseData


class EmbeddingSchema(BaseModel):
    dim: int
    distance: Optional[str] = "cosine"
    database_url: Optional[str] = None


class ExtractorMetadata(BaseModel):
    name: str
    version: str
    description: str
    input_mime_types: List[str]
    system_dependencies: List[str]
    python_dependencies: List[str]
    input_mime_types: List[str]
    embedding_schemas: Dict[str, EmbeddingSchema]
    # Make this a dynamic model since its a json schema
    input_params: Optional[Dict]
    # for backward compatibility
    metadata_schemas: Optional[Dict]


class PlacementConstraints(BaseModel):
    min_python_version: Optional[str] = "3.9"
    max_python_version: Optional[str] = None
    platform: Optional[str] = None
    image_name: Optional[str] = None


class Extractor(ABC):
    name: str = ""
    version: str = "0.0.0"
    base_image: Optional[str] = None
    system_dependencies: List[str] = []
    python_dependencies: List[str] = []
    description: str = ""
    input_mime_types = ["text/plain"]
    embedding_indexes: Dict[str, EmbeddingSchema] = {}
    placement_constraints: Optional[PlacementConstraints] = None

    @abstractmethod
    def extract(self, *args, **kwargs) -> Union[List[Any], Any]:
        pass

    def partial(self, **kwargs) -> Callable:
        from functools import partial

        return partial(self.extract, **kwargs)


def extractor(
    name: Optional[str] = None,
    description: Optional[str] = "",
    version: Optional[str] = "",
    base_image: Optional[str] = "ubuntu:22.04",
    python_dependencies: Optional[List[str]] = None,
    system_dependencies: Optional[List[str]] = None,
    input_mime_types: Optional[List[str]] = None,
    embedding_indexes: Optional[Dict[str, EmbeddingSchema]] = None,
):
    def construct(fn):
        args = locals().copy()
        args["name"] = args["name"] if args.get("name", None) else fn.__name__
        args["description"] = (
            args["description"] if args.get("description", None) else fn.__doc__
        )

        class IndexifyFnExtractor(Extractor):
            def extract(self, *args, **kwargs) -> Union[List[Any], Any]:
                return fn(*args, **kwargs)

            update_wrapper(extract, fn)

        for key, value in args.items():
            if key != "fn" and key != "self":
                setattr(IndexifyFnExtractor, key, value)

        return IndexifyFnExtractor

    return construct


class ExtractorWrapper:
    def __init__(self, extractor: Extractor):
        self.extractor: Extractor = extractor()

    def get_output_model(self) -> List[Type[BaseModel]]:
        if not isinstance(self.extractor, Extractor):
            raise TypeError("Input must be an instance of Extractor")

        extract_method = self.extractor.extract
        type_hints = get_type_hints(extract_method)
        return_type = type_hints.get("return", Any)
        return return_type

    def extract(self, input: BaseData) -> List[BaseData]:
        input = input.payload
        if str(type(input)) == "<class 'indexify.extractor_sdk.data.DynamicModel'>":
            extracted_data = self.extractor.extract(**input.model_dump())
        else:
            extracted_data = self.extractor.extract(input)

        if not isinstance(extracted_data, list):
            return [BaseData(payload=extracted_data)]

        outputs = []
        for data in extracted_data:
            outputs.append(BaseData(payload=data))
        return outputs
