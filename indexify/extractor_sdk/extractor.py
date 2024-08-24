import inspect
import json
import os
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

import requests
from pydantic import BaseModel, Field

from .data import BaseData, Content, Feature

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


class Extractor(ABC):
    name: str = ""

    version: str = "0.0.0"

    base_image: Optional[str] = None

    system_dependencies: List[str] = []

    python_dependencies: List[str] = []

    description: str = ""

    input_mime_types = ["text/plain"]

    embedding_indexes: Dict[str, EmbeddingSchema] = {}

    @abstractmethod
    def extract(
        self, *args, **kwargs) -> Union[List[Any], Any]:
        """
        Extracts information from the content. Returns a list of features to add
        to the content.
        It can also return a list of Content objects, which will be added to storage
        and any extraction policies defined will be applied to them.
        """
        pass

    def partial(self, **kwargs) -> Callable:
        from functools import partial
        return partial(self.extract, **kwargs)


def extractor(
    name: Optional[str] = None,
    description: Optional[str] = "",
    version: Optional[str] = "",
    python_dependencies: Optional[List[str]] = None,
    system_dependencies: Optional[List[str]] = None,
    input_mime_types: Optional[List[str]] = None,
    embedding_indexes: Optional[Dict[str, EmbeddingSchema]] = None,
    sample_content: Optional[Callable] = None,
):
    args = locals()
    del args["sample_content"]

    def construct(fn):
        def wrapper():
            description = fn.__doc__ or args.get("description", "")

            if not args.get("name"):
                args[
                    "name"
                ] = f"{inspect.getmodule(inspect.stack()[1][0]).__name__}:{fn.__name__}"

            class DecoratedFn(Extractor):
                @classmethod
                def extract(cls, input: Type[BaseData], params: Type[BaseModel] = None) -> List[Union[Type[BaseModel], Type[Feature]]]:  # type: ignore
                    # TODO we can force all the functions to take in a parms object
                    # or check if someone adds a params
                    if params is None:
                        return fn(input)
                    else:
                        return fn(input, params)

                def sample_input(self) -> Content:
                    return sample_content() if sample_content else self.sample_text()

            for key, val in args.items():
                setattr(DecoratedFn, key, val)
            DecoratedFn.description = description

            return DecoratedFn

        wrapper._extractor_name = fn.__name__
        wrapper.name = fn.__name__

        return wrapper

    return construct
