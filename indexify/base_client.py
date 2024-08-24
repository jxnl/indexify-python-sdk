from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .extractor_sdk import Feature, Graph


class BaseClient(ABC):

    ### Operational APIs
    @abstractmethod
    def register_extraction_graph(self, graph: Graph):
        pass

    @abstractmethod
    def graphs(self) -> str:
        pass

    @abstractmethod
    def namespaces(self) -> str:
        pass

    @abstractmethod
    def create_namespace(self, namespace: str):
        pass

    ### Ingestion APIs
    @abstractmethod
    def invoke_graph_with_object(self, graph: str, object: Any) -> str:
        """
        Invokes a graph with an input object.
        graph: str: The name of the graph to invoke
        object: Any: The input object to the graph. It should be JSON serializable
        return: str: The ID of the ingested object
        """
        pass

    @abstractmethod
    def invoke_graph_with_file(self, graph: str, path: str) -> str:
        """
        Invokes a graph with an input file. The file's mimetype is appropriately detected.
        graph: str: The name of the graph to invoke
        path: str: The path to the file to be ingested
        return: str: The ID of the ingested object
        """
        pass

    ### Retrieval APIs
    @abstractmethod
    def extracted_objects(
        self, graph: str, ingested_object_id: str, extractor_name: Optional[str]
    ) -> Union[Dict[str, List[Any]], List[Any]]:
        """
        Returns the extracted objects by a graph for an ingested object. If the extractor name is provided, only the objects extracted by that extractor are returned.
        If the extractor name is not provided, all the extracted objects are returned for the input object.
        graph: str: The name of the graph
        ingested_object_id: str: The ID of the ingested object
        extractor_name: Optional[str]: The name of the extractor whose output is to be returned if provided
        return: Union[Dict[str, List[Any]], List[Any]]: The extracted objects. If the extractor name is provided, the output is a list of extracted objects by the extractor. If the extractor name is not provided, the output is a dictionary with the extractor name as the key and the extracted objects as the value. If no objects are found, an empty list is returned.
        """
        pass

    @abstractmethod
    def features(
        self, object_id: str, graph: Optional[str]
    ) -> Union[Dict[str, List[Feature]], List[Feature]]:
        """
        Returns the features of an object.
        object_id: str: The ID of the object
        return: List[Feature]: The features associated with the object that were extracted. If a graph name is provided, only the features extracted by that graph are returned.
        """
        pass
