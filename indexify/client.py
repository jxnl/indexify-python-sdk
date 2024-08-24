import hashlib
import json
import logging
import uuid
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import httpx
import yaml

from indexify.exceptions import ApiException

from .data_loaders import DataLoader
from .error import Error
from .extraction_policy import ExtractionGraph
from .extractor_sdk.data import ContentMetadata
from .extractor_sdk.extractor import ExtractorMetadata
from .settings import DEFAULT_SERVICE_URL, DEFAULT_SERVICE_URL_HTTPS
from .utils import json_set_default

Document = namedtuple("Document", ["text", "labels", "id"])

SQLQueryRow = namedtuple("SQLQueryRow", ["content_id", "data"])


def generate_unique_hex_id():
    """
    Generate a unique hexadecimal identifier

    Returns:
        str: a unique hexadecimal string
    """
    return uuid.uuid4().hex[:16]


def generate_hash_from_string(input_string: str):
    """
    Generate a hash for the given string and return it as a hexadecimal string.

    Args:
        input_string (str): The input string to hash.

    Returns:
        str: The hexadecimal hash of the input string.
    """
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()[:16]


@dataclass
class SqlQueryResult:
    result: List[Dict]


class IndexifyClient:
    """
    IndexifyClient is the main entry point for the SDK.
    For the full list of client features, see the
    [httpx Client documentation](https://www.python-httpx.org/api/#client).

    :param service_url: The URL of the Indexify service to connect to.
    :param args: Arguments to pass to the httpx.Client constructor
    :param kwargs: Keyword arguments to pass to the httpx.Client constructor

    Example usage:
    ```
    from indexify import IndexifyClient

    client = IndexifyClient()
    assert client.heartbeat() == True
    ```
    """

    def __init__(
        self,
        service_url: str = DEFAULT_SERVICE_URL,  # switch this to DEFAULT_SERVICE_URL_HTTPS for TLS
        namespace: str = "default",
        config_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if config_path:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            if config.get("use_tls", False):
                tls_config = config["tls_config"]
                self._client = httpx.Client(
                    http2=True,
                    cert=(tls_config["cert_path"], tls_config["key_path"]),
                    verify=tls_config.get("ca_bundle_path", True),
                )
            else:
                self._client = httpx.Client(*args, **kwargs)
        else:
            self._client = httpx.Client(*args, **kwargs)

        self.namespace: str = namespace
        self.extraction_graphs: List[ExtractionGraph] = []
        self.labels: dict = {}
        self._service_url = service_url
        self._timeout = kwargs.get("timeout")

        # get namespace data
        self.extraction_graphs = self.get_extraction_graphs()

    @classmethod
    def with_mtls(
        cls,
        cert_path: str,
        key_path: str,
        ca_bundle_path: Optional[str] = None,
        service_url: str = DEFAULT_SERVICE_URL_HTTPS,
        *args,
        **kwargs,
    ) -> "IndexifyClient":
        """
        Create a client with mutual TLS authentication. Also enables HTTP/2,
        which is required for mTLS.
        NOTE: mTLS must be enabled on the Indexify service for this to work.

        :param cert_path: Path to the client certificate. Resolution handled by httpx.
        :param key_path: Path to the client key. Resolution handled by httpx.
        :param args: Arguments to pass to the httpx.Client constructor
        :param kwargs: Keyword arguments to pass to the httpx.Client constructor
        :return: A client with mTLS authentication

        Example usage:
        ```
        from indexify import IndexifyClient

        client = IndexifyClient.with_mtls(
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
        )
        assert client.heartbeat() == True
        ```
        """
        if not (cert_path and key_path):
            raise ValueError("Both cert and key must be provided for mTLS")

        client_certs = (cert_path, key_path)
        verify_option = ca_bundle_path if ca_bundle_path else True
        client = IndexifyClient(
            *args,
            **kwargs,
            service_url=service_url,
            http2=True,
            cert=client_certs,
            verify=verify_option,
        )
        return client

    def _request(self, method: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, timeout=self._timeout, **kwargs)
            status_code = str(response.status_code)
            if status_code.startswith("4"):
                raise ApiException(
                    "status code: " + status_code + " request args: " + str(kwargs)
                )
            if status_code.startswith("5"):
                raise ApiException(response.text)
                # error = Error.from_tonic_error_string(str(response.url), response.text)
                # self.__print_additional_error_context(error)
                # raise error
        except httpx.ConnectError:
            message = (
                f"Make sure the server is running and accesible at {self._service_url}"
            )
            error = Error(status="ConnectionError", message=message)
            print(error)
            raise error
        return response

    def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """
        Make a GET request to the Indexify service.

        :param endpoint: The endpoint to make the request to.

        Example usage:
        ```
        from indexify import IndexifyClient

        client = IndexifyClient()
        response = client.get("namespaces")
        print(response.json())
        ```
        """
        return self._request("GET", url=f"{self._service_url}/{endpoint}", **kwargs)

    def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """
        Make a POST request to the Indexify service.

        :param endpoint: The endpoint to make the request to.

        Example usage:

        ```
        from indexify import IndexifyClient

        client = IndexifyClient()
        response = client.post("namespaces", json={"name": "my-repo"})
        print(response.json())
        ```
        """
        return self._request("POST", url=f"{self._service_url}/{endpoint}", **kwargs)

    def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """
        Make a PUT request to the Indexify service.

        :param endpoint: The endpoint to make the request to.

        Example usage:

        ```
        from indexify import IndexifyClient

        client = IndexifyClient()
        response = client.put("namespaces", json={"name": "my-repo"})
        print(response.json())
        ```
        """
        return self._request("PUT", url=f"{self._service_url}/{endpoint}", **kwargs)

    def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """
        Make a DELETE request to the Indexify service.

        :param endpoint: The endpoint to make the request to.

        Example usage:

        ```
        from indexify import IndexifyClient

        client = IndexifyClient()
        response = client.delete("namespaces")
        print(response.json())
        ```
        """
        return self._request("DELETE", url=f"{self._service_url}/{endpoint}", **kwargs)

    def close(self):
        """
        Close the underlying httpx.Client.
        """
        self._client.close()

    # __enter__ and __exit__ allow the client to be used as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def heartbeat(self, heartbeat_response="Indexify Server") -> bool:
        """
        Check if the Indexify service is alive.
        """
        response = self.get(f"")
        # Server responds with text: "Indexify Server"
        return response.text == heartbeat_response

    def namespaces(self) -> list[str]:
        """
        Get a list of all namespaces.
        """
        response = self.get(f"namespaces")
        namespaces_dict = response.json()["namespaces"]
        namespaces = []
        for item in namespaces_dict:
            namespaces.append(item["name"])
        return namespaces

    @classmethod
    def create_namespace(
        self,
        namespace: str,
        extraction_graphs: list = [],
        labels: dict = {},
        service_url: str = DEFAULT_SERVICE_URL,
    ) -> "IndexifyClient":
        """
        Create a new namespace.

        Returns:
            IndexifyClient: a new client with the given namespace
        """
        extraction_graphs = []
        for bd in extraction_graphs:
            if isinstance(bd, extraction_graphs):
                extraction_graphs.append(bd.to_dict())
            else:
                extraction_graphs.append(bd)

        req = {
            "name": namespace,
            "extraction_graphs": extraction_graphs,
            "labels": labels,
        }

        with httpx.Client() as client:
            client.post(f"{service_url}/namespaces", json=req)

        client = IndexifyClient(namespace=namespace, service_url=service_url)
        return client

    def _add_content_url(self, content):
        """
        Add download content_url url property
        """
        return {
            **content,
            "content_url": f"{self._service_url}/namespaces/{self.namespace}/content/{content['id']}/download",
        }

    def indexes(self) -> dict:
        """
        Get the indexes of the current namespace.

        Returns:
            List[Index]: list of indexes in the current namespace
        """
        response = self.get(f"namespaces/{self.namespace}/indexes")
        return response.json()["indexes"]

    def extractors(self) -> List[ExtractorMetadata]:
        """
        Get a list of all extractors.

        Returns:
            List[Extractor]: list of extractors
        """
        response = self.get(f"extractors")
        extractors_dict = response.json()["extractors"]
        extractors = []
        for ed in extractors_dict:
            print(ed)
            extractors.append(ExtractorMetadata.model_validate(ed))
        return extractors

    def get_extraction_graphs(self) -> List[ExtractionGraph]:
        """
        Retrieve and update the list of extraction policies for the current namespace.
        """
        response = self.get(f"namespaces/{self.namespace}/extraction_graphs")
        json = response.json()

        self.extraction_graphs = []
        for graph in json["extraction_graphs"]:
            self.extraction_graphs.append(ExtractionGraph.from_dict(graph))

        return self.extraction_graphs

    def create_extraction_graph(self, extraction_graph: ExtractionGraph):
        """
        Create a new extraction graph.

        Args:
            - extraction_graph (ExtractionGraph): the extraction graph to create
        """
        req = extraction_graph.to_dict()
        req["namespace"] = self.namespace
        request_body = json.dumps(req, default=json_set_default)
        response = self.post(
            f"namespaces/{self.namespace}/extraction_graphs",
            data=request_body,
            headers={"Content-Type": "application/json"},
        )
        return

    def link_extraction_graphs(
        self, source_graph: str, content_source: str, linked_graph: str
    ):
        """
        Link an extraction graph to another extraction graph.

        Args:
            - source_graph (str): source extraction graph
            - content_source (str): content source in source graph
            - linked_graph (str): target extraction graph
        """
        req = {
            "content_source": content_source,
            "linked_graph_name": linked_graph,
        }
        response = self.post(
            f"namespaces/{self.namespace}/extraction_graphs/{source_graph}/links",
            json=req,
            headers={"Content-Type": "application/json"},
        )
        return

    def get_content_metadata(self, content_id: str) -> dict:
        """
        Get metadata for a specific content ID in a given index.

        Args:
            - content_id (str): content id to query
        """
        response = self.get(
            f"namespaces/{self.namespace}/content/{content_id}/metadata"
        )
        return response.json()["content_metadata"]

    def download_content(self, content_id: str) -> bytes:
        """
        Download content from id. Return bytes

        Args:
            - content_id (str): id of content to download
        """
        response = self.get(
            f"namespaces/{self.namespace}/content/{content_id}/download"
        )
        return response.content

    def add_documents(
        self,
        extraction_graphs: Union[str, List[str]],
        documents: Union[Document, str, List[Union[Document, str]]],
        doc_id=None,
    ) -> Union[str, List[str]]:
        """
        Add documents to current namespace.

        Args:
            - documents (Union[Document, str, List[Union[Document, str]]]): this can be a list of strings, list of Documents or a mix of both
        """
        if isinstance(extraction_graphs, str):
            extraction_graphs = [extraction_graphs]
        if isinstance(documents, Document):
            documents = [documents]
        elif isinstance(documents, str):
            documents = [Document(documents, {}, id=doc_id)]
        elif isinstance(documents, list):
            new_documents = []
            for item in documents:
                if isinstance(item, Document):
                    new_documents.append(item)
                elif isinstance(item, str):
                    new_documents.append(
                        Document(item, {}, id=None)
                    )  # don't pass in id for a string content because doesn't make sense to have same content id for all strings
                else:
                    raise ValueError(
                        "List items must be either Document instances or strings."
                    )
            documents = new_documents
        else:
            raise TypeError(
                "Invalid type for documents. Expected Document, str, or list of these."
            )
        for document in documents:
            document.labels["mime_type"] = "text/plain"
        content_ids = []
        if isinstance(extraction_graphs, str):
            extraction_graphs = [extraction_graphs]
        for extraction_graph in extraction_graphs:
            for document in documents:
                response = self.post(
                    f"namespaces/{self.namespace}/extraction_graphs/{extraction_graph}/extract",
                    files={"file": document.text},
                    data={"labels": json.dumps(document.labels)},
                )
                response_json = response.json()
                content_id = response_json["content_id"]
                content_ids.append(content_id)
        return content_ids

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from current namespace.

        Args:
            - document_ids (List[str]): list of document ids to delete
        """
        req = {"content_ids": document_ids}
        response = self.delete(
            f"namespaces/{self.namespace}/content",
            json=req,
            headers={"Content-Type": "application/json"},
        )

    def update_labels(self, document_id: str, labels: Dict[str, str]) -> None:
        """
        Update labels for a document.

        Args:
            - document_id (str): id of document to update
            - labels (Dict[str, str]): labels to update
        """
        req = {"labels": labels}
        response = self.put(
            f"namespaces/{self.namespace}/content/{document_id}/labels",
            json=req,
            headers={"Content-Type": "application/json"},
        )

    def update_content(self, document_id: str, path: str) -> None:
        """
        Update a piece of content with a new file

        Args:
            - path (str): relative path to the file to be uploaded
        """
        with open(path, "rb") as f:
            response = self.put(
                f"namespaces/{self.namespace}/content/{document_id}", files={"file": f}
            )

    def get_structured_data(self, content_id: str) -> dict:
        """
        Query metadata for a specific content ID in a given index.

        Args:
            - content_id (str): content id to query
        """
        response = self.get(
            f"namespaces/{self.namespace}/content/{content_id}/metadata"
        )
        return response.json().get("metadata", [])

    def search_index(
        self, name: str, query: str, top_k: int, filters: List[str] = []
    ) -> dict:
        """
        Search index in the current namespace.

        Args:
            - name (str): name of index to search
            - query (str): query string
            - top_k (int): top k nearest neighbors to be returned
            - filters (List[str]): list of filters to apply
        """
        req = {"query": query, "k": top_k, "filters": filters}
        response = self.post(
            f"namespaces/{self.namespace}/indexes/{name}/search",
            json=req,
            headers={"Content-Type": "application/json"},
        )
        return response.json()["results"]

    def list_content(
        self,
        extraction_graph: str,
        extraction_policy: str = "",
        labels_filter: List[str] = [],
        start_id: str = "",
        limit: int = 10,
    ) -> List[ContentMetadata]:
        """
        List content in the current namespace.

        Args:
            - extraction_graph (str): extraction graph name
            - start_index (str): start index for pagination
            - limit (int): number of items to return
        """
        params = {"start_id": start_id, "limit": limit}
        if extraction_policy:
            params["source"] = extraction_policy
        else:
            params["source"] = "ingestion"
        if len(labels_filter) > 0:
            params["labels_filter"] = labels_filter
        response = self.get(
            f"namespaces/{self.namespace}/extraction_graphs/{extraction_graph}/content",
            params=params,
        )
        content_list = response.json()["content_list"]
        content = []
        for item in content_list:
            content.append(ContentMetadata.from_dict(item))
        return content

    def upload_file(
        self,
        extraction_graph: str,
        path: str,
        file_bytes: bytes = None,
        id=None,
        labels: dict = {},
    ) -> str:
        """
        Upload a file from a path or the bytes.

        Args:
            - extraction_graph (str): name of the extraction graph to use for extraction
            - path (Union[str, bytes]): relative path to the file to be uploaded, or the bytes of the file
            - labels (dict): labels to be associated with the file
        """
        params = {}
        if id is not None:
            params["id"] = id

        if file_bytes == None:
            with open(path, "rb") as f:
                response = self.post(
                    f"namespaces/{self.namespace}/extraction_graphs/{extraction_graph}/extract",
                    files={"file": f},
                    data={"labels": json.dumps(labels)},
                    params=params,
                )
        else:
            response = self.post(
                f"namespaces/{self.namespace}/extraction_graphs/{extraction_graph}/extract",
                files={"file": (path, file_bytes)},
                data={"labels": json.dumps(labels)},
                params=params,
            )
            file_content = path

        response_json = response.json()
        content_id = response_json["content_id"]
        return content_id

    def ingest_from_loader(
        self, loader: DataLoader, extraction_graph: str
    ) -> List[str]:
        """
        Loads content using the loader, uploads them to Indexify and returns the content ids.
        loader: DataLoader: The DataLoader object to use for loading content
        extraction_graph: str: The name of the extraction graph to use for extraction
        """
        content_ids = []
        files = loader.load()
        for file_metadata in files:
            labels = {"file_name": file_metadata.path}
            content_id = self.upload_file(
                extraction_graph,
                file_metadata.path,
                loader.read_all_bytes(file_metadata),
                labels=labels,
            )
            content_ids.append(content_id)
        return content_ids

    def list_schemas(self) -> List[str]:
        """
        List all schemas in the current namespace.
        """
        response = self.get(f"namespaces/{self.namespace}/schemas")
        return response.json()

    def get_extracted_content(
        self,
        ingested_content_id: str,
        graph_name: str,
        policy_name: str,
        blocking=False,
    ):
        """
        Get list of child for a given content id and their content up to the specified level.

        Args:
        - ingested_content_id (str): id of content
        - graph_name (str): name of extraction graph
        - policy_name(str): name of extraction policy in the graph
        - blocking (bool): wait for extraction to complete before returning (default: False)
        """
        if blocking:
            self.wait_for_extraction(ingested_content_id)
        response = self.get(
            f"namespaces/{self.namespace}/extraction_graphs/{graph_name}/content/{ingested_content_id}/extraction_policies/{policy_name}"
        )
        content_tree = response.json()
        child_list = []
        for item in content_tree["content_tree_metadata"]:
            if (
                graph_name in item["extraction_graph_names"]
                and item["source"] == policy_name
            ):
                content = self.download_content(item["id"])
                child_list.append(
                    {
                        "id": item["id"],
                        "mime_type": item["mime_type"],
                        "content": content,
                    }
                )

        return child_list

    def sql_query(self, query: str):
        """
        Execute a SQL query.

        Args:
            - query (str): SQL query to be executed
        """
        req = {"query": query}
        response = self.post(
            f"namespaces/{self.namespace}/sql_query",
            json=req,
            headers={"Content-Type": "application/json"},
        )
        result = response.json()
        rows = []
        for row in result["rows"]:
            data = row["data"]
            rows.append(data)
        return SqlQueryResult(result=rows)

    def ingest_remote_file(
        self,
        extraction_graph: str,
        url: str,
        mime_type: str,
        labels: Dict[str, str] = {},
        id=None,
    ):
        req = {
            "url": url,
            "mime_type": mime_type,
            "labels": labels,
            "id": id,
        }
        response = self.post(
            f"namespaces/{self.namespace}/extraction_graphs/{extraction_graph}/extract_remote",
            json=req,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()["content_id"]

    def wait_for_extraction(self, content_ids: Union[str, List[str]]):
        """
        Wait for extraction to complete for a given content id

        Args:
            - content_id (str): id of content
        """
        if type(content_ids) == str:
            content_ids = [content_ids]
        print(
            "Waiting for extraction to complete for content id: ", ",".join(content_ids)
        )
        for content_id in content_ids:
            response = self.get(
                f"namespaces/{self.namespace}/content/{content_id}/wait"
            )
            print("Extraction completed for content id: ", content_id)
        response.raise_for_status()

    def generate_unique_hex_id(self):
        """
        Generate a unique hexadecimal identifier

        Returns:
            str: a unique hexadecimal string
        """
        logging.warning(
            "This method is deprecated. Use generate_unique_hex_id from indexify instead."
        )
        return uuid.uuid4().hex[:16]

    def generate_hash_from_string(self, input_string: str):
        """
        Generate a hash for the given string and return it as a hexadecimal string.

        Args:
            input_string (str): The input string to hash.

        Returns:
            str: The hexadecimal hash of the input string.
        """
        logging.warning(
            "This method is deprecated. Use generate_hash_from_string from indexify instead."
        )
        hash_object = hashlib.sha256(input_string.encode())
        return hash_object.hexdigest()[:16]

    def __print_additional_error_context(self, error: Error):
        print(error)

        if error.status == "ExtractionGraphError":
            graphs = [eg.name for eg in self.extraction_graphs]
            extractors = [ext.name for ext in self.extractors()]
            print(f"Available extraction graphs: {graphs}")
            print(f"Available extractors: {extractors}")

        if error.status == "SearchError":
            indexes = [index["name"] for index in self.indexes()]
            print(f"Available indexes: {indexes}")
