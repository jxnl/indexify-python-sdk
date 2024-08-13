from indexify.client import IndexifyClient, Document, ExtractionGraph
from indexify.extraction_policy import ExtractionPolicy
import time
import os
import unittest
import uuid
from httpx import ConnectError

class TestIntegrationTest(unittest.TestCase):
    """
    Must have wikipedia and minilml6 extractors running
    """

    def __init__(self, *args, **kwargs):
        super(TestIntegrationTest, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.client = IndexifyClient()

    def generate_short_id(self, size: int = 4) -> str:
        return uuid.uuid4().__str__().replace("-", "")[:size]

    def test_list_namespaces(self):
        client = IndexifyClient()
        namespaces = client.namespaces()
        assert len(namespaces) >= 1

    def test_get_namespace(self):
        namespace = "default"
        client = IndexifyClient(namespace=namespace)
        assert client.namespace == namespace

    def test_create_namespace(self):
        namespace_name = "test.createnamespace"

        minilm_binding = ExtractionPolicy(
            extractor="tensorlake/minilm-l6",
            name="minilm-l6",
            content_source="source",
            input_params={},
        )

        client = IndexifyClient.create_namespace(
            namespace_name, extraction_policies=[minilm_binding]
        )
        assert client.namespace == namespace_name

    def test_add_documents(self):
        # Add single documents
        namespace_name = "test.adddocuments"
        client = IndexifyClient.create_namespace(namespace_name)

        extraction_graph_spec = """
        name: 'test_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: 'test_policy'
        """

        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)

        client.add_documents(
            "test_graph",
            Document(
                text="This is a test",
                labels={"source": "test"},
                id=None
            )
        )

        # Add multiple documents
        client.add_documents(
            "test_graph",
            [
                Document(
                    text="This is a new test",
                    labels={"source": "test"},
                    id=None
                ),
                Document(
                    text="This is another test",
                    labels={"source": "test"},
                    id=None,
                ),
            ]
        )

        # Add single string
        client.add_documents("test_graph", "test", doc_id=None)

        # Add multiple strings
        client.add_documents("test_graph", ["one", "two", "three"], doc_id=None)

        # Add mixed
        client.add_documents("test_graph", ["string", Document("document string", {}, id=None)], doc_id=None)

    def test_get_content(self):
        namespace_name = "test.getcontent"
        client = IndexifyClient.create_namespace(namespace=namespace_name)
        
        extraction_graph_spec = """
        name: 'test_get_content_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: 'content_policy'
        """

        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)

        client.add_documents(
            "test_get_content_graph",
            [Document(text="one", labels={"l1": "test"}, id=None), "two", "three"]
        )
        content = client.get_extracted_content()
        assert len(content) == 3
        # validate content_url
        for c in content:
            assert c.get("content_url") is not None

        # parent doesn't exist
        content = client.get_extracted_content(content_id="idontexist")
        assert len(content) == 0

    def test_download_content(self):
        namespace_name = "test.downloadcontent"
        client = IndexifyClient.create_namespace(namespace=namespace_name)
        
        extraction_graph_spec = """
        name: 'test_download_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: 'download_policy'
        """

        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)

        client.add_documents(
            "test_download_graph",
            ["test download"],
            doc_id=None
        )
        content = client.get_extracted_content()
        assert len(content) == 1

        data = client.download_content(content[0].get('id'))
        assert data.decode("utf-8") == "test download"

    def test_search(self):
        namespace_name = self.generate_short_id()
        extractor_name = self.generate_short_id()

        client = IndexifyClient.create_namespace(namespace_name)
        source = "test"

        extraction_graph_spec = f"""
        name: '{extractor_name}_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: '{extractor_name}'
        """

        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)

        client.add_documents(
            f"{extractor_name}_graph",
            [
                Document(
                    text="Indexify is also a retrieval service for LLM agents!",
                    labels={"source": source},
                    id=None,
                )
            ]
        )
        time.sleep(10)
        results = client.search_index(f"{extractor_name}.embedding", "LLM", 1)
        assert len(results) == 1

    def test_list_extractors(self):
        extractors = self.client.extractors()
        assert len(extractors) >= 1

    def test_add_extraction_policy(self):
        name = "minilml6_test_add_extraction_policy"
        namespace_name = "test.bindextractor"
        client = IndexifyClient.create_namespace(namespace_name)
        
        extraction_graph_spec = f"""
        name: '{name}_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: '{name}'
        """
    
        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)

    def test_get_metadata(self):
        """
        need to add a new extractor which produce the metadata index
        wikipedia extractor would be that, would have metadata index
        use same way
        """

        namespace_name = "metadatatest"
        client = IndexifyClient.create_namespace(namespace_name)
        time.sleep(2)
        client.add_extraction_policy(
            "tensorlake/wikipedia",
            "wikipedia",
        )

        time.sleep(2)
        client.upload_file(
            os.path.join(
                os.path.dirname(__file__), "files", "steph_curry_wikipedia.html"
            ),
            id=None
        )
        time.sleep(25)
        content = client.get_extracted_content()
        content = list(filter(lambda x: x.get("source") != "ingestion", content))
        assert len(content) > 0
        for c in content:
            metadata = client.get_content_metadata(c.get("id"))
            assert len(metadata) > 0

    def test_extractor_input_params(self):
        name = "chunk_test_extractor_input_params"
        client = IndexifyClient.create_namespace(namespace="test.extractorinputparams")
        
        extraction_graph_spec = f"""
        name: '{name}_graph'
        extraction_policies:
          - extractor: 'tensorlake/chunk-extractor'
            name: '{name}'
            input_params:
              text_splitter: 'recursive'
              chunk_size: 1000
              overlap: 200
        """
    
        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)

    def test_get_bindings(self):
        name = "minilml6_test_get_bindings"
        client = IndexifyClient.create_namespace("test.getbindings")
        
        extraction_graph_spec = f"""
        name: '{name}_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: '{name}'
        """
    
        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)
        bindings = client.extraction_policies
        assert len(list(filter(lambda x: x.name.startswith(name), bindings))) == 1

    def test_get_indexes(self):
        name = "minilml6_test_get_indexes"
        client = IndexifyClient.create_namespace("test.getindexes")
        
        extraction_graph_spec = f"""
        name: '{name}_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: '{name}'
        """
    
        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)
        indexes = client.indexes()
        assert len(list(filter(lambda x: x.get("name").startswith(name), indexes))) == 1

    def test_upload_file(self):
        test_file_path = os.path.join(os.path.dirname(__file__), "files", "test.txt")
        
        extraction_graph_spec = """
        name: 'upload_file_graph'
        extraction_policies:
          - extractor: 'tensorlake/minilm-l6'
            name: 'upload_policy'
        """

        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        self.client.create_extraction_graph(extraction_graph)
        
        self.client.upload_file("upload_file_graph", test_file_path)

    def test_ingest_remote_url(self):
        url = "https://gif-search.diptanu-6d5.workers.dev/OlG-5EjOENZLvlxHcXXmT.gif"
        content_id = self.client.generate_hash_from_string(url)
        res = self.client.ingest_remote_file(url, "image/gif", {}, id=content_id)
        assert res.get("content_id") == content_id
    
    def test_timeout(self):
        with self.assertRaises(ConnectError):
            IndexifyClient(timeout=0)

        try:
            IndexifyClient(timeout=None)
        except Exception as e:
            self.fail(f"IndexifyClient raised an exception with timeout=None: {e}")

        try:
            IndexifyClient()
        except Exception as e:
            self.fail(f"IndexifyClient raised an exception with default timeout: {e}")


    # def test_langchain_retriever(self):
    #     # import langchain retriever
    #     from indexify_langchain import IndexifyRetriever

    #     # init client
    #     client = IndexifyClient.create_namespace("test-langchain")
    #     client.add_extraction_policy(
    #         "tensorlake/minilm-l6",
    #         "minilml6",
    #     )

    #     # Add Documents
    #     client.add_documents("Lucas is from Atlanta Georgia", doc_id=None)
    #     time.sleep(10)

    #     # Initialize retriever
    #     params = {"name": "minilml6.embedding", "top_k": 9}
    #     retriever = IndexifyRetriever(client=client, params=params)

    #     # Setup Chat Prompt Template
    #     from langchain.prompts import ChatPromptTemplate

    #     template = """You are an assistant for question-answering tasks. 
    #     Use the following pieces of retrieved context to answer the question. 
    #     If you don't know the answer, just say that you don't know. 
    #     Use three sentences maximum and keep the answer concise.
    #     Question: {question} 
    #     Context: {context} 
    #     Answer:
    #     """
    #     prompt = ChatPromptTemplate.from_template(template)

    #     # Ask llm question with retriever context
    #     from langchain_openai import ChatOpenAI
    #     from langchain.schema.runnable import RunnablePassthrough
    #     from langchain.schema.output_parser import StrOutputParser

    #     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    #     rag_chain = (
    #         {"context": retriever, "question": RunnablePassthrough()}
    #         | prompt
    #         | llm
    #         | StrOutputParser()
    #     )

    #     query = "Where is Lucas from?"
    #     assert "Atlanta" in rag_chain.invoke(query)

    # TODO: metadata not working outside default namespace
        
    def test_sql_query(self):        
        # namespace_name = "sqlquerytest"
        # client = IndexifyClient.create_namespace(namespace_name)
        client = IndexifyClient()
        time.sleep(2)
        client.add_extraction_policy(name="wikipedia", extractor="tensorlake/wikipedia")

        time.sleep(2)
        client.upload_file(
            os.path.join(
                os.path.dirname(__file__), "files", "steph_curry_wikipedia.html"
            )
        )
        time.sleep(25)

        query_result = client.sql_query("select * from ingestion")
        assert len(query_result.result) == 1

if __name__ == "__main__":
    unittest.main()
