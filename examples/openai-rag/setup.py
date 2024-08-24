from langchain_community.document_loaders import WikipediaLoader

from indexify import ExtractionGraph, IndexifyClient

client = IndexifyClient()


def create_extraction_graph():
    with open("graph.yaml", "r") as file:
        extraction_graph_spec = file.read()
        extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
        client.create_extraction_graph(extraction_graph)


def load_data():
    docs = WikipediaLoader(query="Kevin Durant", load_max_docs=20).load()

    for doc in docs:
        client.add_documents("summarize_and_chunk", doc.page_content)


if __name__ == "__main__":
    create_extraction_graph()
    load_data()
