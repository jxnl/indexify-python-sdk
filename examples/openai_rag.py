from indexify import ExtractionGraph, IndexifyClient


def get_context(question: str, index: str, top_k=3):
    results = client.search_index(name=index, query=question, top_k=top_k)
    context = ""
    for result in results:
        context = (
            context
            + f"content id: {result['content_id']} \n\n passage: {result['text']}\n"
        )
    return context


def create_prompt(question, context):
    return f"Answer the question, based on the context.\n question: {question} \n context: {context}"


# Initialize Indexify Client
client = IndexifyClient()

# Create Extraction Graph from YAML
extraction_graph_spec = """
name: 'nbakb'
extraction_policies:
  - extractor: 'tensorlake/chunk-extractor'
    name: 'chunker'
    input_params:
        chunk_size: 1000
        overlap: 100
  - extractor: 'tensorlake/minilm-l6'
    name: 'wikiembedding'
    content_source: 'chunker'
"""
extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
client.create_extraction_graph(extraction_graph)


# Query Wikipedia
from langchain_community.document_loaders import WikipediaLoader

docs = WikipediaLoader(query="Kevin Durant", load_max_docs=1).load()
for doc in docs:
    client.add_documents("nbakb", doc.page_content)


# Perform Rag
from openai import OpenAI

client_openai = OpenAI()

question = "When and where did Kevin Durant win NBA championships?"
context = get_context(question, "nbakb.wikiembedding.embedding")
prompt = create_prompt(question, context)

chat_completion = client_openai.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-3.5-turbo",
)
print(chat_completion.choices[0].message.content)
