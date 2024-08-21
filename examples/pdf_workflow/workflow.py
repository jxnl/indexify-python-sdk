from typing import List
import json

import httpx

from indexify.extractor_sdk.extractor import extractor, Extractor
from indexify.extractor_sdk.data import BaseData, PDFFile

from sentence_transformers import SentenceTransformer

from indexify.graph import Graph
from tt_module import get_tables
import chromadb
import pymupdf
import fitz

from pydantic import BaseModel


@extractor(description="Download pdf")
def download_pdf(url: str) -> PDFFile:
    import urllib.request
    filename, _ = urllib.request.urlretrieve(url)

    resp = httpx.request(url=url, method="GET")

    with open(filename, 'rb') as f:
        output = PDFFile(data=resp.content, mime_type="application/pdf")

    return output


class PageText(BaseData):
    text: str

    page_num: int


@extractor(description="Extract page text from pdf")
def extract_page_text(pdf_file: PDFFile) -> List[PageText]:
    output = []
    with pymupdf.open("pdf", pdf_file.data) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            output.append(PageText(text=text, page_num=page_num))

    return output

class PageImage(BaseData):
    content_type: str
    data: bytes

    page_num: int
    img_num: int

@extractor(description="Extract image from pdf")
def extract_images(pdf_file: PDFFile) -> List[PageImage]:
    output = []

    doc = fitz.open("pdf", pdf_file.data)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            output.append(PageImage(content_type="image/png", data=image_bytes, page_num=page_num, img_num=img_index))

    return output


class PageTable(BaseData):
    data: bytes

    page_num: int

@extractor(description="Extract table from pdf")
def extract_tables(pdf_file: PDFFile) -> List[PageTable]:
    output = []

    tables = get_tables(pdf_path=pdf_file.data)
    for page_num, content in tables.items():
        output.append(PageTable(data=json.dumps(content), page_num=page_num))

    return output


class TextChunk(BaseData):
    chunk: str

class ChunkParams(BaseModel):
    chunk_size: int

@extractor(description="Make chunks")
def make_chunks(page_text: PageText, params: ChunkParams= None) -> List[TextChunk]:
    text = page_text.text
    chunk_len = params['chunk_size']
    chunk_size = len(text) // chunk_len
    chunks = []
    for i in range(chunk_size+1):
        s = i*chunk_len
        if len(text[s:s+chunk_len]) > 0:  # sentence bert crashes for empty input
            chunks.append(TextChunk(chunk=text[s:s+chunk_len]))

    return chunks


class Embedding(BaseData):
    embedding: List[float]

    text: str


class EmbeddingExtractor(Extractor):
    name = "temp/embedding"
    description = "Extractor class that captures an embedding model"
    system_dependencies = []
    input_mime_types = ["text"]

    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def extract(self, input: TextChunk, params: BaseModel=None) -> List[Embedding]:
        text = input.chunk

        embeddings = self.model.encode([text])

        embeddings = [Embedding(embedding=i, text=j) for i, j in zip(embeddings, [text])]

        return embeddings

    @classmethod
    def sample_input(cls) -> TextChunk:
        return TextChunk(chunk="this is some sample chunked text")


class VectorStoreParams(BaseModel):
    collection_name: str


@extractor(description="Write to vector store")
def write_to_vector_store(
    em: Embedding,
    params: VectorStoreParams = None
):
    collection = chroma_client.get_or_create_collection(name=params["collection_name"])
    print(dir(em))
    collection.add(
        embeddings = em.embedding,
        documents = em.text
    )

if __name__ == "__main__":
    # For this example we are setting up a local chroma instance to save the
    # embeddings. We will use this client in the main method to query.

    TEST_COLLECTION_NAME = "test-collection"
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name=TEST_COLLECTION_NAME)

    g = Graph("Extract pages, tables, images from pdf", input=str, start_node=download_pdf, run_local=True)

    g.add_edge(download_pdf, extract_page_text)
    g.add_edge(download_pdf, extract_images)
    g.add_edge(download_pdf, extract_tables)

    g.add_edge(extract_page_text, make_chunks)
    g.add_edge(make_chunks, EmbeddingExtractor)

    g.add_edge(EmbeddingExtractor, write_to_vector_store)

    g.add_param(make_chunks, {"chunk_size": 2000})

    g.add_param(write_to_vector_store, {"collection_name": TEST_COLLECTION_NAME})

    # Clear caches if required
    # g.clear_cache_for_node(extract_tables)
    # g.clear_cache_for_all_nodes()

    url = "https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf"
    g.run(wf_input=url, local=True)

    # These outputs will confirm the executor outputs.
    print(f"number of pages {len(g.get_result(extract_page_text))}")
    print(f"number of images {len(g.get_result(extract_images))}")
    print(f"number of tables {len(g.get_result(extract_tables))}")
    print(f"number of embeddings {len(g.get_result(EmbeddingExtractor))}")

    # Uncomment these lines for sanity check on the output
    # print('\n---- Text output')
    # print(g.get_result(extract_page_text)[3])
    # print('---- /Text output\n')
    #
    # print('\n---- Embedding output')
    # print(g.get_result(EmbeddingExtractor)[3])
    # print('---- /Embedding output\n')


    results = collection.query(
        query_texts=["this is a test"],
        n_results=2
    )
    print(results)
