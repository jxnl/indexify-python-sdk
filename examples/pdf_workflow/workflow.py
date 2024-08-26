import io
import base64
from typing import List, Optional, Union

import httpx
from pydantic import BaseModel, Base64Bytes

from indexify.extractor_sdk.data import BaseData, Feature, File
from indexify.extractor_sdk.extractor import Extractor, extractor
from indexify.extractors.pdf_parser import Page, PDFParser, PageFragmentType
from indexify.graph import Graph
from indexify.local_runner import LocalRunner


@extractor()
def download_pdf(url: str) -> File:
    """
    Download pdf from url
    """
    resp = httpx.get(url=url, follow_redirects=True)
    resp.raise_for_status()
    return File(data=base64.b64encode(resp.content), mime_type="application/pdf")


class Document(BaseModel):
    pages: List[Page]


@extractor()
def parse_pdf(file: File) -> Document:
    """
    Parse pdf file and returns pages:
    """
    parser = PDFParser(base64.b64decode(file.data))
    pages: List[Page] = parser.parse()
    return Document(pages=pages)


class TextChunk(BaseData):
    chunk: str
    metadata: dict = {}
    embeddings: Optional[List[float]] = None


class ChunkParams(BaseModel):
    chunk_size: int


@extractor()
def extract_chunks(
    document: Document, chunk_size: int, overlap: int
) -> List[TextChunk]:
    """
    Extract chunks from document
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, overlap=overlap
    )
    chunks: List[TextChunk] = []
    for page in document.pages:
        for fragment in page.fragments:
            if fragment.fragment_type == PageFragmentType.TEXT:
                texts = text_splitter.split_text(fragment.text)
                for text in texts:
                    chunks.append(
                        TextChunk(chunk=text, metadata={"page_number": page.number})
                    )

    return chunks


class ImageDescription(BaseModel):
    description: str
    page_number: int
    figure_number: int


@extractor()
def describe_images(document: Document) -> List[ImageDescription]:
    """
    Describe images in document
    """
    descriptions = []
    return descriptions


class TextEmbeddingExtractor(Extractor):
    name = "text-embedding"
    description = "Extractor class that captures an embedding model"
    system_dependencies = []
    input_mime_types = ["text"]

    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def extract(self, input: TextChunk) -> List[float]:
        embeddings = self.model.encode(input.chunk)
        return embeddings.tolist()


class ImageWithEmbedding(BaseModel):
    embedding: List[float]
    page_number: int
    figure_number: int


class ImageEmbeddingExtractor(Extractor):
    name = "image-embedding"
    description = "Extractor class that captures an embedding model"
    system_dependencies = []
    input_mime_types = ["text"]

    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("clip-ViT-B-32")

    def extract(self, document: Document) -> List[ImageWithEmbedding]:
        from PIL import Image

        embedding = []
        for page in document.pages:
            for fragment in page.fragments:
                if fragment.fragment_type == "image":
                    image = fragment.image
                    img_emb = self.model.encode(Image.open(io.BytesArray(image.data)))
                    embedding.append(
                        ImageWithEmbedding(
                            embedding=img_emb,
                            page_number=page.number,
                            figure_number=fragment.number,
                        )
                    )
        return embedding


class LancedDBWriter(Extractor):
    python_dependencies = ["lancedb", "pyarrow"]
    def __init__(self):
        super().__init__()
        import lancedb
        import pyarrow as pa

        self._client = lancedb.connect("vectordb.lance")
        text_emb_schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), list_size=384)),
                pa.field("document_id", pa.string()),
                pa.field("page_number", pa.int32()),
            ]
        )

        img_emb_schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), list_size=512)),
                pa.field("document_id", pa.string()),
                pa.field("page_number", pa.int32()),
            ]
        )
        self._text_emb_table = self._client.create_table(
                "text_embeddings", schema=text_emb_schema, exist_ok=True
            )
        self._img_emb_table = self._client.create_table(
                "img_embeddings", schema=img_emb_schema, exist_ok=True
        )

    def extract(
        self, input: Union[ImageWithEmbedding, TextChunk]
    ) -> None:
        if type(input) == ImageWithEmbedding:
            self._img_emb_table.write(
                {
                    "img_embeddings": input.embedding,
                    "document_id": "document_id",
                    "page_number": input.page_number,
                }
            )
        elif type(input) == TextChunk:
            self._text_emb_table.write(
                {
                    "text_embeddings": input.embeddings,
                    "document_id": "document_id",
                    "page_number": input.metadata["page_number"],
                }
            )

def build_graph():
    data: File = download_pdf(url="https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf")
    document = parse_pdf(data)
    chunks = extract_chunks(document, 1000, 200)
    text_embeddings = text_embedding(chunks)
    image_embeddings = clip_embedding(document)
    lancedb_writer(text_embeddings + image_embeddings)

if __name__ == "__main__":
    g = Graph(
        "Extract pages, tables, images from pdf",
        start_node=download_pdf,
    )

    clip_embedding = ImageEmbeddingExtractor()
    text_embedding = TextEmbeddingExtractor()
    write_to_vector_db = LancedDBWriter()

    # Parse the PDF which was downloaded
    g.add_edge(download_pdf, parse_pdf)

    g.add_edge(parse_pdf, extract_chunks)

    g.add_edge(parse_pdf, describe_images)
    # Embed all the images in the PDF 
    g.add_edge(parse_pdf, clip_embedding)

    # Embed all the text chunks in the PDF
    g.add_edge(extract_chunks, text_embedding)

    # Describe all the images in the PDF
    g.add_edge(describe_images, text_embedding)

    # Write all the embeddings to the vector database
    g.add_edge(text_embedding, write_to_vector_db)
    g.add_edge(clip_embedding, write_to_vector_db)

    local_runner = LocalRunner()
    local_runner.run(g, url="https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf")
