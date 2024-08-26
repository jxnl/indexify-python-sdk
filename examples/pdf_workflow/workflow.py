import io
from typing import List, Optional, Union

import httpx
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from indexify.extractor_sdk.data import BaseData, Feature, File
from indexify.extractor_sdk.extractor import Extractor, extractor
from indexify.extractors.pdf_parser import Page, PDFParser
from indexify.graph import Graph


@extractor()
def download_pdf(url: str) -> File:
    """
    Download pdf from url
    """
    import urllib.request

    filename, _ = urllib.request.urlretrieve(url)

    resp = httpx.request(url=url, method="GET")

    with open(filename, "rb") as f:
        output = File(data=resp.content, mime_type="application/pdf")

    return output


class Document(BaseModel):
    pages: List[Page]


@extractor()
def parse_pdf(file: File) -> Document:
    """
    Parse pdf file and returns pages:
    """
    parser = PDFParser(file.data)
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
            if fragment.fragment_type == "text":
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
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def extract(self, input: TextChunk) -> TextChunk:
        embeddings = self.model.encode(input.chunk)
        input.embeddings = embeddings
        return input


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
    def __init__(self):
        super().__init__()
        import lancedb
        import pyarrow as pa

        self._client = lancedb.connect("vectordb.lance")
        text_emb_schema = pa.schema(
            [
                pa.field("text_embeddings", pa.list_(pa.float32(), list_size=384)),
                pa.field("document_id", pa.string()),
                pa.field("page_number", pa.int32()),
            ]
        )

        img_emb_schema = pa.schema(
            [
                pa.field("img_embeddings", pa.list_(pa.float32(), list_size=384)),
                pa.field("document_id", pa.string()),
                pa.field("page_number", pa.int32()),
            ]
        )
        self._text_emb_table = self._client.create_table(
            "text_embeddings", text_emb_schema
        )
        self._img_emb_table = self._client.create_table(
            "img_embeddings", img_emb_schema
        )

    def write_embeddings(
        self, input: Union[ImageWithEmbedding, TextChunk]
    ) -> List[Feature | BaseModel]:
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


if __name__ == "__main__":
    g = Graph(
        "Extract pages, tables, images from pdf",
        input=str,
        start_node=download_pdf,
        run_local=True,
    )

    clip_embedding = ImageEmbeddingExtractor()
    text_embedding = TextEmbeddingExtractor()
    write_to_vector_db = LancedDBWriter()

    g.add_edge(download_pdf, parse_pdf)
    g.add_edge(parse_pdf, extract_chunks)
    g.add_edge(parse_pdf, describe_images)
    g.add_edge(parse_pdf, clip_embedding)
    g.add_edge(extract_chunks, text_embedding)
    g.add_edge(describe_images, text_embedding)
    g.add_edge(text_embedding, write_to_vector_db)
    g.add_edge(clip_embedding, write_to_vector_db)

    url = "https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf"
    g.run(wf_input=url, local=True)
