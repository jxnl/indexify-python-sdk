from typing import List
import io
import json
from typing import List

import fitz
import httpx
import httpx

from indexify.extractor_sdk.extractor import extractor, Extractor
from indexify.extractor_sdk.data import BaseData, Feature, File
from indexify.extractors.pdf_parser import PDFParser, Page

from sentence_transformers import SentenceTransformer, util

from indexify.graph import Graph

@extractor()
def download_pdf(url: str) -> File:
    """
    Download pdf from url
    """
    import urllib.request

    filename, _ = urllib.request.urlretrieve(url)

    resp = httpx.request(url=url, method="GET")

    with open(filename, 'rb') as f:
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

class PageImage(BaseData):
    content_type: str
    data: bytes

    page_num: int
    img_num: int


@extractor()
def extract_images(pdf_file: PDFFile) -> List[PageImage]:
    """
    Extract images from pdf
    """
    output = []

    doc = fitz.open("pdf", pdf_file.data)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            output.append(
                PageImage(
                    content_type="image/png",
                    data=image_bytes,
                    page_num=page_num,
                    img_num=img_index,
                )
            )

    return output


class PageTable(BaseData):
    data: bytes

    page_num: int


@extractor()
def extract_tables(pdf_file: PDFFile) -> List[PageTable]:
    """
    Extract tables from pdf
    """
    output = []

    tables = get_tables(pdf_path=pdf_file.data)
    for page_num, content in tables.items():
        output.append(PageTable(data=json.dumps(content), page_num=page_num))

    return output


class TextChunk(BaseData):
    chunk: str
    metadata: dict = {}

class ChunkParams(BaseModel):
    chunk_size: int


@extractor()
def extract_chunks(document: Document, chunk_size: int, overlap: int) -> List[TextChunk]:
    """
    Extract chunks from document
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,overlap=overlap)
    chunks: List[TextChunk] = []
    for page in document.pages:
        for fragment in page.fragments:
            if fragment.fragment_type == "text":
                texts = text_splitter.split_text(fragment.text)
                for text in texts:
                    chunks.append(TextChunk(chunk=text, metadata={"page_number": page.number}))

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
class Embedding(BaseData):
    embedding: List[float]
    embedding_type: str

class EmbeddingExtractor(Extractor):
    name = "text-embedding"
    description = "Extractor class that captures an embedding model"
    system_dependencies = []
    input_mime_types = ["text"]

    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def extract(self, input: TextChunk, params: BaseModel=None) -> Embedding:
        embeddings = self.model.encode(input.chunk)
        return Embedding(embedding=embeddings, embedding_type="text")
    
class ImageEmbedding(Extractor):
    name = "image-embedding"
    description = "Extractor class that captures an embedding model"
    system_dependencies = []
    input_mime_types = ["text"]

    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('clip-ViT-B-32')

    def extract(self, document: Document) -> List[Embedding]:
        from PIL import Image
        embedding = []
        for page in document.pages:
            for fragment in page.fragments:
                if fragment.fragment_type == "image":
                    image = fragment.image
                    img_emb = self.model.encode(Image.open(io.BytesArray(image.data)))
                    embedding.append(Embedding(embedding=img_emb, embedding_type="image"))  
        return embedding
    
class LancdDBWriter(Extractor):
    def __init__(self):
        super().__init__()
        import lancedb
        import pyarrow as pa
        self._client = lancedb.connect("vectordb.lance")
        text_emb_schema = pa.schema([pa.field("text_embeddings", pa.list_(pa.float32(), list_size=384)),
                            pa.field("document_id", pa.string()),
                            pa.field("page_number", pa.int32()),])
        
        img_emb_schema = pa.schema([pa.field("img_embeddings", pa.list_(pa.float32(), list_size=384)),
                            pa.field("document_id", pa.string()),
                            pa.field("page_number", pa.int32()),])
        self._text_emb_table = self._client.create_table("text_embeddings", text_emb_schema)
        self._img_emb_table = self._client.create_table("img_embeddings", img_emb_schema)

    def extract(self, embeddings: List[Embedding]) -> List[Feature | BaseModel]:
        for emb in embeddings:
            if emb.embedding_type == "text":
                pass
            elif emb.embedding_type == "image":
                pass
    
if __name__ == "__main__":
    g = Graph(
        "Extract pages, tables, images from pdf",
        input=str,
        start_node=download_pdf,
        run_local=True,
    )

    text_embedding = TextEmbedding()
    clip_embedding = ImageEmbedding()
    write_to_vectordb = LancdDBWriter()

    g.add_edge(download_pdf, parse_pdf)
    g.add_edge(parse_pdf, extract_chunks.par)
    g.add_edge(parse_pdf, describe_images)
    g.add_edge(extract_chunks, text_embedding)
    g.add_edge(describe_images, clip_embedding)
    g.add_edge(text_embedding, write_to_vectordb)
    g.add_edge(clip_embedding, write_to_vectordb)

    url = "https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf"
    g.run(wf_input=url, local=True)

    #print(f"number of pages {len(g.get_result(extract_page_text))}")
    #print(f"number of images {len(g.get_result(extract_images))}")
    #print(f"number of tables {len(g.get_result(extract_tables))}")
    #print(f"number of embeddings {len(g.get_result(EmbeddingExtractor))}")

    print('\n---- Text output')
    #print(g.get_result(extract_page_text)[3])
    print('---- /Text output\n')

    print('\n---- Embedding output')
    #print(g.get_result(EmbeddingExtractor)[3])
    print('---- /Embedding output\n')
