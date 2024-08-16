from typing import List
import json
from indexify.extractor_sdk.extractor import extractor
from indexify.extractor_sdk.data import BaseData

from indexify.graph import Graph
from indexify.local_runner import LocalRunner
from tt_module import get_tables
import pymupdf
import fitz


@extractor(description="Download pdf")
def download_pdf(url: str) -> str:
    import urllib.request
    filename, _ = urllib.request.urlretrieve(url)

    print(f'download {filename}')

    return filename


class PageText(BaseData):
    text: str

    page_num: float


@extractor(description="Extract page text from pdf")
def extract_page_text(input_file_path: str) -> List[PageText]:
    output = []
    with pymupdf.open(input_file_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            output.append(PageText(text=text, page_num=page_num))

    return output

class PageImage(BaseData):
    content_type: str
    data: bytes

    page_num: float
    img_num: float

@extractor(description="Extract image from pdf")
def extract_images(input_file_path) -> List[PageImage]:
    output = []

    doc = fitz.open(input_file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            output.append(PageImage(content_type="image/png", data=image_bytes, page_num=page_num, img_num=img_index))

    return output


class PageTable(BaseData):
    content_type: str
    data: bytes

    page_num: float

@extractor(description="Extract table from pdf")
def extract_tables(input_file_path: str) -> List[PageTable]:
    output = []

    with open(input_file_path, 'rb') as f:
        tables = get_tables(pdf_path=f.read())
    for page_num, content in tables.items():
        output.append(PageTable(content_type="application/json", data=json.dumps(content), page_num=page_num))

    return output

if __name__ == "__main__":
    g = Graph("Extract pages, tables, images from pdf", input=str, start_node=download_pdf)

    g.add_edge(download_pdf, extract_page_text)
    g.add_edge(download_pdf, extract_images)
    g.add_edge(download_pdf, extract_tables)

    runner = LocalRunner()

    url = "https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf"
    runner.run(g, wf_input=url)

    print(f"number of pages {len(runner.get_result(extract_page_text))}")
    print(f"number of images {len(runner.get_result(extract_images))}")
    print(f"number of tables {len(runner.get_result(extract_tables))}")

    print(runner.get_result(extract_page_text)[3])

