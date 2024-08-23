from pydantic import BaseModel
from enum import Enum
from typing import List, Optional


class PageFragmentType(str, Enum):
    TEXT = "text"
    FIGURE = "image"
    TABLE = "table"


class Image(BaseModel):
    data: bytes
    mime_type: str

class TableEncoding(str, Enum):
    CSV = "csv"
    JSON = "json"

class Table(BaseModel):
    data: str
    encoding: TableEncoding

class PageFragment(BaseModel):
    fragment_type: PageFragmentType
    text: Optional[str]
    image: Optional[Image]
    table: Optional[Table]
    reading_order: int


class Page(BaseModel):
    number: int
    fragments: List[PageFragment]

class PDFParser:

    def __init__(self, data: bytes):
        self._data = data


    def parse(self) -> List[Page]:
        pass