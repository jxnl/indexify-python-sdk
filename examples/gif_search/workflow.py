from typing import List, Optional

from pydantic import BaseModel

from indexify import Graph


class Image(BaseModel):
    data: bytes
    mime_type: str
    caption: Optional[str] = None
    caption_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None


def describe_image(image: Image):
    pass


def embed_caption(image: Image):
    pass


def embed_image(image: Image):
    pass


def write_to_db(image: Image):
    pass


if __name__ == "__main__":
    g = Graph(namespace="gif_search", start_node=describe_image)
    g.add_edge(describe_image, embed_caption)
    g.add_edge(embed_caption, embed_image)
    g.add_edge(embed_image, write_to_db)
    g.run(image=Image(data=b"", mime_type="image/png"))
