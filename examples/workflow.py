from indexify import Content, Feature, extractor
from indexify.graph import Graph
from indexify.local_runner import LocalRunner

from typing import List
from pydantic import BaseModel, Field

class ChunkOptions(BaseModel):
    chunk_size: int = 100

class ProfanityOptions(BaseModel):
    words: List[str] = Field(default_factory=lambda: [])


@extractor(description="Extracts text from PDFs")
def pdf_extraction(content: Content) -> List[Content]:
    return [Content.from_text("this is stupid"), Content.from_text("fuck why"), Content.from_text("this isn't stupid")]

@extractor(description="Profanity filter")
def filter_for_profanity(content: Content, params: ProfanityOptions) -> List[Content]:
    for profane_word in params['words']:
        if profane_word in content.data.decode('utf-8'):
            return []

    return [content]

@extractor(description="Object detector in the images")
def object_detector(content: Content) -> List[Content]:
    # TODO This should call a model or whatever
    return [Content.from_text("detected r12c201"), Content.from_text("detected r1c1")]

@extractor(description="Chunks text into paragraphs")
def text_chunks(content: Content, params: ChunkOptions) -> List[Content]:
    # TODO this is hardcoded and should call the model
    metadata1 = Feature.metadata(value={'a': 'one', 'b': 'two', 'c': "three"})
    metadata2 = Feature.metadata(value={'a': '2', 'b': '1'})
    features = [metadata1, metadata2]

    text = content.data.decode('utf-8')
    ret = [Content.from_text(i + "--" + j, features=features) for i, j in zip(text.split()[:-1], text.split()[1:])]

    ret.extend([Content.from_text("text chunk that wasn't filtered")])

    return ret

@extractor(description="Embeds text chunks")
def chunk_embeddings(content: Content) -> List[Feature]:
    return [content]


if __name__ == "__main__":
    g = Graph("FilterGraph")

    filter_for_profanity.params = {"words": ["fuck"]}
    text_chunks.params = {"chunk_size": 500}

    (
        g.steps(pdf_extraction, [filter_for_profanity, object_detector])
        .step(filter_for_profanity, text_chunks)
        .step(object_detector, text_chunks, prefilter_predicates="a=one and c=three")
        .step(text_chunks, chunk_embeddings)
    )

    # Use this edge replacement in the comment as an example of how to use this edge without a predicated is commented out below.
    #g.step(object_detector, text_chunks)
    #expected output: ['this--is', 'is--stupid', "this--isn't", "isn't--stupid", 'detected--r12c201', 'detected--r1c1', 'detected--r12c201', 'detected--r1c1', 'detected--r12c201', 'detected--r1c1']

    # TODO read from the pdf file.
    #content = Content.from_file("sample.pdf")
    local_runner = LocalRunner()
    local_runner.run(g, Content.from_text(""))

    print([i.data.decode('utf-8') for i in local_runner.get_result(chunk_embeddings)])
    # output: ['this--is', 'is--stupid', "text chunk that wasn't filtered", "this--isn't", "isn't--stupid", "text chunk that wasn't filtered"]
    # you should see the `Content`` repeated in the output since the return value of the `text-chunker` is hardcoded and because text-chunker is
    # invoked twice, once after `filter-for-profanity` and once after `object-detector`.
