from fasterwhisper import FasterWhisper
from asr_extractor import ASRExtractor, ASRExtractorConfig
from indexify import Content, Feature, extractor
from indexify.graph import Graph
from indexify.local_runner import LocalRunner
from indexify.data import BaseData

from typing import List
from pydantic import BaseModel, Field

import json
import subprocess


class YoutubeURL(BaseData):
    url: str = Field(..., description="URL of the youtube video")
    resolution: str = Field("720p", description="Resolution of the video")


class DiarizedSpeechSegment(BaseData):
    speaker: str
    text: str
    start_ts: int
    end_ts: int


class DiarizedSpeech(BaseData):
    segments: List[DiarizedSpeechSegment]


class Summary(BaseModel):
    summary: str


class SpeechClassification(BaseData):
    classification: str
    confidence: float


class DiarizedSpeechWithClassification(BaseData):
    diarized_speech: DiarizedSpeech
    classification: SpeechClassification


class UploadFile(BaseModel):
    data: bytes


@extractor(description="Get yt video")
def get_yt_video_and_extract_audio(url: YoutubeURL) -> List[UploadFile]:
    # TODO download video from yt but let's hardcode it for now.
    file_loc = "./indexify_example_data/Mock Interview Preparation： Common Questions with Feedback! [R_dxlajqA4s].mp4"
    output_loc = "./indexify_example_data/audio.mp3"
    # try:
    #    # -y
    #    result = subprocess.call(['ffmpeg', '-i', file_loc, output_loc])
    # except CalledProcessError as e:
    #    raise e
    #    pass
    f = open("requirements.txt", "br")
    return [UploadFile(data=f.read())]


# kind of annoying to not know the types of the output being generated.
@extractor(description="Diarize Speakers in audio")
def diarize_speakers(file: UploadFile) -> DiarizedSpeech:
    # params = ASRExtractorConfig(batch_size=1)
    # extractor = ASRExtractor()
    # results = extractor.extract(content, params=params)
    # return results

    # hardcoded because cpu diarizer isn't working
    return DiarizedSpeech(
        segments=[
            DiarizedSpeechSegment(
                speaker="Speaker 1",
                text="Hello, my name is John Doe",
                start_ts=0,
                end_ts=5,
            ),
            DiarizedSpeechSegment(
                speaker="Speaker 2",
                text="Hello, my name is Jane Doe",
                start_ts=5,
                end_ts=10,
            ),
        ]
    )


@extractor(description="Classify text into job interview or sales call")
def classify_text_feature(speech: DiarizedSpeech) -> List[Feature]:
    return [Feature.metadata(value={"intent": "job-interview"})]


@extractor(description="Summarize Job interview")
def summarize_job_interview(speech: DiarizedSpeech) -> Summary:
    return Summary(summary="This is a summary of the job interview")


@extractor(description="Summarize sales call")
def summarize_sales_call(speech: DiarizedSpeech) -> Summary:
    return Summary(summary="This is a summary of the sales call")


def create_graph():
    g = Graph("Crawler", input=YoutubeURL, start_node=get_yt_video_and_extract_audio)
    g.add_edge(get_yt_video_and_extract_audio, diarize_speakers)
    g.add_edge(diarize_speakers, classify_text_feature)
    g.add_edge(
        classify_text_feature,
        summarize_job_interview,
        prefilter_predicates="intent=job-interview",
    )
    g.add_edge(
        classify_text_feature,
        summarize_sales_call,
        prefilter_predicates="intent=sales-call",
    )
    return g


if __name__ == "__main__":
    g = create_graph()

    runner = LocalRunner()
    runner.run(g, wf_input=YoutubeURL(url="https://www.youtube.com/watch?v=R_dxlajqA4s"))

    print(f"--- wf output: {runner.get_result(classify_text_feature)}")
    print(f"--- wf output: {runner.get_result(summarize_job_interview)}")
