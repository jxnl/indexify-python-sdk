from fasterwhisper import FasterWhisper
from asr_extractor import ASRExtractor, ASRExtractorConfig
from indexify import Content, Feature, extractor
from indexify.graph import Graph
from indexify.local_runner import LocalRunner

from typing import List
from pydantic import BaseModel, Field

import json
import subprocess

@extractor(description="Get yt video")
def get_yt_video_and_extract_audio(_: Content) -> List[Content]:
    # TODO download video from yt but let's hardcode it for now.
    file_loc = './indexify_example_data/Mock Interview Preparation： Common Questions with Feedback! [R_dxlajqA4s].mp4'
    output_loc = './indexify_example_data/audio.mp3'
    try:
        # -y
        result = subprocess.call(['ffmpeg', '-i', file_loc, output_loc])
    except CalledProcessError as e:
        # TODO how do we handle this?
        pass

    return [Content.from_file(output_loc)]

# kind of annoying to not know the types of the output being generated.
@extractor(description="Diarize Speakers in audio")
def diarize_speakers(content: Content) -> List[Content]:
    # params = ASRExtractorConfig(batch_size=1)
    # extractor = ASRExtractor()
    # results = extractor.extract(content, params=params)
    # return results

    # hardcoded because cpu diarizer isn't working
    return [Content.from_file('indexify_example_data/assembly-transcript.txt')]

@extractor(description="Classify text into job interview or sales call")
def classify_text_feature(content: Content) -> List[Content]:
    import openai
    KEY = ''

    client = openai.OpenAI(api_key=KEY)

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a classifier assistant, skilled in classifying audio transcripts with ease."},
        {"role": "user", "content": "Can you classify the following transcript as either a job interview or a sales call? Simply return either `job-interview` or `sales-call` and nothing else."},
        {"role": "user", "content": f"{content}"},
    ]
    )

    classification = completion.choices[0].message.content
    print(classification)

    if classification == 'job-interview':
        feature = Feature.metadata(value=json.dumps({"classification": "job-interview"}))
        content.features.append(feature) 
    else:
        # TODO check if the content is actually one or the other and figure out how to do error handling here.
        pass

    return [content]

@extractor(description="Summarize Job interview")
def summarize_job_interview(content: Content) -> List[Content]:
    import openai
    KEY = ''

    client = openai.OpenAI(api_key=KEY)

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a summarizer assistant, skilled in summarizing audio transcripts with ease."},
        {"role": "user", "content": "Can you summarize the following job interview transcript and tell me what the key highlights are."},
        {"role": "user", "content": f"{content}"},
    ]
    )

    # TODO check if the content is actually one or the other and figure out how to do error handling here.

    return Content.from_text(completion.choices[0].message.content)

@extractor(description="Summarize sales call")
def summarize_sales_call():
    pass


@extractor(description="Get transcripts for audio using FasterWhiser")
def extract_transcript(audioContent: Content) -> List[Content]:
    faster_whisper = FasterWhisper()
    return faster_whisper.extract(audioContent)


if __name__ == "__main__":
    g = Graph("FilterGraph")

    (
        g.step(diarize_speakers, classify_text_feature)
        .step(classify_text_feature, summarize_job_interview, prefilter_predicates="classification=job-interview")
        .step(classify_text_feature, summarize_sales_call, prefilter_predicates="classification=sales-call")
    )

    local_runner = LocalRunner()
    local_runner.run(g, Content.from_text(""))

    print([i for i in local_runner.get_result(summarize_job_interview)])

    # print([i.data.decode('utf-8') for i in local_runner.get_result(summarize_job_interview)])