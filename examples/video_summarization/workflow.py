from typing import List, Optional

from pydantic import BaseModel, Field, Json

from indexify.functions_sdk.indexify_functions import indexify_function
from indexify.functions_sdk.data_objects import File 
from indexify.graph import Graph
from indexify.local_runner import LocalRunner

# Table -> Youtube Video Index 
# Col 1 -> Youtube Video URL
# Col 2 -> JSON of the transcription with segments
# Col 3 -> Classification results
# Col 4 -> Summary of the video

class YoutubeVideoData(BaseModel):
    file: File 
    transcription: Json = None
    classification: str = None
    summary: str = None
    labels: List[str] = []


@indexify_function()
def write_to_db(a: File, b: Transctipn, c: SpeechClassification, d: Summary) -> YoutubeVideoData:
    """
    Write the youtube video data to the database.
    """
    pass

def create_graph():
    g = Graph("Youtube_Video_Summarizer", start_node=get_youtube_video_data)
    g.add_edge(get_youtube_video_data, extract_audio_from_video)
    g.add_edge(extract_audio_from_video, transcribe_audio)
    g.add_edge(transcribe_audio, classify_meeting_intent)
    g.add_edge(classify_meeting_intent, summarize_job_interview)

class YoutubeURL(BaseModel):
    url: str = Field(..., description="URL of the youtube video")
    resolution: str = Field("480p", description="Resolution of the video")


@indexify_function()
def download_youtube_video(url: YoutubeURL) -> List[File]:
    """
    Download the youtube video from the url.
    """
    from pytubefix import YouTube

    yt = YouTube(url.url)
    # content = yt.streams.filter(res=url.resolution).first().download()
    # This doesn't always work as YT might not have the resolution specified
    content = yt.streams.first().download()
    return [File(data=content, mime_type="video/mp4")]


@indexify_function()
def extract_audio_from_video(file: File) -> File:
    """
    Extract the audio from the video.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(file.data)
    audio.export("audio.wav", format="wav")
    return File(
        data=audio.export("audio.wav", format="wav").read(), mime_type="audio/wav"
    )


class SpeechSegment(BaseModel):
    speaker: Optional[str] = None
    text: str
    start_ts: float
    end_ts: float


class SpeechClassification(BaseModel):
    classification: str
    confidence: float


class Transcription(BaseModel):
    segments: List[SpeechSegment]
    classification: Optional[SpeechClassification] = None


@indexify_function()
def transcribe_audio(file: File) -> Transcription:
    """
    Transcribe audio and diarize speakers.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu")
    segments, _ = model.transcribe("audio.wav")
    audio_segments = []
    for segment in segments:
        audio_segments.append(
            SpeechSegment(text=segment.text, start_ts=segment.start, end_ts=segment.end)
        )
    return Transcription(segments=audio_segments)


@indexify_function()
def classify_meeting_intent(speech: Transcription) -> Transcription:
    """
    Classify the intent of the audio.
    """
    from llama_cpp import Llama

    model = Llama.from_pretrained(
        repo_id="NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
        filename="*Q8_0.gguf",
        verbose=True,
        n_ctx=60000,
    )
    transcription_text = "\n".join([segment.text for segment in speech.segments])
    prompt = f"""
    You are a helpful assistant that classifies the intent of the audio.
    Classify the intent of the audio. These are the possible intents:
    - job-interview
    - sales-call
    - customer-support-call
    - technical-support-call
    - marketing-call
    - product-call
    - financial-call
    Write the intent of the audio in the following format:
    intent: job-interview

    The transcription of the audio is:
    {transcription_text}
    """
    output = model(prompt=prompt, max_tokens=50, stop=["\n"])
    response = output["choices"][0]["text"]
    output_tokens = response.split(":")
    if len(output_tokens) > 1:
        if output_tokens[0].strip() == "intent":
            if output_tokens[1].strip() in [
                "job-interview",
                "sales-call",
                "customer-support-call",
                "technical-support-call",
                "marketing-call",
                "product-call",
                "financial-call",
            ]:
                speech.classification = SpeechClassification(
                    classification=output_tokens[1].strip(), confidence=0
                )
                return speech
    speech.classification = SpeechClassification(classification="unknown", confidence=0)
    return speech


def route_transcription_to_summarizer(speech: Transcription) -> Optional[str]:
    """
    Route the transcription to the summarizer based on the classification result from
    the classify_text_feature extractor.
    """
    if speech.classification.classification == "job-interview":
        return "summarize_job_interview"
    elif speech.classification.classification == "sales-call":
        return "summarize_sales_call"
    return None


class Summary(BaseModel):
    summary: str


@indexify_function()
def summarize_job_interview(speech: Transcription) -> Summary:
    """
    Summarize the job interview.
    """
    from llama_cpp import Llama

    model = Llama.from_pretrained(
        repo_id="NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
        filename="*Q8_0.gguf",
        verbose=False,
        n_ctx=60000,
    )
    transcription_text = "\n".join([segment.text for segment in speech.segments])
    prompt = f"""
    I have a transcript of a job interview that took place on [date]. The interview included discussions about the candidate’s background, skills, and experience, as well as their 
    responses to specific questions and scenarios. Please summarize the key points from the interview, including:

    1. Candidate’s Strengths and Qualifications: Highlight any notable skills, experiences, or achievements mentioned.
    2. Key Responses and Insights: Summarize the candidate’s answers to important questions or scenarios.
    3. Cultural Fit and Soft Skills: Provide an overview of the candidate’s fit with the company culture and any relevant soft skills.
    4. Areas of Concern or Improvement: Note any reservations or areas where the candidate might need further development.
    5. Overall Impression and Recommendation: Offer a brief assessment of the candidate’s suitability for the role and any suggested next steps.

    The transcript is:
    {transcription_text}
    """
    output = model(prompt=prompt, max_tokens=30000, stop=["\n"])
    response = output["choices"][0]["text"]
    return Summary(summary=response)


@indexify_function()
def summarize_sales_call(speech: Transcription) -> Summary:
    """
    Summarize the sales call.
    """
    from llama_cpp import Llama

    model = Llama.from_pretrained(
        repo_id="NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
        filename="*Q8_0.gguf",
        verbose=True,
        n_ctx=60000,
    )
    transcription_text = "\n".join([segment.text for segment in speech.segments])
    prompt = f"""
    I had a sales call with a prospective client earlier today. The main points of the conversation included [briefly describe key topics discussed, such as client needs, product features, 
    objections, and any agreements or follow-ups]. Please summarize the call, highlighting the key details, client concerns, and any action items or next steps. Additionally, 
    if there are any recommendations for improving our approach based on the discussion, please include those as well

    The transcript is:
    {transcription_text}
    """
    output = model(prompt=prompt, max_tokens=30000, stop=["\n"])
    response = output["choices"][0]["text"]
    return Summary(summary=response)


def create_graph():
    g = Graph("Youtube_Video_Summarizer", start_node=download_youtube_video)
    g.add_edge(download_youtube_video, extract_audio_from_video)
    g.add_edge(extract_audio_from_video, transcribe_audio)
    g.add_edge(transcribe_audio, classify_meeting_intent)

    g.add_node(summarize_job_interview)
    g.add_node(summarize_sales_call)

    g.route(classify_meeting_intent, route_transcription_to_summarizer)
    return g


if __name__ == "__main__":
    g = create_graph()
    runner = LocalRunner()
    runner.register_extraction_graph(g)
    content_id = runner.invoke_graph_with_object(
        g.name, url=YoutubeURL(url="https://www.youtube.com/watch?v=R_dxlajqA4s")
    )
    print(f"[bold] Retrieving transcription for {content_id} [/bold]")
    outputs = runner.graph_outputs(
        g.name, ingested_object_id=content_id, extractor_name=transcribe_audio.name
    )
    transcription = outputs[0].payload
    for segment in transcription.segments:
        print(f"[bold] {segment.start_ts} - {segment.end_ts} [/bold]")
        print(f"[bold] {segment.text} [/bold]\n")
