from pprint import pformat

from indexify_langchain import IndexifyRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from indexify import IndexifyClient

client = IndexifyClient()


def setup():
    # Add contents
    file_names = [
        "skate.jpg",
        "congestion.jpg",
        "bushwick-bred.jpg",
        "141900.jpg",
        "132500.jpg",
        "123801.jpg",
        "120701.jpg",
        "103701.jpg",
    ]
    file_urls = [
        f"https://extractor-files.diptanu-6d5.workers.dev/images/{file_name}"
        for file_name in file_names
    ]
    for file_url in file_urls:
        client.ingest_remote_file(file_url, "image/png", {})

    client.add_extraction_policy(
        extractor="tensorlake/yolo-extractor", name="object_detection"
    )
    pass


# Fetch the schema containing the bounding box and object name
schema = client.list_schemas()["ddls"]["ingestion"]


def ask(question: str):
    template = (
        f"""
    Images are stored in the database with the following schema:
    {schema}

    fyi. "bounding_box" LIST NULL COMMENT 'Bounding box coordinates in the format [x1, y1, x2, y2]'

    Generate the SQL query based on the following question below:

    """
        + "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    chain = {"question": RunnablePassthrough()} | prompt | model | StrOutputParser()
    generated_sql = chain.invoke(question)

    print("\n--------------------")
    print("\nQuestion:\n", question)
    print("\nGenerated SQL:\n", generated_sql)

    query_result = client.sql_query(generated_sql)
    query_result = pformat(query_result.result).replace("{", "").replace("}", "")

    template = (
        f"""
    Images are stored in the database with the following schema:
    {schema}

    Generated SQL query based on the following question below:
    {generated_sql}

    Result of the SQL query based on the following question below:
    {query_result}

    Answer to the following question below:

    """
        + "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template)

    chain = {"question": RunnablePassthrough()} | prompt | model | StrOutputParser()

    answer = chain.invoke(question)

    print("\nAnswer:\n", answer)
    print("\n--------------------")
    pass


# Setup the environment - call this only once
setup()

ask("List all object names and their counts in the images")

ask("I want to find the largest bounding box in the images")
