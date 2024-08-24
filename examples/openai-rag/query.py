from openai import OpenAI

from indexify import IndexifyClient

client = IndexifyClient()
client_openai = OpenAI()


def query_database(question: str, index: str, top_k=3):
    retrieved_results = client.search_index(name=index, query=question, top_k=top_k)
    context = "\n-".join([item["text"] for item in retrieved_results])
    response = client_openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Answer the question, based on the context.\n question: {question} \n context: {context}",
            },
        ],
        model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(
        query_database(
            "What accomplishments did Kevin durant achieve during his career?",
            "summarize_and_chunk.wikiembedding.embedding",
            4,
        )
    )
