import os
import dotenv
import cohere

from typing import Optional

def main() -> None:
    dotenv.load_dotenv()
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    co = cohere.ClientV2(api_key=COHERE_API_KEY)

    docs: list[str] = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
    "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
    ]

    response: cohere.V2RerankResponse = co.rerank(
        model="rerank-v3.5", # https://docs.cohere.com/docs/rerank-2
        query="What is the capital of the United States?",
        documents=docs,
        # top_n=3,
        top_n=len(docs),
        )

    # results = response.model_dump_json()
    results: list[dict[int, float]] = [{f"#{(i.index + 1)}" : f"{i.relevance_score} - {docs[i.index]}"} for i in response.results]

    for j in results:
        print(j)


if __name__ == "__main__":
    main()
    print("🐬")
