import os
from typing import Optional
import dotenv
import cohere

dotenv.load_dotenv()
COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key=COHERE_API_KEY)
# rerank = co.rerank(model="rerank-v3.5")
chat: cohere.ChatResponse = co.chat(
    model="command-a-03-2025", #
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    )

CHAT_MODELS: list[str] = [
    "command-a-03-2025", # https://docs.cohere.com/docs/command-a#model-details
    "command-r7b-12-2024", # https://docs.cohere.com/docs/command-r7b#model-details
    "command-r-plus-08-2024", # https://docs.cohere.com/docs/command-r-plus#model-details
    "command-r-08-2024", # https://docs.cohere.com/docs/command-r#model-details
    "command", # https://docs.cohere.com/docs/command-beta#model-details
]

EMBEDDING_MODELS: list[str] = [
     # https://docs.cohere.com/docs/cohere-embed#english-models
     # https://docs.cohere.com/reference/embed#request.body.model
     "embed-english-v3.0", # 1024
     "embed-multilingual-v3.0", # 1024
     "embed-english-v2.0", # 4096
     ]

def get_responses(prompt: str) -> dict[str, str]:
    responses: dict[str, str] = {}
    for model in CHAT_MODELS:
        test_chat: cohere.ChatResponse = co.chat(model=model, messages=[{"role": "user", "content": prompt}])
        responses[model] = test_chat.model_dump()["message"]["content"][0]["text"]
    return responses

def rerank_responses(prompt: str, responses: list[str]) -> list[dict[int, float]]:
    """Rerank the responses based on relevance to the prompt."""
    rerank_response: cohere.V2RerankResponse = co.rerank(
        model="rerank-v3.5",
        query=prompt,
        documents=responses,
        # // documents=responses if responses else get_responses(prompt).values(),
        top_n=len(responses),
    )
    
    results: list[dict[int, float]] = [
        {f"#{(i.index + 1)}": f"{i.relevance_score} - {CHAT_MODELS[i.index]} - {responses[i.index][:100]}..."}
        for i in rerank_response.results
    ]
    
    return results


if __name__ == "__main__":
    question: str = "How many Rs are in the word strawberry?"
    for model in CHAT_MODELS:
        print(f"Question: {question}\n")
        test_chat = co.chat(model=model, messages=[{"role": "user", "content": question}])
        print(
            f"Model: {model}\n"
            f"Response: {test_chat.model_dump()["message"]["content"][0]["text"]}\n"
            # f"Relevance: {test_chat}\n"
            "--------------------------------------------"
        )

    
    print("🐬")