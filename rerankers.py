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
    """Get responses from all models for a given prompt."""
    responses: dict[str, str] = {}
    for model in CHAT_MODELS:
        test_chat: cohere.ChatResponse = co.chat(
            model=model, 
            messages=[{"role": "user", "content": prompt}]
        )
        responses[model] = test_chat.model_dump()["message"]["content"][0]["text"]

    return responses

def rerank_documents(query: str, documents: list[str], top_n: int = 3) -> list[dict[str, str]]:
    """Rerank a list of documents based on relevance to the query."""
    rerank_response: cohere.V2RerankResponse = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_n,
        )
    
    results: list[dict[str, str]] = [
        {f"#{(i.index + 1)}": f"{i.relevance_score:.4f} - {documents[i.index][:150]}..."}
        for i in rerank_response.results
        ]
    
    return results

def rerank_model_responses(query: str, responses: list[str], top_n: int = None) -> list[dict[str, str]]:
    """Rerank model responses based on relevance to the query."""
    if top_n is None:
        top_n = len(responses)
        
    rerank_response: cohere.V2RerankResponse = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=responses,
        top_n=top_n,
        )
    
    results: list[dict[str, str]] = [
        {f"#{(i.index + 1)}": f"{i.relevance_score:.4f} - {CHAT_MODELS[i.index]} - {responses[i.index][:150]}..."}
        for i in rerank_response.results
        ]
    
    return results


if __name__ == "__main__":
    # Documents for reranking demo
    documents: list[str] = [
        "Climate change is primarily caused by the burning of fossil fuels like coal, oil, and natural gas. These fuels release carbon dioxide and other greenhouse gases when burned, trapping heat in the atmosphere and causing global warming.",
        "Renewable energy sources like solar, wind, and hydroelectric power generate electricity without producing greenhouse gas emissions. Transitioning to these clean energy sources is crucial for mitigating climate change.",
        "Electric vehicles (EVs) produce zero tailpipe emissions, but their overall environmental impact depends on how the electricity used to charge them is generated. If powered by renewable energy, EVs can significantly reduce carbon emissions compared to conventional vehicles.",
        "The mining of lithium for EV batteries has significant environmental impacts, including high water usage in water-scarce regions, potential chemical leakage, and habitat disruption. However, these impacts must be weighed against the long-term benefits of reduced emissions.",
        "Deforestation contributes to climate change by reducing carbon sequestration capacity and releasing stored carbon. Forests act as carbon sinks, absorbing CO2 from the atmosphere. When trees are cut down, this carbon is released back into the atmosphere.",
        "Carbon capture and storage (CCS) technologies aim to remove CO2 from point sources like power plants or directly from the atmosphere, then store it underground in geological formations. While promising, these technologies are still being developed at scale.",
        "The Paris Agreement is an international treaty on climate change adopted in 2015. Its goal is to limit global warming to well below 2°C, preferably 1.5°C, compared to pre-industrial levels. Countries submit their own climate action plans called Nationally Determined Contributions (NDCs)."
    ]
    
    # New prompt designed for effective reranking
    reranking_prompt = """
    Identify the most comprehensive explanation of solutions to climate change that includes both technological and policy approaches. 
    The best response should discuss renewable energy transition, carbon capture technologies, international agreements, and address 
    the challenges of implementation. Prioritize explanations that balance optimism with realistic assessment of challenges.
    """
    
    # Set number of top results to return
    top_n_docs = 5
    top_n_models = 5
    
    # Demo 1: Reranking a set of documents
    print(f"\n=== DOCUMENT RERANKING DEMO ===\n")
    print(f"Query: {reranking_prompt}\n")
    print(f"Reranking {len(documents)} documents, showing top {top_n_docs} results:\n")
    
    doc_results = rerank_documents(reranking_prompt, documents, top_n_docs)
    for result in doc_results:
        print(result)
        print("")
    
    # Demo 2: Reranking model responses
    print(f"\n=== MODEL RESPONSE RERANKING DEMO ===\n")
    
    # Get responses from all models
    responses_dict: dict[str, str] = get_responses(reranking_prompt)
    responses_list = list(responses_dict.values())
    
    print(f"Query: {reranking_prompt}\n")
    
    # Display each model's response
    for i, model in enumerate(CHAT_MODELS):
        print(
            f"Model: {model}\n"
            f"Response: {responses_list[i][:200]}...\n"
            "--------------------------------------------"
        )
    
    # Rerank the model responses
    print(f"\nReranking model responses, showing top {top_n_models} results:\n")
    model_results: list[dict[str, str]] = rerank_model_responses(reranking_prompt, responses_list, top_n_models)
    
    for result in model_results:
        print(result)
        print("")
    
    print("🐬")