# 📄 RERANKING

This repository contains a Python script that demonstrates the power of reranking in two critical NLP applications:
1. Evaluating and comparing outputs from multiple large language models
2. Finding the most relevant documents from a collection based on a query

The tool uses Cohere's API as an implementation example, but the concepts can be applied with any reranking system.

## About Reranking

Reranking is a powerful technique that goes beyond traditional search and retrieval methods. Unlike simple keyword matching or embedding similarity, reranking evaluates content based on deeper relevance to specific criteria.

Key advantages of reranking:

- **Contextual understanding**: Evaluates semantic relevance beyond keyword matching
- **Multi-dimensional assessment**: Considers multiple aspects of quality simultaneously
- **Flexible criteria**: Can be tailored to specific evaluation needs
- **Comparative analysis**: Provides relative quality scores across multiple options

## How This Tool Works

Our script demonstrates two distinct reranking workflows:

### Document Reranking
1. Provides a collection of documents on a related topic
2. Defines a specific query with clear evaluation criteria
3. Uses a reranking model to score each document's relevance
4. Returns the top N most relevant documents with scores

### Model Response Reranking
1. Sends the same prompt to multiple language models
2. Collects each model's unique response
3. Uses a reranking model to evaluate all responses against the query criteria
4. Ranks the responses by relevance score, showing which model performed best

## Creating Effective Reranking Queries

The repository includes an example query specifically designed for effective reranking:

> Identify the most comprehensive explanation of solutions to climate change that includes both technological and policy approaches. The best response should discuss renewable energy transition, carbon capture technologies, international agreements, and address the challenges of implementation. Prioritize explanations that balance optimism with realistic assessment of challenges.

What makes this an effective reranking query:

- **Specific criteria**: Clearly defines what makes a response "good"
- **Multiple dimensions**: Evaluates across several aspects (comprehensiveness, balance, etc.)
- **Detailed expectations**: Specifies content elements that should be present
- **Quality differentiation**: Allows meaningful distinction between superficial and thorough responses

## Getting Started

### Prerequisites

- Python 3.7+
- Cohere API key (or substitute with your preferred provider)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install cohere python-dotenv
   ```
3. Create a `.env` file in the root directory with your API key:
   ```
   COHERE_API_KEY=your_api_key_here
   ```

### Usage

Run the script:
```
python rerankers.py
```

The script will:
1. Perform document reranking demo:
   - Rerank a collection of climate change documents
   - Display the top N most relevant documents with scores
   
2. Perform model comparison demo:
   - Query multiple models with the same prompt
   - Display preview of each model's response
   - Rerank all responses based on relevance to criteria
   - Show the top N ranked responses with scores

## Practical Applications

Reranking can be applied in numerous real-world scenarios:

### Document Retrieval
- **Research assistance**: Find the most relevant academic papers for a research question
- **Legal discovery**: Identify the most pertinent documents in a large collection
- **Knowledge management**: Surface the most helpful documentation for internal teams

### Model Evaluation
- **LLM benchmarking**: Compare model performance on specific tasks
- **Response quality assurance**: Ensure AI outputs meet quality standards
- **Model selection**: Identify which model performs best for specific use cases

### Content Curation
- **Content recommendation**: Surface the most relevant content for users
- **Data filtering**: Select highest quality examples for training datasets
- **Summarization**: Identify the most important passages to include in summaries

## Customization

You can easily modify the script to:
- Change the document collection
- Adjust the reranking query
- Modify the number of top results returned
- Use different models for generation or reranking
- Apply different formatting for the results


## License
This project is licensed under the MIT License - see the LICENSE file for details.