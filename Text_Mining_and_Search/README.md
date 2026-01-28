# ✍️ Wikipedia Summarizer: Topic Modelling & Text Summarisation

This project investigates topic modelling and single-document text summarisation on the Wikipedia Featured Articles corpus. We compare statistical and embedding-based topic models and evaluate extractive and abstractive summarisation methods, analyzing the trade-off between linguistic quality and factual consistency.

# Authors
- Marco Rosato
- Elena Maggiore

## Project Overview
The project is structured around two core tasks:

1. **Topic Modelling:** Identification of latent thematic structures in long, diverse documents.
2. **Text Summarisation:** Generation of concise summaries that resemble human-written references, comparing extractive and abstractive approaches.

We evaluate how different topic modelling techniques impact interpretability and how summarisation methods balance linguistic quality against factual consistency.

# Dataset

* Wikipedia Featured Articles (English)
* 3,793 training documents, 30 test documents
* Human-written summaries used as gold standards

The project is structured into several key stages:

# Topic Modelling

We implemented and compared

* **pLSA** (via NMF with KL divergence)
* **LDA**
* **BERTopic** (embeddings + clustering)

Models were tuned using **Topic Coherence (C_v)**.
**pLS*A** with **40 topics** achieved the best interpretability and was selected for downstream tasks.

# Text Summarisation

## Extractive
Sentences are scored based on their representation in topic space and selected using Maximal Marginal Relevance to reduce redundancy.
* **Global Extractive:** corpus-aware pLSA topics
* **Local Extractive:** document-specific topic inference

## Abstractive
* **Longformer Encoder-Decoder (LED)** for long-context summarisation
* **Qwen2.5-Instruct (7B)** with chunking and Map-Reduce strategy

# Evaluation
Summaries were evaluated using:

* **ROUGE**
* **BERTScore**
* **SummaC** (factual consistency)

Abstractive LLMs achieved the best fluency and semantic similarity, while global extractive summarisation provided the highest factual reliability.

# Conclusion
The results show that pLSA is an effective topic modelling method for long encyclopedic documents, offering strong interpretability. In summarisation, abstractive models produce more fluent text but suffer from hallucinations, while extractive methods ensure factual consistency at the cost of readability. This highlights a clear trade-off between linguistic quality and reliability.
