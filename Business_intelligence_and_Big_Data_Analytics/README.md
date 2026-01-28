# 🎓 SyllabusRAG: AI Academic Advisor

This project aims to develop a chatbot that leverages the syllabus of the Data Science Master’s degree at the University of Milano-Bicocca. The system is built using a Retrieval-Augmented Generation (RAG) approach to provide students with accurate, reliable and context-aware information regarding course content, prerequisites, and assessments.

The conversational interface is powered by the **Qwen3-4B-Instruct** model, integrated with a MongoDB database containing structured course information.

# Authors
- Marco Rosato
- Cristina Papi
- Rowyda Askalani

## Core Challenges & Solutions: Hybrid Retrieval & Context
A primary challenge in querying academic syllabi is balancing the need for semantic understanding (concepts) with exact keyword matching (course codes), as well as handling multi-turn conversations and complex course structures.

To solve this, we implemented a multi-layered strategy:

1.  **Hybrid Embeddings:** We utilized the **BGE-M3 model** to generate two types of vectors:
    *   *Dense Vectors:* To capture semantic meaning.
    *   *Sparse Vectors:* To capture exact keyword importance.

2.  **Integrated Course Logic:** A specific logic was developed to detect if a course is a "module" of a larger exam. The system injects warning notes into the retrieved chunks, ensuring the LLM informs students that they must pass related modules to record a grade.

3.  **Conversational Memory (Query Rewriting):** To handle follow-up questions (e.g., "Is it mandatory?"), we implemented a query rewriter that uses chat history to make user questions self-contained before retrieval.

# Project Workflow
The project is structured into several key stages:

1. **Knowledge Base Construction**
    *   **Structured Chunking:** Syllabus documents were split into logical sections (Identity, Content, Objectives, Requirements) to improve retrieval precision.
    *   **Data Injection:** "Integration notes" were automatically added to text chunks for courses that are part of larger integrated exams.
    *   **Storage:** Chunks, metadata, and hybrid embeddings were stored in MongoDB.

2. **Advanced Retrieval Pipeline**
    *   **Stage 1 (Hybrid Search):** A combination of dense and sparse scores retrieves a broad set of candidate documents.
    *   **Stage 2 (Reranking):** A Cross-Encoder (**BGE-Reranker-v2**) analyzes the query-document pairs to reorder candidates based on high-precision relevance.

3. **LLM Integration & Chat History**
    *   **Model Optimization:** The Qwen3-4B-Instruct model was loaded in 4-bit mode to optimize VRAM usage while maintaining instruction-following capabilities.
    *   **Contextualization:** A "Academic Advisor" persona was defined via system prompts to ensure factual answers and avoid hallucinations.

4. **Evaluation & Testing**
    *   The system was tested against a "Baseline" (LLM only) and "RAG without History."
    *   **Metrics:** Performance was measured using Cosine Similarity, BERTScore, and Semantic Answer Similarity (SAS).
    *   **Results:** The "RAG with History" approach demonstrated a significant improvement in conversational accuracy, particularly for follow-up questions where memory gaps were successfully bridged.
