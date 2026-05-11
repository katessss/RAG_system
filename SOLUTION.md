# RAG-Based Hybrid Search Systems

**for Technical Documentation**

**Author:** Semenova Ekaterina Evgenievna

**Git Repository:** [https://github.com/katessss/RAG_system](https://github.com/katessss/RAG_system)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [System Architecture](#3-system-architecture)
   - 3.1 [Data Loading Pipeline](#31-data-loading-pipeline)
   - 3.2 [Search Pipeline](#32-search-pipeline)
4. [PDF Document Parsing](#4-pdf-document-parsing)
   - 4.1 [How It Works Internally](#41-how-it-works-internally)
   - 4.2 [Why This Matters for RAG](#42-why-this-matters-for-rag)
5. [Chunking Logic](#5-chunking-logic)
   - 5.1 [Chunking Parameters](#51-chunking-parameters)
   - 5.2 [Text Cleaning](#52-text-cleaning)
6. [Data Storage](#6-data-storage)
   - 6.1 [ChromaDB: Vector Index](#61-chromadb-vector-index)
   - 6.2 [SQLite FTS5: Full-Text Index](#62-sqlite-fts5-full-text-index)
7. [Embedding Models](#7-embedding-models)
8. [Result Merging Strategies](#8-result-merging-strategies)
   - 8.1 [Full-Text Search and Morphology](#81-full-text-search-and-morphology)
   - 8.2 [Reranking](#82-reranking)
9. [Answer Generation (LLM Layer)](#9-answer-generation-llm-layer)
10. [Quality Evaluation System](#10-quality-evaluation-system)
11. [Testing Results Report](#11-testing-results-report)
    - 11.1 [Experiment Description](#111-experiment-description)
    - 11.2 [Semantic Search (without FTS)](#112-semantic-search-without-fts)
    - 11.3 [Hybrid Search: All Strategies](#113-hybrid-search-all-strategies)
12. [Conclusions](#124-conclusions)
13. [Summary](#13-summary)
14. [Project Structure](#11-project-structure)

---

## 1. System Overview

The system is designed for intelligent search over a corpus of technical documentation in PDF format. It implements the RAG (Retrieval-Augmented Generation) approach: it first retrieves the most relevant document fragments, then passes them to a language model to generate an answer to a natural-language question.

The key feature is **hybrid search**, combining two fundamentally different methods:

- **Semantic search** — finds fragments that are semantically close to the query, even if they don't contain the same words.
- **Full-Text Search (FTS)** — finds fragments containing the exact terms and abbreviations present in the query.

Results from both methods are merged and optionally reranked, compensating for the weaknesses of each approach individually.

The top 5 retrieved results are passed to the LLM to produce a structured answer.

---

## 2. Technology Stack

| **Component** | **Technology** | **Purpose** |
|---|---|---|
| **PDF Parsing** | Docling | OCR, table extraction, and document structure recognition |
| **Vector DB** | ChromaDB (PersistentClient) | Storing and searching embeddings |
| **Full-Text DB** | SQLite FTS5 | BM25 search over morphologically normalized text |
| **Embedding Models** | SentenceTransformers | Vector similarity computation |
| **Reranker** | Cross-encoder | Reranking of candidate results |
| **Morphology** | NLTK SnowballStemmer | Russian text stemming for FTS queries |
| **Logging** | colorlog | Colorized structured console output |
| **LLM** | Qwen2.5, 7B | Chunk analysis and structured answer generation |

---

## 3. System Architecture

The system consists of two independent pipelines:

- **Data loading pipeline** — runs once during document indexing.
- **Search pipeline** — runs on every user query.

### 3.1 Data Loading Pipeline

The goal of this pipeline is to transform raw PDF files into two indexes: a vector index and a full-text index.

**Step 1 — PDF Parsing.** Each document is processed by the Docling library with OCR and table recognition enabled. The output is a structured document representation with element markup: headings, text blocks, tables, figures.

**Step 2 — Chunking.** The structured representation is split into semantic fragments (chunks). The chunking logic accounts for element type and its position in the document.

**Step 3 — Embedding Creation.** A vector representation is computed for each chunk using the selected embedding model. Before encoding, section context and a model-specific prefix are added to the text.

**Step 4 — Saving to ChromaDB.** Chunks along with their vectors are stored in the local ChromaDB vector database. Metadata (page number, context, element type) is stored alongside each chunk.

**Step 5 — Saving to SQLite FTS5.** In parallel, the same chunks are saved to SQLite with full-text search support. Each chunk stores two text variants: the original (for result display) and the morphologically normalized version (for searching).

### 3.2 Search Pipeline

When a user query arrives, the system performs search in both indexes in parallel and merges the results.

**Step 1 — Query Preprocessing.** The user's question is formatted for semantic search (a model-specific QUERY prefix is added) and for full-text search (tokenization, stop-word removal, stemming).

**Step 2 — Semantic Search.** The query is encoded into a vector by the same model used during indexing. ChromaDB performs nearest-neighbor search using cosine distance.

**Step 3 — Full-Text Search.** Stemmed query tokens are passed to SQLite FTS5. First, strict AND mode is applied (all tokens must be present in the document); if no results are found, a softer OR mode is used. Ranking is done using the BM25 metric.

**Step 4 — Result Merging.** Results from both methods are merged using one of three strategies (Base, RRF, or Reranker). Duplicates are removed.

**Step 5 — LLM Analysis.** After the search pipeline returns the top-K relevant chunks, they are passed to the language model, which formulates the final answer to the user's question.

---

## 4. PDF Document Parsing

**Docling** is an IBM library for extracting structured content from documents, primarily PDFs.

**The problem it solves**

A typical PDF is not text, but a set of rendering instructions: where to place a character, where to draw a line. There is no semantics: no concept of "heading", "table", or "caption". Simple text extraction (pdfplumber, PyMuPDF) produces a jumbled mess — especially for scanned documents or those with complex tables.

Docling addresses this: it doesn't just extract text — it **understands the document structure**.

### 4.1 How It Works Internally

Docling runs several models sequentially:

- **Layout model** — a neural network (based on RT-DETR) that looks at a page as an image and marks up regions: where the heading is, where the paragraph is, where the table and figure are. This is computer vision, not PDF operator parsing.
- **Table Structure Recognition (TSR)** — a separate model that takes a cropped table region and reconstructs its grid: rows, columns, merged cells. The output is a full data matrix that can be exported to Markdown or HTML.
- **OCR** — activated for scanned pages or text inside images. The implementation uses RapidOCR.
- **DoclingDocument Assembly** — all recognized elements are assembled into a unified structure with hierarchy and type labels.

### 4.2 Why This Matters for RAG

Chunking quality directly depends on parsing quality. If the parser cannot distinguish a heading from body text, semantic block boundaries cannot be correctly identified. If a table has been broken into garbage lines, it becomes useless — and even harmful — in the index, while tables in technical literature are highly significant and contain a large amount of important information.

---

## 5. Chunking Logic

Chunking — splitting a document into fragments — is one of the key stages of a RAG system. The quality of chunking directly affects search quality: chunks that are too small lose context, while chunks that are too large reduce precision.

The system implements **structure-aware chunking**, processing each document element type according to its own rules:

| **Element Type** | **Processing Logic** |
|---|---|
| **Section heading** | Captured as context. The current buffer is flushed to a chunk before a section change. |
| **Regular text** | Accumulated in a buffer. When the threshold (500 characters) is reached, the buffer is saved as a chunk. |
| **Table** | Exported to Markdown format. If preceded by a "Table N…" caption, it is included in the heading. |
| **Figure** | Saved only if a caption is present. The text buffer before the figure is flushed as a separate chunk. |
| **Table of Contents** | Skipped entirely. ToC pages are not indexed. |
| **Garbage text** | Filtered via `is_junk_text()` — lines consisting only of digits and spaces are discarded. |

### 5.1 Chunking Parameters

- Maximum text chunk size — **500 characters**
- Merge threshold for small chunks — **300 characters** (adjacent text chunks below the threshold are merged)
- Each chunk stores the section heading in the `context` metadata field

### 5.2 Text Cleaning

Before saving, each chunk undergoes cleaning: Unicode normalization, replacement of non-standard bullet characters, removal of ellipses and extra whitespace, and normalization of table markup to a consistent format.

---

## 6. Data Storage

### 6.1 ChromaDB: Vector Index

ChromaDB is used in PersistentClient mode: data is saved to disk and loaded on the next launch without recomputing embeddings. A separate collection is created for each embedding model, allowing switching between models without reindexing.

Each document in the collection contains: the chunk text, the embedding vector, and metadata (page number, section context, element type).

### 6.2 SQLite FTS5: Full-Text Index

The built-in FTS5 SQLite module is used for full-text search. The table stores two variants of each chunk: the original text (returned to the user) and the stemmed text (used for search). This enables searching over normalized word forms without sacrificing readability in the results.

> **Note:** In production systems of this type, it is common practice to use out-of-the-box solutions such as ElasticSearch or OpenSearch. For testing purposes, the simplified SQLite database is sufficient.

---

## 7. Embedding Models

The system supports three embedding models, each requiring its own input text formatting. This is important: supplying text without the required prefix degrades vector quality.

### intfloat/multilingual-e5-large (E5)

A multilingual model based on the E5 architecture. Requires explicit prefixes: `"passage: "` for indexed chunks and `"query: "` for search queries.

### deepvk/USER2-base (USER2)

A Russian-language model by DeepVK, optimized for information retrieval tasks. Uses prefixes `"search_document: "` and `"search_query: "`.

### ai-sage/Giga-Embeddings-instruct (GIGA)

An instruction-following model supporting arbitrary tasks via a prompt. For search, a special instruct prompt is used: *"Given a question, find the paragraph of text containing the answer."*

*Operates in bfloat16 format to conserve GPU memory.*

---

## 8. Result Merging Strategies

After obtaining candidates from both indexes, the system applies one of three strategies to produce the final ranked list:

| **Strategy** | **Principle** | **Use Case** |
|---|---|---|
| **Base** | Merging by total score; chunks found by both methods receive boosted priority. *Note: FTS has a greater influence since its score ranges from 0 to 15, while semantic search score ranges from −1 to 1. This is a normal strategy for technical documentation.* | Fast mode when time savings and low footprint are critical |
| **RRF** | Reciprocal Rank Fusion: final score = sum of 1/(k+rank) for each method, k=60 — the standard magic constant. | Primary hybrid mode |
| **Reranker** | The candidate pool from both methods is deduplicated and reranked by a cross-encoder model. | Maximum quality |

### 8.1 Full-Text Search and Morphology

For correct FTS search in Russian, the system applies stemming (Snowball algorithm) to all indexed texts and search queries. This enables finding documents regardless of grammatical case, number, or word form.

Stop words (functional parts of speech and interrogative words) are additionally removed from search queries, reducing noise in the results.

### 8.2 Reranking

**The reranker (BAAI/bge-reranker-v2-m3)** is a cross-encoder model: it takes a "question + fragment" pair as input and outputs a scalar relevance score. Unlike bi-encoder embedders, a cross-encoder sees both texts simultaneously and accounts for their interaction, providing more precise ranking.

Reranking is applied to the merged, deduplicated candidate pool from semantic and FTS search, after which the top-K results are returned.

---

## 9. Answer Generation (LLM Layer)

### Model: Qwen2.5-7B-Instruct

The model used is **Qwen2.5-7B-Instruct** — an instruction-finetuned version of the Qwen 2.5 model with 7 billion parameters by Alibaba. The model was chosen as a compromise between answer quality and resource requirements: it fits on a single mid-range GPU and follows system instructions reliably.

Loading is performed via `AutoModelForCausalLM` from the Transformers library. The `torch_dtype="auto"` parameter lets Transformers automatically select the optimal numeric format (bfloat16 on supported GPUs), reducing memory consumption without quality loss. The target device is read from the `DEVICE` environment variable (default: `cuda`).

### Prompt Construction

The model request is built in chat-message format with two roles:

- **system** — sets the role and constraint: the model is presented as a technical support engineer and receives an explicit instruction to answer only based on the provided text. This prevents hallucinations — situations where the model answers from general knowledge rather than from the documentation.
- **user** — contains the retrieved context (concatenated chunks from the search pipeline) and the user's question.

### Generation Parameters

| **Parameter** | **Value** | **Meaning** |
|---|---|---|
| max_new_tokens | 512 | Maximum response length |
| temperature | 0.1 | Low value — responses are deterministic, with minimal variation |
| top_p | 0.9 | Nucleus sampling — filters out low-probability tokens |

A low temperature (0.1) was chosen intentionally: technical documentation requires precision, not variety of phrasing.

---

## 10. Quality Evaluation System

### Benchmark Generation

For objective comparison of configurations, an automated benchmark was implemented. The test set contains **409 question–reference chunk pairs**, generated using GPT-4o from real documentation fragments.

### Evaluation Metrics

- **Hit@K** — the proportion of queries in which the reference chunk appeared in the top-K results (K = 1, 5, 10)
- **MRR (Mean Reciprocal Rank)** — the mean value of the reciprocal rank of the reference chunk; accounts not only for whether the chunk was found, but also its position
- **Avg Time** — average query processing time in milliseconds

The benchmark sequentially tests all "model × strategy" combinations and outputs a summary table for comparison.

---

## 11. Testing Results Report

### 11.1 Experiment Description

Testing was conducted on 409 questions generated from real chunks of ViPNet technical documentation. For each question, the reference chunk (ground truth) is known in advance. The system is considered successful if the target chunk appears in the top-K search results.

The evaluation metrics described above (Hit@K, MRR, Avg Time) were analyzed.

Three embedding models were tested:

- intfloat/multilingual-e5-large (E5)
- deepvk/USER2-base (USER2)
- ai-sage/Giga-Embeddings-instruct (GIGA)

For each model, three ranking strategies were compared:

- **Base** — merging semantic and FTS results by total score
- **RRF (Reciprocal Rank Fusion)** — weighted merging by ranks from two sources
- **Reranker** — reranking the merged pool via cross-encoder BAAI/bge-reranker-v2-m3

### 11.2 Semantic Search (without FTS)

The first stage tested pure semantic search without the full-text component. The goal was to evaluate the quality of the embedding models themselves.

| **Model** | **Hit@1** | **Hit@5** | **Hit@10** | **MRR** | **Avg Time** |
|---|---|---|---|---|---|
| E5 | 59.17% | 87.78% | 92.18% | 0.7079 | 15.83 ms |
| USER2 | 52.81% | 80.93% | 88.02% | 0.6477 | 24.68 ms |
| **GIGA** | **68.95%** | **93.64%** | **96.09%** | **0.7953** | **42.62 ms** |

The GIGA model showed the best results in semantic mode: Hit@1 of 68.95% and MRR of 0.7953 — though it is approximately 2.7× slower than E5 and requires significantly more GPU memory. USER2 showed the weakest quality of the three, underperforming E5 on all metrics while also having higher latency.

### 11.3 Hybrid Search: All Strategies

In the second stage, full-text search (SQLite FTS5 with stemming) was added. Below are the results for all 9 configurations (3 models × 3 strategies).

| **Model** | **Strategy** | **Hit@1** | **Hit@5** | **Hit@10** | **MRR** | **Avg Time** |
|---|---|---|---|---|---|---|
| E5 | Base | 63.08% | 90.46% | 93.64% | 0.7474 | 20.93 ms |
| E5 | RRF | 63.57% | 92.67% | 95.11% | 0.7607 | 19.87 ms |
| **E5** | **Reranker** | **73.59%** | **93.15%** | **95.84%** | **0.8212** | **251.65 ms** |
| USER2 | Base | 64.06% | 90.95% | 94.13% | 0.7552 | 29.95 ms |
| USER2 | RRF | 60.39% | 90.46% | 95.60% | 0.7372 | 23.44 ms |
| **USER2** | **Reranker** | **74.08%** | **93.64%** | **96.09%** | **0.8289** | **269.88 ms** |
| GIGA | Base | 63.81% | 90.46% | 94.13% | 0.7517 | 46.42 ms |
| GIGA | RRF | 71.15% | 94.13% | 96.33% | 0.8094 | 46.03 ms |
| **GIGA** | **Reranker** | **74.08%** | **94.38%** | **96.58%** | **0.8289** | **290.27 ms** |

**An Interesting "Paradox"**

When FTS was added, metrics improved for both USER2 and E5, which is expected. FTS helped both models equally — USER2 simply started from a lower baseline (52.81% vs. 59.17%), making its gain from FTS look more impressive (+11% vs. +4%). In absolute terms, however, they converged.

Notably, USER2 even surpassed E5 in some cases (e.g., with reranking), despite significantly underperforming it in pure semantic tests.

**Possible reason: saturation effect.** E5 already finds most chunks semantically, and FTS adds new candidates, but many of those were already found by E5. The gain exists but is modest, because the overlap between the two methods is higher for a stronger embedder.

USER2 has less overlap — it was missing chunks that FTS found. Therefore, FTS provided genuinely new candidates rather than duplicates of what was already retrieved. In any case, the total difference is only about 0.49%, which is statistically almost nothing over 409 queries (roughly 2 queries). They can be considered equal. Drawing conclusions about USER2's superiority over E5 from such a margin would be incorrect.

**Conclusion:** USER2 did not objectively become better than E5 — it simply benefited more from hybridization because it started weaker. With the reranker, they converged to the same result.

**Final decision before LLM input:**

Since GIGA's advantage is not significant, the decision was made to use the E5 model for cost efficiency.

The main benefit comes from applying reranking. Therefore, for selecting final chunks before passing to the LLM, **E5 + Reranker** is used.

---

## 12. Research Question and Hypothesis

### Motivation

The central engineering observation in this project — that USER2, a weaker embedding model, gains more from hybridization than the stronger E5 — raises a question that goes beyond system optimization:

> **Does the marginal benefit of adding full-text search to a RAG pipeline depend on the quality of the base embedding model? And if so, why?**

This is not obvious. One could expect a stronger embedder to benefit more from FTS, since it already retrieves better candidates and reranking can make better use of them. The observed results suggest the opposite.

### Hypothesis

**H1 — Saturation hypothesis:** A stronger embedding model already achieves high recall on its own. The chunks it misses tend to be genuinely hard cases (ambiguous phrasing, rare terminology) where FTS also struggles. Therefore, FTS adds mostly redundant candidates — chunks already found by semantic search. The overlap between the two retrieval signals is high, and the marginal gain is low.

A weaker embedding model, by contrast, has lower recall and misses more chunks for "easy" reasons — vocabulary mismatch, morphological variation. These are precisely the cases where FTS excels. The overlap between methods is lower, FTS adds genuinely new candidates, and the gain is larger.

**Prediction:** If H1 holds, we should observe that:
- The intersection of semantic and FTS result sets is larger for stronger embedders
- FTS gains are concentrated in queries where the query tokens appear literally in the target chunk
- With reranking (which maximally exploits the candidate pool), the gap between models shrinks — because the bottleneck shifts from retrieval to ranking

The data is consistent with all three predictions. USER2 and E5 converge to identical Hit@1 (74.08% vs 73.59%) and MRR (0.8289 vs 0.8212) under the Reranker strategy, despite a 16-point gap in pure semantic mode.

### Open Questions

This analysis raises several follow-up questions that go beyond what can be answered with the current benchmark:

1. **Is the saturation effect model-specific or universal?** The three models tested differ not only in quality but in architecture and training data. Controlled experiments on a larger model family would be needed to isolate the embedding quality variable.

2. **Does domain specificity interact with this effect?** Technical documentation with rigid terminology (commands, part numbers, abbreviations) may amplify FTS gains in ways that don't generalize to other domains.

3. **What is the optimal candidate pool size for reranking as a function of base model quality?** If weaker embedders miss more relevant chunks in the top-10, expanding the reranking window (e.g., to top-50) might disproportionately benefit them.

4. **Can this effect be exploited deliberately?** A lightweight embedder paired with FTS and a strong reranker might achieve quality comparable to a heavy embedder alone, at lower compute cost — a potentially valuable trade-off for production deployment.

These questions motivate future experiments and represent the primary direction for extending this work into a more formal research contribution.

---



### Impact of Strategy

Reranker consistently outperforms Base and RRF on Hit@1 for all three models (+10–12% relative to Base). RRF provides improvement only for GIGA (+7.34%), while for USER2 it even shows a slight decrease (−3.67%), indicating RRF's sensitivity to the quality of the base model.

### Best Configurations

- **Highest Hit@1 and MRR** — GIGA+Reranker, USER2+Reranker (74.08% / MRR 0.8289) and E5+Reranker (73.59% / MRR 0.8212)
- **Best quality/speed balance** — GIGA+RRF: Hit@1 = 71.15%, MRR = 0.8094, ~46 ms latency
- **Fastest option** — E5+Base (20.93 ms) with acceptable Hit@1 = 63.08%

---

## 13. Summary

In the course of this research, a RAG system for technical documentation was built and tested. Three embedding models (E5-Large, USER2, Giga-Embeddings) and three search strategies (Base Hybrid, RRF, Reranker) were compared. The obtained data allows formulating a clear deployment strategy.

### 10.1 Practical Recommendations

- **For production** without strict latency requirements: any embedder + Reranker is suitable; it all depends on GPU capabilities.
- **For real-time scenarios** (<50 ms): GIGA + RRF — a balance between speed and quality.
- **If GPU is unavailable:** E5 + Base as the most lightweight option.

### 10.2 Further Directions

- **LLM layer improvement:** Integration of retrieved chunks into the final answer. Main focus — training the model to correctly cite page numbers and synthesize data from Markdown tables without hallucinations.
- **RRF fine-tuning:** Experiments with the smoothing parameter k (currently 60). For small technical documentation collections, reducing k may give more weight to top search results and improve Hit@1.
- **Reranker fine-tuning:** Using the generated benchmark (409 question–answer pairs) to fine-tune the Cross-Encoder model on domain-specific terminology: commands, platform article numbers.
- **Top-K optimization for Reranker:** Current testing was done with 10–25 candidates passed to reranking. Increasing this window to 50 may improve accuracy in complex cases where the correct answer was ranked deep in the search results.
- **Transition to semantic chunking:** Experiments with dynamic chunk size based on Docling's logical structure, instead of the hard 500–1200 character limit.
- **Build an API.**

---

## 11. Project Structure

| **Module / File** | **Description** |
|---|---|
| **src/processing/pdf_parser.py** | Docling initialization, PDF parsing, result caching to JSON |
| **src/processing/chunker.py** | Chunk splitting, batch embedding creation |
| **src/processing/cleaners.py** | Text cleaning, stemming, query formatting per model |
| **src/databases/chroma.py** | Saving chunks with vectors to ChromaDB |
| **src/databases/sqlite.py** | FTS5 table initialization, chunk saving |
| **src/databases/__init__.py** | Factory functions for database connections |
| **src/models/embedders.py** | Embedding model loading (E5, USER2, GIGA) |
| **src/models/reranker.py** | Cross-encoder reranker loading |
| **src/models/llm.py** | Qwen2.5 7B model initialization |
| **src/utils/retrivers.py** | Semantic and full-text search functions |
| **src/utils/combinations.py** | Result merging strategies: Base, RRF, Reranker |
| **src/load_data.py** | Entry point for the indexing pipeline |
| **src/tests_for_rag.py** | Benchmark: testing all configurations, metric computation |
| **src/test_for_qwen.py** | Benchmark: query generation via Qwen2.5 7B based on retrieved chunks |
| **benchmarks_generation/generate_sintetic.py** | Generating test "question–answer" pairs via GPT-4o |
| **logger_config.py** | Colorized logging configuration |
