# ChromaDB Performance Benchmarking: Scaling Vector Search


## 📖 Project Overview

As Retrieval-Augmented Generation (RAG) systems scale, the underlying Vector Database becomes a critical bottleneck. This project is a systematic study of **ChromaDB's** performance behavior. By using real-world scientific data (arXiv papers), we simulate the growth of a knowledge base to observe how data volume impacts search efficiency and result quality.


---

## 🎯 Motivation

Most developers start with small-scale vector stores where search is instantaneous and 100% accurate. However, in production:
* **Latency** can creep up as the index grows.
* **Accuracy** might drop due to the approximate nature of the **HNSW** (Hierarchical Navigable Small World) algorithm.

This project provides the tools to quantify these trade-offs, helping developers decide when to scale their infrastructure or tune their indexing parameters.

---

## 📊 Key Metrics Explained

We focus on two primary dimensions of performance:

### 1. Search Latency (Speed)
We measure the time (in milliseconds) it takes to retrieve the top $K$ neighbors. As the collection grows from 1,000 to 10,000+ pages, we track whether the latency scales linearly, sub-linearly, or exponentially.

### 2. Recall@K (Accuracy)
Since ChromaDB uses Approximate Nearest Neighbor (ANN) search for speed, it may occasionally miss the "true" closest vectors. We calculate accuracy by comparing ANN results against a **Brute-force (Exact)** search.

$$\text{Recall@K} = \frac{|\text{ANN Results} \cap \text{Ground Truth Results}|}{K}$$



---

## 🛠 Experimental Workflow

The benchmarking is conducted through a structured four-stage pipeline:

1.  **Data Ingestion**: Programmatic downloading of PDF documents from the arXiv repository.
2.  **Incremental Indexing**: Building multiple ChromaDB collections in stages (e.g., Step 1: 1k pages, Step 2: 2k pages) to create a growth curve.
3.  **Stress Testing**: Executing a standardized set of test queries against each collection size.
4.  **Comparative Analysis**: Generating visualization plots that map **Data Size vs. Speed** and **Data Size vs. Recall**.

---

## 🧰 Technical Ecosystem

* **Vector Engine**: ChromaDB (utilizing HNSW)
* **Embeddings**: `Sentence-Transformers` (SBERT) 
* **Data Source**: arXiv Open Access via `requests` and `PyPDF2`.
* **Analysis**: `Pandas` for data aggregation and `Matplotlib/Seaborn` for performance visualization.

---


