# Semantix: Vision-Language Image–Text Retrieval System

> A multimodal retrieval system that enables semantic search between images and text using Vision-Language Models (VLMs), vector embeddings, and FAISS similarity search.

## Overview

Semantix is a Vision-Language based retrieval system developed to perform semantic matching between images and textual descriptions. The project maps both modalities into a shared embedding space, enabling efficient cross-modal retrieval tasks such as:

* Text-to-Image Retrieval
* Image-to-Text Retrieval
* Top-K Semantic Search
* Embedding Space Analysis

The system leverages OpenCLIP, FAISS, and FastAPI to deliver accurate and scalable multimodal retrieval.

---

## Features

* Semantic Text-to-Image Retrieval
* Semantic Image-to-Text Retrieval
* Shared Multimodal Embedding Space
* Top-K Ranked Results
* FAISS-Based Vector Similarity Search
* OpenCLIP Vision-Language Model Integration
* FastAPI Backend Services
* React + Vite Frontend
* Real-Time Search Interface
* Scalable Retrieval Pipeline

---

## System Architecture

```text
User Query
     │
     ▼
┌─────────────────┐
│ Text / Image    │
│ Input Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OpenCLIP        │
│ Encoder         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Shared Embedding│
│ Space           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FAISS Index     │
│ Similarity      │
│ Search          │
└────────┬────────┘
         │
         ▼
   Top-K Results
```

---

## Technology Stack

| Category              | Technology              |
| --------------------- | ----------------------- |
| Frontend              | React, TypeScript, Vite |
| Backend               | FastAPI, Python         |
| Deep Learning         | PyTorch                 |
| Vision-Language Model | OpenCLIP ViT-B-32       |
| Vector Database       | FAISS                   |
| Data Processing       | NumPy, Pandas           |
| Visualization         | Matplotlib, UMAP, t-SNE |
| API Layer             | REST APIs               |
| Version Control       | Git, GitHub             |

---

## Dataset Information

| Dataset        | Usage                |
| -------------- | -------------------- |
| Flickr30K      | Image–Text Retrieval |
| Custom Queries | Evaluation & Testing |

### Dataset Statistics

| Metric           | Value                   |
| ---------------- | ----------------------- |
| Images Indexed   | 31,783                  |
| Captions Indexed | 158,914                 |
| Retrieval Engine | FAISS                   |
| Search Modes     | Text→Image, Image→Text  |
| Ranking Strategy | Top-K Similarity Search |

---

## Retrieval Tasks

### Text-to-Image Retrieval

Given a natural language query, the system retrieves the most semantically relevant images.

**Example Query**

```text
two dogs playing in snow
```

**Output**

* Top-K matching images
* Similarity-ranked results

---

### Image-to-Text Retrieval

Given an image query, the system retrieves the most relevant textual descriptions.

**Output**

* Top-K matching captions
* Ranked semantic matches

---

## Project Structure

```text
Semantix-VLM-Retrieval/
│
├── backend/
│   ├── routes/
│   ├── services/
│   ├── models/
│   └── main.py
│
├── src/
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   └── assets/
│
├── ml_pipeline/
│
├── storage/
│
├── requirements.txt
├── package.json
├── vite.config.ts
└── README.md
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Aditya0825-crypto/Semantix-VLM-Retrieval.git

cd Semantix-VLM-Retrieval
```

---

### 2. Create Python Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Start Backend Server

```bash
uvicorn backend.main:app --reload --port 8000
```

Backend runs at:

```text
http://localhost:8000
```

---

### 5. Install Frontend Dependencies

```bash
npm install
```

---

### 6. Start Frontend

```bash
npm run dev
```

Frontend runs at:

```text
http://localhost:8080
```

---

## API Workflow

| Step | Description                                         |
| ---- | --------------------------------------------------- |
| 1    | User submits text or image query                    |
| 2    | Query is converted into embeddings                  |
| 3    | Embeddings are projected into a shared vector space |
| 4    | FAISS performs nearest-neighbor search              |
| 5    | Top-K results are ranked by similarity              |
| 6    | Results are returned to the frontend                |

---

## Evaluation Metrics

| Metric            | Purpose                     |
| ----------------- | --------------------------- |
| Recall@K          | Measures retrieval coverage |
| Precision@K       | Measures retrieval accuracy |
| Cosine Similarity | Semantic similarity score   |
| Top-K Accuracy    | Ranking effectiveness       |

---

## Future Work

* Integration of advanced Vision-Language Models
* Enhanced embedding visualization techniques
* Large-scale retrieval benchmarking
* Multilingual retrieval support
* Cloud deployment and scalability improvements
* Research paper publication and conference submission

---



## Acknowledgements

This project explores recent advances in Vision-Language Models, multimodal learning, semantic retrieval systems, and vector similarity search. It draws inspiration from contemporary research in cross-modal representation learning and embedding-based retrieval.

---

## License

This project is intended for educational, academic, and research purposes.
