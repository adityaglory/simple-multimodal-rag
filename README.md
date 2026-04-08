# Multimodal RAG Assistant (Local & Containerized)

A production-ready, fully local Multimodal Retrieval-Augmented Generation (RAG) application. This system allows users to securely query local PDF documents and analyze images using a conversational AI interface, all running entirely on your local machine without sending data to external APIs.

## Key Features

* **Multimodal Vision:** Upload images alongside your text prompts. The AI can "see" the image and combine that context with your document database.
* **Stateful Memory:** Implements a Sliding Window memory architecture, allowing for natural, multi-turn conversations without overflowing the LLM context window.
* **Privacy-First & Local:** Uses Ollama to run models locally. No data ever leaves your machine.
* **Enterprise MLOps:** * Fully containerized with **Docker & Docker Compose**.
  * Automated **CI/CD pipeline** via GitHub Actions for continuous testing.
  * Experiment tracking ready via **MLflow**.
* **Modern UI:** Built with a responsive, chat-based **Streamlit** interface.

## Tech Stack

* **LLM Engine:** Ollama (Models: `llava` for vision/text, `nomic-embed-text` for embeddings)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit
* **Infrastructure:** Docker, Docker Compose, GitHub Actions, Pytest

## Quickstart Guide

### Prerequisites
1. Install [Docker](https://www.docker.com/).
2. Install [Ollama](https://ollama.com/) and pull the required models to your host machine:
   ```bash
   ollama pull llava
   ollama pull nomic-embed-text
   ```
### Running the Application
1. Clone this repository:
```bash
git clone [https://github.com/adityaglory/simple-multimodal-rag.git](https://github.com/adityaglory/simple-multimodal-rag.git)
cd simple-multimodal-rag
```
2. Add your PDF documents to the /data folder.
3. Build and launch the container:
```bash
docker compose up --build
```
4. Open your browser and navigate to:
```bash
http://localhost:8501
```

### Project Structure
- /src - Core Python backend (vlm.py, app.py, ingest.py)

- /data - Drop your raw PDF files and images here

- /local_rag_db - ChromaDB persistent storage

- /tests - Pytest unit tests for CI/CD

- .github/workflows - CI/CD YAML configurations
