# Stock Price LLM Analysis

A full-stack project that integrates stock data, news sentiment analysis, and LLM-powered insights.
The system provides a Streamlit dashboard for visualization, a FastAPI backend for API services, and supports Dockerized deployment.

## 🚀 Tech Stack

Backend: FastAPI + LangChain + OpenAI API + RAG (FAISS/Pinecone for vector search)

Frontend: Streamlit Dashboard (interactive visualization)

Database: PostgreSQL (structured data) + FAISS/Pinecone (vector embeddings for RAG)

Deployment: Docker(local)

## 🛠 Run Locally
 1. Install dependencies
pip install -r requirements.txt

 2. Start backend & frontend
bash run.sh

## 🐳 Run with Docker
1.Clone this repo:
  git clone https://github.com/loverui129/Stock-Price-LLM-Analysis.git
  cd Stock-Price-LLM-Analysis

2.Build the Docker image:
  docker build -t llm-finance 

3.Run the container (replace YOUR_API_KEY with your OpenAI key):
 docker run -p 8000:8000 -p 8501:8501 \
   -e OPENAI_API_KEY=YOUR_API_KEY \
   llm-finance

## 🌐 Access

Frontend (Streamlit Dashboard) → http://localhost:8501

Backend (FastAPI docs) → http://localhost:8000/docs
