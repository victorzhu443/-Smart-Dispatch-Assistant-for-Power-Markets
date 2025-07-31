# Smart Dispatch Assistant for Power Markets

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Motivation & Goals](#motivation--goals)  
3. [Key Features](#key-features)  
4. [Tech Stack](#tech-stack)  
5. [System Architecture & Phases](#system-architecture--phases)  
6. [Quick Start](#quick-start)  
7. [Folder Structure (Suggested)](#folder-structure-suggested)  
8. [Testing & Verification](#testing--verification)  
9. [Deployment](#deployment)  
10. [Success Metrics](#success-metrics)  
11. [Traceability & Documentation](#traceability--documentation)  
12. [Contribution Guidelines](#contribution-guidelines)  
13. [Troubleshooting](#troubleshooting)  
14. [Authors & Contact](#authors--contact)  
15. [License](#license)

---

## Project Overview

The **Smart Dispatch Assistant** is an AI-powered decision support system for power markets. It ingests real-time energy market data, produces short-term forecasts, and enables natural-language interaction through a Retrieval-Augmented Generation (RAG) LLM interface to assist with intelligent dispatch decisions.

## Motivation & Goals

Power markets are volatile and require timely, data-driven dispatch decisions. This project integrates real-time data ingestion, ML forecasting, and contextual LLM-backed consultation to:

- Improve forecast accuracy over naive baselines  
- Provide explainable, interactive decision support  
- Deploy as a modular, scalable, containerized system  

## Key Features

- Real-time ingestion of public energy market APIs (e.g., ERCOT)  
- Feature engineering with sliding windows and technical indicators  
- Forecasting via a sequence model (e.g., LSTM)  
- Retrieval + LLM integration (RAG) for Q&A over market state  
- Web interface (Django) for forecasts and chat interaction  
- Containerized backend services with Docker and Kubernetes  
- Monitoring of model and API performance (Prometheus metrics)  
- Simulation and retrospective analysis for dispatch strategy evaluation  

## Tech Stack

- **Data ingestion / ETL:** Python, `requests`, Pandas, SQLAlchemy, PostgreSQL  
- **ML Forecasting:** PyTorch (LSTM), training pipelines  
- **Embeddings & RAG:** SentenceTransformers, vector store (e.g., Pinecone or alternative)  
- **LLM:** Hugging Face models (fine-tuned), prompt/response pipeline  
- **Backend API:** Django + REST endpoints  
- **Containerization:** Docker, Kubernetes (minikube for local)  
- **Monitoring:** Prometheus (latency, model metrics)  
- **Infrastructure:** Cron / Airflow for scheduled ingestion  
- **Deployment Orchestration:** Helm / kubectl (optional extensions)  

## System Architecture & Phases

Aligns with the PRD’s phased breakdown. Each phase contains atomic, testable steps:

1. **API Data Ingestion**  
   - Discover & validate market API endpoints  
   - Retrieve, parse, sanitize, and persist data  

2. **Feature Engineering (ETL)**  
   - Load raw data, generate sliding windows  
   - Compute statistics (mean, std, trends)  
   - Persist feature matrix  

3. **Forecasting Model**  
   - Load features, define LSTM architecture  
   - Train, validate (RMSE vs baseline), serialize model  

4. **LLM + Retrieval Integration (RAG)**  
   - Embed market data  
   - Fine-tune LLM on dispatch Q&A  
   - Expose `/query` endpoint with contextual retrieval  

5. **Backend Services**  
   - Dockerize `/forecast` and `/query` APIs  
   - Deploy locally or to cluster (Kubernetes)  

6. **Frontend & Django**  
   - Web UI: dashboard, forecast chart, chatbot  
   - Wire UI to backend via AJAX/REST  

7. **Monitoring & Logging**  
   - Expose latency and model metrics  
   - Historical tracking of performance  

8. **Simulation & Retrospective**  
   - Simulate dispatch decisions vs actuals  
   - Build traceability matrix linking requirements → tests  

## Quick Start

### Prerequisites

- Python 3.10+ (recommend using Conda)
- Docker  
- Kubernetes (e.g., `minikube` for local)  
- PostgreSQL instance  
- GPU (optional but recommended for model training)  
- Environment variables (example):
  ```bash
  export DATABASE_URL=postgresql://user:password@localhost:5432/market_db
  export VECTORSTORE_API_KEY=your_pinecone_key
  export LLM_MODEL_NAME=your_fine_tuned_model
