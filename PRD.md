# Product Requirements Document (PRD): Smart Dispatch Assistant for Power Markets

## Document Metadata
- **Author**: Victor Zhu, Product Manager
- **Version**: 1.0
- **Date**: July 31, 2025

---

## 1. Objective & Scope

**Objective**: Develop an AI-powered system that ingests real-time energy market data, forecasts future price/demand curves, and enables intelligent dispatch decision support via LLM-based interaction.

**Key Deliverables**:
- Real-time API ingestion from power market sources (e.g. ERCOT)
- ML-based forecasting using historical market data
- Retrieval-Augmented Generation (RAG) system using LLMs
- Web UI (via Django) to access forecasts and LLM responses
- Containerized microservices deployed via Docker and Kubernetes

Each deliverable is decomposed into micro-steps with testable checkpoints.

---

## 2. Stakeholders
Victor Zhu
- PM 
- ML Engineers
- Data Engineers
- Backend Developers
- Frontend Developers
- DevOps Engineers

---

## 3. Step-by-Step Breakdown With Verification

### Phase 1: API Data Ingestion

**Step 1.1: Identify Public Energy Market API**
- Example: ERCOT, IEA, ElectricityMaps
- **Test Case**: Confirm endpoint URL responds with sample query

**Step 1.2: Perform Basic API Call**
- Use Python requests.get()
- **Test Case**: HTTP 200 and non-empty JSON returned

**Step 1.3: Parse JSON to Pandas DataFrame**
- Extract keys: timestamp, price, demand
- **Test Case**: Print DataFrame head with expected schema

**Step 1.4: Handle Missing/Invalid Fields**
- Drop or impute NaNs
- **Test Case**: No NaNs in critical columns after processing

**Step 1.5: Save DataFrame to SQL Table market_data**
- Use SQLAlchemy with PostgreSQL
- **Test Case**: Query from DB matches DataFrame row count

**Step 1.6: Schedule Ingestion with Cron or Airflow**
- Trigger every hour
- **Test Case**: Logs show ingestion timestamped runs

### Phase 2: Feature Engineering (ETL)

**Step 2.1: Load Data from SQL**
- Query past 7 days
- **Test Case**: Resulting DataFrame has > 1000 rows

**Step 2.2: Generate Sliding Windows (24h to predict next hour)**
- Use rolling() + lag shift
- **Test Case**: First row contains exactly 24 timestamps

**Step 2.3: Compute Technical Features (mean, std, trend)**
- Apply over each window
- **Test Case**: Check columns for features exist

**Step 2.4: Write Feature Matrix to SQL Table features**
- **Test Case**: Verify number of rows = number of sliding windows

### Phase 3: Forecasting Model

**Step 3.1: Load Feature Matrix into ML Training Script**
- **Test Case**: Script reads rows, shape is (N, M)

**Step 3.2: Define LSTM Model Architecture (PyTorch)**
- Input: windowed features; Output: price forecast
- **Test Case**: model.summary() confirms layers and dimensions

**Step 3.3: Train Model on Sample Data (small batch)**
- Train for 5 epochs
- **Test Case**: Loss decreases over epochs

**Step 3.4: Evaluate RMSE on Validation Set**
- **Test Case**: RMSE < naive last-hour prediction baseline

**Step 3.5: Save Model to File model.pt**
- **Test Case**: File exists and reloads without error

### Phase 4: LLM + Retrieval Integration (RAG)

**Step 4.1: Embed Market Data with SentenceTransformers**
- Store in Pinecone
- **Test Case**: Vector length and ID return correctly from Pinecone

**Step 4.2: Fine-Tune Hugging Face LLM on Dispatch Q&A**
- Train with 100 example Q&A pairs
- **Test Case**: Train loss decreases

**Step 4.3: Implement Retrieval API Endpoint /query**
- Inputs: user question; Output: LLM-generated answer
- **Test Case**: /query?q=What is the forecast? returns non-empty text

**Step 4.4: Measure Response Perplexity**
- Use GPT-generated reference
- **Test Case**: Perplexity < base model baseline

### Phase 5: Backend Services (Docker/Kubernetes)

**Step 5.1: Dockerize /forecast API**
- Input: timestamp; Output: predicted price
- **Test Case**: curl /forecast returns JSON with price

**Step 5.2: Dockerize /query API**
- Connect to RAG pipeline
- **Test Case**: Chatbot interface returns answers

**Step 5.3: Deploy to Local Kubernetes (minikube)**
- 2 services + ingress config
- **Test Case**: Accessible via localhost/query and localhost/forecast

### Phase 6: Frontend & Django

**Step 6.1: Create Django Web App**
- Pages: dashboard, forecast graph, chatbox
- **Test Case**: localhost:8000 loads basic UI

**Step 6.2: Connect UI to Backend APIs**
- AJAX call to /forecast, /query
- **Test Case**: Chat interface and graph update dynamically

### Phase 7: Monitoring & Logging

**Step 7.1: Add Prometheus Logging for API Latency**
- **Test Case**: Prometheus dashboard shows live metrics

**Step 7.2: Track ML Model Metrics Over Time**
- Track RMSE, accuracy
- **Test Case**: CSV or dashboard shows historical scores

### Phase 8: Simulation & Retrospective

**Step 8.1: Simulate Dispatch Based on Forecasts**
- Compare forecast vs actual
- **Test Case**: Net savings > baseline bid strategy

**Step 8.2: Document Full Traceability Matrix**
- Link each step to requirement and test
- **Test Case**: All requirements marked covered with passing tests

---

## 4. Success Metrics
- 100% steps have passing tests
- RMSE < baseline
- Perplexity < unfine-tuned LLM
- Forecast latency < 500ms
- End-to-end system demo completes without failure

---

## 5. Dependencies
- Public energy market API (ERCOT, IEA, etc.)
- GPUs for training
- Docker/Kubernetes runtime

---

**This PRD ensures every task is atomic, testable, and traceable. You can now confidently assign each piece and validate progress incrementally.**

---

*Document prepared by Victor [Your Last Name] - Software Engineer Intern Candidate*  
*Specializing in renewable energy systems and AI-powered grid optimization*