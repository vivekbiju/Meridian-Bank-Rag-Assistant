# Meridian Bank RAG Assistant

An enterprise-grade, evaluation-gated Retrieval-Augmented Generation (RAG) assistant built to answer customer inquiries accurately from the Meridian Bank Handbook.

This repository implements a production RAG pipeline, complete with dynamic PDF extraction, vector search via `pgvector`, automated metrics evaluation, and a GitHub Actions CI/CD quality gate that blocks regressions from entering production.

---

## Live Deployments

* **Vercel (Frontend App):** https://meridian-bank-rag-assistant-39x1leq5i-vivekbiju.vercel.app/
* **Render (Backend API):** https://meridian-bank-rag-assistant-1.onrender.com/docs

---

## Architecture Overview

* **Frontend:** Next.js application deployed on **Vercel** featuring loading states, dynamic regex Markdown rendering, and proxy rewriting.
* **Backend:** FastAPI microservice running inside Docker on **Render**.
* **Vector Database:** PostgreSQL hosted on **Neon Postgres** using the `pgvector` extension and HNSW indexing (`vector_cosine_ops`).
* **Embeddings:** Local CPU-based FastEmbed (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions).
* **LLM Engine:** Groq API (`groq/compound`) for fast inference.
* **CI/CD Quality Gate:** GitHub Actions runner executing `scripts/eval.py` on every push/PR against `thresholds.yaml`.

---

## Key Features & Pipeline Details

### 1. Document Extraction & Idempotent Ingestion (`ingest.py`)
* **Extraction:** Ingests raw multi-page PDF documents (`meridian-handbook.pdf`) via `pdfplumber`.
* **Chunking & Storage:** Splits text into 512-character chunks with 64-character overlap. Chunks are assigned section metadata and saved in Neon Postgres.
* **Idempotency:** Utilizes a `UNIQUE (section, chunk_index)` constraint to ensure repeated ingestion runs do not duplicate vectors in the index.

### 2. Distance-Filtered Retrieval & Strict Citation Grounding
* **Cosine Distance Filter:** Retrieves top $k=5$ chunks via cosine distance (`<=>`). A strict distance threshold of $0.85$ strips out low-relevance noise before prompting.
* **Strict Prompt Verification:** Mandates exact inline section bracket citations (e.g., `[1. Your cards]`) and enforces explicit fallbacks (*"I am sorry, but the Meridian Bank handbook does not contain information to answer your request."*) when no context passes distance filtering.
* **Prompt Injection Resilience:** System prompts command the model to ignore override instructions embedded within retrieved context blocks.

### 3. Async Document Ingestion API
* **`POST /documents`:** Accepts PDF uploads ($\le 5\text{MB}$), generates a unique `job_id`, and processes ingestion asynchronously in the background to prevent serverless timeouts.
* **`GET /documents/{job_id}`:** Allows frontend clients to poll ingestion status (`processing`, `completed`, `failed`).

---

## Benchmark Evaluation & CI/CD Quality Gate

The system is continuously benchmarked against a **59-question Golden Set** (`golden_set.json`) covering:
* **12 Baseline Questions:** Standard Option C test suite.
* **25 Answerable Questions:** Factual queries tagged with ground-truth sections.
* **10 Unanswerable Questions:** Out-of-scope bank queries testing refusal correctness.
* **4 Contradiction Questions:** Tests handling of internal handbook contradictions from both directions.
* **8 Adversarial Cases:** Attacks including corpus prompt injections, policy modification requests, non-English inputs, and oversized payloads.

---

## CI Gate Configuration (`thresholds.yaml`)

```yaml
# Performance and Quality Thresholds for CI Gate
retrieval:
  recall_at_k: 0.85 # Minimum recall required to pass build
  mrr: 0.80

generation:
  refusal_correctness: 0.90
  answer_correctness: 0.80
  ```
Automated builds (`.github/workflows/eval.yml`) execute `scripts/eval.py --offline-fail-under-config thresholds.yaml` on every commit. If retrieval recall or refusal correctness drops below thresholds, `eval.py` raises `sys.exit(1)` and blocks deployment.

---

## Operational Metrics & Production Runbook (Part 7)

1. Benchmark Performance & Cost (50-Request Sample)
   * **p50 Latency:** 3.84s (3,836 ms)
   * **p95 Latency:** 4.51s (4,512 ms)
   * **Error Rate:** 0.0% (50/50 successful responses.
   * **Cost at List Prices:**
       * **Render & Neon Postgres:** Free Tier ($0.00)
       * **Groq API (`groq/compound`):** Free Tier / < $0.01 total list price for 50 benchmark requests.
         
2. Cold Start Penalty
   * **Cold Request (idle > 1 hour):** 46.20s (Render free container spin-up + FastEmbed ONNX runtime load)
   * **Warm Request:** 3.84 s
   * **Cold Start Penalty Overhead:** +42.36s
     
3. Production Monitoring Signals & Alert Thresholds
  * **HTTP 5xx Error Rate:** Alert threshold > 2.0 % over a 5-minute rolling window (indicates database connection drops or Groq API outages).
  *  **p95 Response Latency:** Alert threshold > 8.0 seconds over a 10-minute rolling window (indicates vector search degradation or upstream LLM throttling).
  *  **Refusal / Fallback Rate:** Alert threshold > 15.0 % over a 15-minute rolling window (indicates embedding drift or document ingestion failures).
    
4. Runbook Entry: Upstream LLM API Failure / Rate Limiting
  * **What Breaks:** Groq API returns 429 Rate Limit Exceeded or 503 Service Unavailable, causing FastAPI to return 500 Internal Server Error responses to the client.
  * **How You Notice:** HTTP 5xx Error Rate alert triggers (> 2% in 5 min), or Render application logs display repeated GroqAPIError exceptions.
  * **Immediate Response Steps:**
    1. **Check Provider Status:** Verify active outages on the upstream Groq status page.
    2. **Inspect Error Traces:** Review Render deployment logs to distinguish between quota limits (429) and server errors (5xx).
    3. **Execute Failover:** Update GROQ_MODEL or secondary fallback environment variables in Render to route inference through an alternative model tier or backup API key.

---

## Local Setup & Development

**1. Prerequisites**
   * Python 3.11+
   * PostgreSQL with `pgvector` enabled (or a free Neon Postgres instance)
     
**2. Environment Variables**
   Create a `.env` file in the project root:
   ```yaml
   DATABASE_URL="postgresql://user:password@ep-cool-name.neon.tech/neondb?sslmode=require"
   GROQ_API_KEY="gsk_your_groq_api_key_here".
   API_URL="[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)"
   ```
**3. Installation & Database Ingestion**
  ```yaml
   # Install dependencies
    pip install -r requirements.txt

  # Execute idempotent document ingestion
    python ingest.py
   ```

**4. Running Backend Server & Evaluation Gate**
 ```yaml
# Start FastAPI backend server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Run evaluation suite locally against gate configuration
python scripts/eval.py --offline-fail-under-config thresholds.yaml
```

---
