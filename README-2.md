# BabyYoday — Your Data, Your Agent

## Problem Statement

A small business owner has their own data — product catalogs, pricing sheets, policies,
FAQs, support history, internal docs — and wants a **private AI agent** that serves their
customers using **only that data**. No general-purpose chatbot answers. No hallucinations
from the internet. Just their business, their knowledge, their rules.

The system must work for **any domain** — a bakery, a law firm, a travel agency, a SaaS
product, a local gym. The business owner uploads their data, and out comes a Dockerized
agent that:

- Answers customer questions **only** from the business's own data.
- Refuses to answer anything it doesn't have data for.
- Stays current as the business feeds it new data (daily menus, updated prices, new policies).
- Ships as a self-contained Docker container with a simple API (plug into a website, app, or chat widget).
- Supports multi-step reasoning (e.g. "Can I book a class on Saturday and pay with my membership?" requires checking schedule + payment rules).

---

## Who Is This For?

| Business | Their data | What the agent does |
|----------|-----------|-------------------|
| **Bakery** | Menu, prices, allergen info, opening hours, custom cake policies | "Do you have gluten-free cupcakes? What's the lead time for a wedding cake?" |
| **Law firm** | Practice areas, fee structures, intake procedures, FAQs | "Do you handle trademark disputes? What documents do I need for a consultation?" |
| **Travel agency** | Packages, destinations, cancellation policies, visa requirements | "What's included in the Bali package? Can I cancel within 48 hours?" |
| **SaaS product** | Documentation, pricing tiers, API guides, changelog | "How do I integrate the webhook? What's the rate limit on the Pro plan?" |
| **Local gym** | Class schedules, membership tiers, trainer bios, facility rules | "Is there a yoga class on Tuesdays? Can I freeze my membership?" |

The business owner doesn't need to know ML. They upload their data, we deliver a
Docker container running on **their own hardware** — a Mac Mini under the counter,
an AWS Lightsail instance, whatever they have. It's theirs. Fully isolated. Nobody
else's data touches it.

---

## Core Design Principle

> **RAG restricts the model to your business's knowledge.**
> **A domain gate refuses everything else.**
> **Fine-tuning (optional) improves how well it speaks your business's language.**

RAG is the load-bearing wall. Without it, even a fine-tuned model still happily answers
general questions from its pretraining data. The model must never answer from its own
memory — only from what the business has provided.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BUSINESS'S CUSTOMER                                │
│               (website chat widget, app, API call)                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY / AUTH                                │
│              (mTLS, API keys, rate limiting, RBAC)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              1. DOMAIN GATE (classifier)                     │       │
│   │                                                              │       │
│   │  • Embed the incoming query                                  │       │
│   │  • Compute cosine similarity against domain centroid         │       │
│   │  • If similarity < threshold → REJECT ("out of domain")     │       │
│   │  • Optionally: lightweight topic classifier (fastText/SVM)   │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ passes                                        │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              2. RETRIEVAL (RAG pipeline)                     │       │
│   │                                                              │       │
│   │  • Embed query using the same embedding model                │       │
│   │  • Search vector DB for top-K relevant chunks                │       │
│   │  • If no chunks above relevance threshold → REFUSE           │       │
│   │  • Assemble context window: retrieved chunks + metadata      │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ context                                       │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              3. PROMPT CONSTRUCTION                          │       │
│   │                                                              │       │
│   │  ┌────────────────────────────────────────────────────┐      │       │
│   │  │ SYSTEM: You are {business_name}'s assistant.        │      │       │
│   │  │ Answer ONLY using the provided context. If the     │      │       │
│   │  │ context doesn't contain the answer, say "I don't   │      │       │
│   │  │ have that information." Cite source IDs.            │      │       │
│   │  ├────────────────────────────────────────────────────┤      │       │
│   │  │ CONTEXT: [retrieved chunks with source IDs]        │      │       │
│   │  ├────────────────────────────────────────────────────┤      │       │
│   │  │ USER: {original query}                             │      │       │
│   │  └────────────────────────────────────────────────────┘      │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ prompt                                        │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              4. SLM INFERENCE                                │       │
│   │                                                              │       │
│   │  • Small model (Mistral-7B / Phi-3 / Llama-3-8B)            │       │
│   │  • Optionally LoRA fine-tuned on domain Q&A pairs            │       │
│   │  • Generates answer grounded in retrieved context            │       │
│   │  • Returns: answer + citations + confidence                  │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ response                                      │
│                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │              5. RESPONSE VALIDATOR                           │       │
│   │                                                              │       │
│   │  • Verify citations map to real source IDs                   │       │
│   │  • Check for hallucination (answer vs. retrieved context)    │       │
│   │  • Strip any out-of-domain content that leaked through       │       │
│   │  • Attach provenance metadata to response                    │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │                                               │
│     THIS BUSINESS'S DOCKER CONTAINER (on their own machine)            │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
                    Response to Client
```

---

## How the Business Feeds Data

The business owner adds data in two ways: during initial setup (we build the image) and
at runtime (they drop files in or use the admin panel). The container handles re-indexing
automatically.

**At build time (we do this):**

```
┌──────────────────────────────────────────────────────────────────────────┐
│              CUSTOMER'S INITIAL DATA                                      │
│   (PDFs, spreadsheets, website content, FAQs, product catalogs,          │
│    policy docs, support history — whatever they have)                     │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 build_customer.py (our build tool)                        │
│                                                                          │
│   ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌───────────┐  │
│   │   Parse      │──▶│   Chunk      │──▶│   Embed    │──▶│  Build    │  │
│   │   docs       │   │   (256-512   │   │  (MiniLM)  │   │  FAISS    │  │
│   │              │   │    tokens)   │   │            │   │  index    │  │
│   └─────────────┘   └──────────────┘   └────────────┘   └─────┬─────┘  │
│                                                                │        │
│                                              ┌─────────────────┘        │
│                                              ▼                          │
│                                    ┌──────────────────┐                 │
│                                    │  Compute domain  │                 │
│                                    │  centroid for     │                 │
│                                    │  the gate         │                 │
│                                    └────────┬─────────┘                 │
│                                             │                           │
│                                             ▼                           │
│                                    ┌──────────────────┐                 │
│                                    │  Bake into       │                 │
│                                    │  Docker image    │                 │
│                                    └──────────────────┘                 │
└──────────────────────────────────────────────────────────────────────────┘
```

**At runtime (customer does this — inside their running container):**

```
  Business owner drops new files        OR       Uses admin panel to upload
  into /data/incoming/ folder                    via browser
           │                                              │
           └──────────────────┬───────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  File watcher    │
                    │  (watchdog/cron) │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  Parse → Chunk → Embed   │
              │  → Update FAISS index    │
              │  → Recompute centroid    │
              └──────────────────────────┘
                             │
                             ▼
                  Agent now knows the new data
```

---

## Three Layers of "Only My Data" Enforcement

| Layer | Mechanism | What it catches | Speed |
|-------|-----------|----------------|-------|
| **1. Domain Gate** | Embedding similarity against the business's data centroid | Off-topic queries ("tell me a joke", "what's the weather") | ~5ms |
| **2. RAG Retrieval Threshold** | No relevant chunks found in FAISS → refuse | Queries that sound plausible but have no matching data | ~50ms |
| **3. System Prompt + Grounding** | Model instructed to use only provided context | Prevents the model from using pretraining knowledge to fill gaps | Part of inference |

All three layers fire sequentially. A query must pass all three to get an answer.
The business owner never worries about this — it's built into the platform.

---

## Component Details

### 0. Delivery Model — One Container Per Customer

Every customer gets their own fully isolated deployment. No shared infrastructure.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOW A CUSTOMER GETS THEIR AGENT                   │
│                                                                      │
│  1. Customer signs up, tells us their business name & type           │
│  2. Customer uploads their data:                                     │
│     • Drag & drop PDFs, CSVs, DOCX, TXT                             │
│     • Paste URLs to scrape (website, FAQ pages)                      │
│     • Connect: Google Drive, Notion, Airtable (future)              │
│  3. We build their container:                                        │
│     • Parse documents                                                │
│     • Chunk into segments                                            │
│     • Generate embeddings                                            │
│     • Build their FAISS index                                        │
│     • Compute domain centroid for the gate                           │
│     • Bake it all into a Docker image                                │
│  4. We deploy to THEIR machine:                                      │
│     • Mac Mini under the counter                                     │
│     • AWS Lightsail ($5–20/mo instance)                              │
│     • Any VPS, home server, or office machine                        │
│  5. Customer gets:                                                   │
│     • Their agent running on their hardware                          │
│     • An API key for their agent                                     │
│     • Embeddable chat widget snippet for their website               │
│     • A simple admin panel to upload new data                        │
└─────────────────────────────────────────────────────────────────────┘
```

```
  CUSTOMER A (Bakery)              CUSTOMER B (Law Firm)          CUSTOMER C (Gym)
  ┌─────────────────────┐          ┌─────────────────────┐       ┌─────────────────────┐
  │  Mac Mini (bakery)  │          │  AWS Lightsail      │       │  Office PC          │
  │                     │          │                     │       │                     │
  │  ┌───────────────┐  │          │  ┌───────────────┐  │       │  ┌───────────────┐  │
  │  │ Docker        │  │          │  │ Docker        │  │       │  │ Docker        │  │
  │  │ container     │  │          │  │ container     │  │       │  │ container     │  │
  │  │               │  │          │  │               │  │       │  │               │  │
  │  │ SLM + FAISS   │  │          │  │ SLM + FAISS   │  │       │  │ SLM + FAISS   │  │
  │  │ + bakery data │  │          │  │ + legal data  │  │       │  │ + gym data    │  │
  │  └───────────────┘  │          │  └───────────────┘  │       │  └───────────────┘  │
  │                     │          │                     │       │                     │
  │  Only knows about   │          │  Only knows about   │       │  Only knows about   │
  │  cakes & pastries   │          │  contracts & law    │       │  classes & members  │
  └─────────────────────┘          └─────────────────────┘       └─────────────────────┘
```

**Each container is completely independent.** No shared database, no shared model server,
no cross-customer data leakage. The SLM weights are the same (copied into each image),
but the FAISS index and config are unique to that business.

### 1. Domain Gate

A lightweight, fast check that runs before anything expensive (retrieval, inference).

- At setup time, we precompute a **centroid vector** from the business's document embeddings.
- For each incoming query, compute cosine similarity against the centroid.
- Reject if similarity < threshold (auto-tuned based on this business's data distribution).
- Fully automatic — the business owner never configures it.

### 2. RAG Pipeline

The core of the system. Every answer is grounded in the business's own documents.

- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, fast, runs on CPU).
- **Vector store**: FAISS — local, no external service needed, runs inside the container.
- **Chunk strategy**: Split documents into 256–512 token chunks with ~50 token overlap. Attach metadata (source ID, timestamp, document name).
- **Retrieval**: Top-K (K=5–10) chunks by cosine similarity. Apply a minimum relevance threshold — if the best chunk scores below it, refuse to answer.
- **Context assembly**: Concatenate retrieved chunks into a prompt context window, respecting the model's max token limit.

### 3. SLM (Inference)

The generative model that lives inside the container and produces answers.

**Recommended base models — choose based on customer's hardware:**

| Model | Parameters | RAM (quantized) | Hardware needed |
|-------|-----------|-----------------|-----------------|
| Phi-3-mini | 3.8B | ~3 GB (4-bit) | Mac Mini 8GB, $5 Lightsail — cheapest option |
| Mistral-7B | 7B | ~5 GB (4-bit) | Mac Mini 16GB, $10 Lightsail — good balance |
| Llama-3-8B | 8B | ~6 GB (4-bit) | Mac Mini 16GB+, $20 Lightsail — best quality |

The model is **domain-agnostic** — it doesn't need to know about bakeries or law firms.
RAG provides the domain knowledge at query time. The model just reads the retrieved
context and produces a grounded answer.

**Optional LoRA fine-tuning (future):**
- After the business accumulates Q&A history, fine-tune a LoRA adapter to improve tone/terminology.
- Adapter is ~10–50 MB. Gets baked into the container on the next image build.
- Not the data restriction mechanism — RAG does that.

### 4. Agent Layer

Orchestrates multi-step reasoning over the SLM + retrieval system.

```
                    ┌─────────────┐
                    │   PLANNER   │
                    │             │
                    │  Breaks the │
                    │  query into │
                    │  sub-tasks  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Retrieve │ │ Retrieve │ │ Compute  │
        │ context  │ │ related  │ │ derived  │
        │ for Q1   │ │ data Q2  │ │ answer   │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │            │
             └─────────────┼────────────┘
                           ▼
                    ┌─────────────┐
                    │  EXECUTOR   │
                    │             │
                    │  Calls SLM  │
                    │  with merged│
                    │  context    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  VALIDATOR  │
                    │             │
                    │  Checks     │
                    │  citations, │
                    │  coherence  │
                    └─────────────┘
```

- **Planner**: Decomposes complex queries into sub-tasks (can be rule-based or LLM-driven).
- **Executor**: Runs each sub-task against the local FAISS index + SLM.
- **Validator**: Ensures the composed answer is grounded and consistent.
- All agent logic is deterministic and fully logged. The business owner can see what the agent did in their admin panel.

### 5. Docker Container — The Deliverable

Every customer gets exactly one container. It runs on their machine. Self-contained.

```
┌──────────────────────────────────────────────────────────────┐
│          CUSTOMER'S DOCKER CONTAINER                          │
│          (runs on their Mac Mini / Lightsail / VPS)          │
│                                                              │
│   ┌────────────────────────────────────┐                     │
│   │         FastAPI Server             │                     │
│   │                                    │                     │
│   │  POST /query    → RAG + SLM       │ ← customer's        │
│   │  POST /ingest   → add new data    │   website/app       │
│   │  GET  /health   → liveness check  │   calls this        │
│   │  GET  /metrics  → Prometheus       │                     │
│   └────────────────────────────────────┘                     │
│                                                              │
│   ┌──────────────┐  ┌───────────────────┐                    │
│   │ SLM weights  │  │ Embedding model   │                    │
│   │ (quantized)  │  │ (all-MiniLM)      │                    │
│   └──────────────┘  └───────────────────┘                    │
│                                                              │
│   ┌──────────────────────────────────────┐                   │
│   │ FAISS index (this business's data)   │                   │
│   └──────────────────────────────────────┘                   │
│                                                              │
│   ┌──────────────────────────────────────┐                   │
│   │ Domain gate centroid + config        │                   │
│   └──────────────────────────────────────┘                   │
│                                                              │
│   ┌──────────────────────────────────────┐                   │
│   │ Admin panel (add data, view logs)    │                   │
│   └──────────────────────────────────────┘                   │
│                                                              │
│   • Distroless / slim base image                             │
│   • Non-root user                                            │
│   • No network egress needed                                 │
│   • Everything runs locally — no cloud calls                 │
│   • Business owns the machine and the data on it             │
└──────────────────────────────────────────────────────────────┘
```

**Target hardware:**

| Machine | Cost | Model it can run | Good for |
|---------|------|-----------------|----------|
| Mac Mini M2 (8 GB) | ~$600 one-time | Phi-3-mini (3.8B, 4-bit) | Small business, low traffic |
| Mac Mini M2 (16 GB) | ~$800 one-time | Mistral-7B (4-bit) | Medium business |
| AWS Lightsail (2 GB) | $5/mo | Phi-3-mini (CPU, slower) | Budget cloud option |
| AWS Lightsail (8 GB) | $40/mo | Mistral-7B (CPU) | Cloud, moderate traffic |
| Any Linux box + GPU | varies | Llama-3-8B (fast) | Best performance |

---

## Request Flow (end to end)

### Example: Local Bakery ("Sweet Rise Bakery")

```
Customer: "Do you have vegan options? Are there any nut-free cakes?"

  1. API Gateway     → Authenticate (widget API key)
  2. Domain Gate     → Embed query, similarity=0.79 (threshold=0.6) → PASS
  3. RAG Retrieval   → Top 5 chunks from local FAISS index:
                        - [DOC-12] Vegan menu section
                        - [DOC-08] Allergen information sheet
                        - [DOC-14] Custom cake ordering policies
                        - [DOC-03] Daily specials format
                        - [DOC-11] Ingredient sourcing FAQ
  4. Prompt Build    → System prompt + 5 chunks + customer query
  5. SLM Inference   → "Yes! Sweet Rise has 6 vegan options including our
                        Chocolate Avocado Cake and Coconut Berry Tart. For
                        nut-free cakes, our Vanilla Cloud and Lemon Drizzle
                        are both made in a nut-free prep area. See our full
                        allergen guide for details. [DOC-12, DOC-08]"
  6. Validator       → Citations DOC-12, DOC-08 exist ✓, answer in context ✓
  7. Return          → JSON response with answer + sources + confidence
```

### Example: Same bakery, off-topic query

```
Customer: "What's the capital of France?"

  1. API Gateway     → Authenticate
  2. Domain Gate     → Embed query, similarity=0.08 (threshold=0.6) → REJECT
  3. Return          → "I can only help with questions about Sweet Rise
                        Bakery — our menu, ordering, allergens, and hours.
                        How can I help with that?"
```

### Example: SaaS Company ("DataPipe.io")

```
Customer: "How do I set up a webhook for the Pro plan?"

  1. API Gateway     → Authenticate (widget API key)
  2. Domain Gate     → Embed query, similarity=0.85 → PASS
  3. RAG Retrieval   → Top chunks from local FAISS index:
                        - [DOC-201] Webhook integration guide
                        - [DOC-187] Pro plan feature matrix
                        - [DOC-203] API authentication docs
  4. Prompt Build    → System prompt + chunks + query
  5. SLM Inference   → "To set up a webhook on the Pro plan: 1) Go to
                        Settings → Integrations → Webhooks, 2) Click
                        'Add Endpoint', 3) Enter your URL and select
                        events to subscribe to. Pro plan supports up to
                        50 webhook endpoints. [DOC-201, DOC-187]"
  6. Validator       → ✓
  7. Return          → JSON response
```

Each business runs on its own machine. There is no shared infrastructure.
The bakery's Mac Mini has never heard of DataPipe.io, and vice versa.

---

## Technology Stack

**Inside the customer's container (runs on their machine):**

| Component | Tool | Why this one |
|-----------|------|-------------|
| **Embedding model** | `all-MiniLM-L6-v2` | Fast, runs on CPU, 80 MB |
| **Vector database** | FAISS | Local, no server needed, runs in-process |
| **SLM** | Phi-3-mini / Mistral-7B / Llama-3-8B | Quantized to fit on small hardware |
| **Quantization** | GGUF (llama.cpp) or bitsandbytes | CPU-friendly inference without a GPU |
| **Inference server** | FastAPI + Uvicorn | Lightweight REST API |
| **Admin panel** | Simple web UI (FastAPI + Jinja or lightweight React) | Upload data, view query logs |
| **Data pipeline** | Built-in cron or watchdog on a folder | Re-embed when new files are dropped in |

**Our build tooling (we use to produce the container):**

| Component | Tool | Purpose |
|-----------|------|---------|
| **Container build** | Docker (distroless base) | Produce the customer's image |
| **CI/CD** | GitHub Actions | Build, test, tag images per customer |
| **Model hosting** | HuggingFace Hub / S3 | Store base model weights for image builds |
| **Fine-tuning** | HuggingFace Transformers + PEFT (LoRA) | Optional per-customer adaptation |
| **Monitoring (optional)** | Prometheus + Grafana | If the customer wants a dashboard |

---

## Repository Layout

```
babyyoday/
│
├── builder/                     # OUR tooling to build a customer's container
│   ├── build_customer.py        # Takes customer data dir → produces Docker image
│   ├── Dockerfile               # Base container definition
│   ├── embed_data.py            # Chunk + embed customer's documents
│   ├── build_gate.py            # Compute domain centroid from embeddings
│   └── config_template.yaml     # Template for per-customer config
│
├── inference/                   # What runs INSIDE the customer's container
│   ├── server.py                # FastAPI application
│   ├── prompt.py                # Prompt template construction
│   ├── validator.py             # Response validation (citation checks)
│   ├── domain_gate.py           # Out-of-domain query rejection
│   ├── retriever.py             # Query → FAISS search → top-K chunks
│   ├── context_builder.py       # Assemble prompt context from chunks
│   └── requirements.txt         # Python dependencies
│
├── agent/                       # Agent orchestration (inside container)
│   ├── planner.py               # Task decomposition
│   ├── executor.py              # Sub-task execution
│   └── router.py                # Query routing logic
│
├── admin/                       # Simple admin panel (inside container)
│   ├── app.py                   # Upload new docs, view query logs
│   └── templates/               # Jinja2 templates
│
├── data_pipeline/               # Runs inside container on a schedule
│   ├── watcher.py               # Watch a folder for new files
│   ├── chunker.py               # Document chunking logic
│   └── reindex.py               # Re-embed and rebuild FAISS index
│
├── model_training/              # Optional — our tooling for LoRA fine-tuning
│   ├── train_lora.py            # Training script
│   ├── eval.py                  # Holdout evaluation
│   └── configs/                 # Hyperparameter configs
│
├── .github/workflows/           # CI/CD
│   └── build.yml                # Build + test pipeline
│
├── tests/
│   ├── test_domain_gate.py
│   ├── test_retrieval.py
│   └── test_inference.py
│
├── docker-compose.yml           # Local dev: spin up a sample customer
├── Readme                       # Original brainstorm
└── README-2.md                  # This document
```

---

## Implementation Roadmap

### Phase 1 — Build the Engine (Weeks 1–4)

| Week | Deliverable |
|------|------------|
| 1 | Document chunker + embedding pipeline. Feed a sample business's docs into FAISS. |
| 2 | RAG pipeline: query embedding → FAISS search → context assembly. |
| 3 | Domain gate: embedding centroid, tune threshold on sample data. |
| 4 | Load quantized SLM (Phi-3-mini or Mistral-7B). Wire up prompt construction. |

**Milestone**: A working RAG + SLM pipeline on your laptop with sample data.

### Phase 2 — Containerize (Weeks 5–8)

| Week | Deliverable |
|------|------------|
| 5 | FastAPI server with `/query` and `/ingest` endpoints. End-to-end flow works. |
| 6 | Response validator (citation checking, grounding). Domain gate integrated. |
| 7 | Dockerfile: bake SLM weights + embedding model + FAISS index into a container. |
| 8 | `build_customer.py` script: takes a folder of docs → produces a ready Docker image. |

**Milestone**: Run `python builder/build_customer.py --data ./sample_bakery/` and get a Docker image that answers bakery questions.

### Phase 3 — Agent & Admin (Weeks 9–12)

| Week | Deliverable |
|------|------------|
| 9 | Agent planner + executor for multi-step queries. |
| 10 | Admin panel: simple web UI to upload new docs and view query logs. |
| 11 | In-container data watcher: drop a file in a folder → auto re-embed + reindex. |
| 12 | CI/CD: automated image builds, test suite, deployment script for Lightsail/Mac. |

**Milestone**: A complete package you can hand to a business owner. They plug it in and it works.

### Phase 4 — Polish & Scale (Weeks 13–16)

| Week | Deliverable |
|------|------------|
| 13 | Deploy to first real customer. Collect feedback. |
| 14 | Chat widget (embeddable JS snippet for their website). |
| 15 | Optional LoRA fine-tuning pipeline for customers who want better answers. |
| 16 | Monitoring, security hardening, documentation for customer handoff. |

**Milestone**: First paying customer running their own agent.

---

## Key Decisions to Make Before Starting

1. **Base model size?** Phi-3-mini (3.8B, runs on $5 Lightsail) vs Mistral-7B (better quality, needs 8+ GB RAM). Start with Phi-3-mini for the widest hardware compatibility.
2. **CPU or GPU inference?** Most small businesses won't have a GPU. Use GGUF quantization + llama.cpp for CPU inference. GPU is a nice-to-have for higher traffic.
3. **How do customers update data?** Start simple: drop files in a watched folder or use the admin panel. Add API connectors (Google Drive, Notion) later.
4. **Pricing model?** One-time setup fee + monthly support? Or recurring subscription that includes hosting on Lightsail?
5. **Fine-tune or RAG-only first?** Start RAG-only. It works out of the box with zero training. Add LoRA later for customers who need it.
