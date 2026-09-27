# RAG

CURRENT
  │
  ▼
Phase 1: Basic RAG  ← YOU ARE HERE
  ├── Retrieval inspection
  ├── Top-K / similarity
  ├── Source citations
  ├── PDF ingestion
  └── Reusable RAG pipeline
        │
        ▼
Phase 2: Better Retrieval
  ├── Better chunking
  ├── Chunk overlap
  ├── Cosine similarity
  ├── Metadata filtering
  └── Hybrid search
        │
        ▼
Phase 3: Retrieval Evaluation
  ├── Precision
  ├── Recall
  ├── MRR
  └── Retrieval evaluation dataset
        │
        ▼
Phase 4: Reranking
  ├── Cross-encoder
  └── Retrieve → Rerank → Context
        │
        ▼
Phase 5: Query Enhancement
  ├── Query rewriting
  ├── Query expansion
  ├── Multi-query
  └── HyDE
        │
        ▼
Phase 6: Advanced RAG
  ├── Parent-child retrieval
  ├── Multi-vector retrieval
  ├── Metadata-aware RAG
  └── Graph RAG
        │
        ▼
Phase 7: Context Engineering
  ├── Context compression
  ├── Context ordering
  ├── Deduplication
  └── Lost-in-the-middle handling
        │
        ▼
Phase 8: RAG Evaluation
  ├── Faithfulness
  ├── Answer relevance
  ├── Context relevance
  └── RAGAS / custom evaluation
        │
        ▼
Phase 9: Production RAG
  ├── FastAPI
  ├── Database/vector DB
  ├── Authentication
  ├── Caching
  ├── Logging
  └── Monitoring
        │
        ▼
Phase 10: Advanced / Agentic RAG
  ├── Agents
  ├── Tool calling
  ├── Self-RAG
  ├── Corrective RAG
  └── Multi-agent RAG
# RAG — Complete Learning & Implementation Process

## 0. Prerequisites

Before starting RAG, understand:

* Python
* NumPy / Pandas
* Basic NLP
* Machine Learning basics
* Neural Networks basics
* Transformers
* APIs
* JSON
* Git / GitHub
* Basic SQL

---

# 1. Understand RAG

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of asking an LLM to answer only from its pretrained knowledge:

```text
User Query
    ↓
Retrieve relevant information
    ↓
Give information + query to LLM
    ↓
Generate answer
```

### Basic RAG architecture

```text
                ┌──────────────┐
                │ User Query   │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │  Retriever   │
                └──────┬───────┘
                       ↓
             Relevant Documents
                       ↓
                ┌──────────────┐
                │     LLM      │
                └──────┬───────┘
                       ↓
                    Answer
```

---

# 2. Basic RAG Pipeline

Learn and implement this first.

```text
Documents
    ↓
Document Loading
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
Vector Database
    ↓
Similarity Search
    ↓
Retrieved Context
    ↓
Prompt + Context
    ↓
LLM
    ↓
Answer
```

## Step 1 — Load Documents

Learn to load:

* TXT
* PDF
* DOCX
* CSV
* HTML
* Markdown
* Web pages

Example:

```text
data/
├── document1.pdf
├── document2.pdf
├── document3.txt
└── documentation.md
```

---

# 3. Document Preprocessing

Before creating embeddings, clean the documents.

Learn:

* Remove unnecessary whitespace
* Remove duplicate text
* Remove headers/footers
* Handle special characters
* Preserve document structure
* Extract metadata

Example metadata:

```json
{
  "source": "machine_learning.pdf",
  "page": 15,
  "section": "Classification"
}
```

---

# 4. Text Chunking

Large documents cannot always be passed directly to an LLM.

Split them into smaller pieces.

```text
Document
    ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

## Learn these chunking strategies

### 4.1 Fixed-size chunking

```text
Chunk size = 500 tokens
Overlap = 50 tokens
```

### 4.2 Sentence-based chunking

Split according to sentences.

### 4.3 Paragraph-based chunking

Split according to paragraphs.

### 4.4 Recursive chunking

Split using a hierarchy:

```text
Document
   ↓
Paragraph
   ↓
Sentence
   ↓
Words
```

### 4.5 Semantic chunking

Split based on semantic meaning instead of only length.

---

# 5. Embeddings

Understand:

> What is an embedding?

An embedding converts text into a numerical vector.

```text
"Machine learning is a subset of AI"
                ↓
        Embedding Model
                ↓
[0.21, -0.53, 0.87, ...]
```

Learn:

* Embedding models
* Vector dimensions
* Cosine similarity
* Euclidean distance
* Dot product
* Semantic similarity

---

# 6. Vector Database

Store embeddings in a vector database.

Learn at least one:

* FAISS
* Chroma
* Qdrant
* Weaviate
* Milvus
* Pinecone

Start with:

```text
FAISS / Chroma
```

Then learn:

```text
Qdrant
```

for more production-oriented work.

---

# 7. Similarity Search

Given a query:

```text
"What is supervised learning?"
```

Convert it into an embedding:

```text
Query
 ↓
Embedding
 ↓
Vector
```

Compare it with document vectors.

```text
Query Vector
     ↓
Similarity Search
     ↓
Top-K Documents
```

Learn:

* Top-K retrieval
* Cosine similarity
* Distance thresholds
* Metadata filtering

---

# 8. Basic Retriever

Implement:

```text
query
 ↓
query embedding
 ↓
vector search
 ↓
top-k chunks
```

Example:

```text
Query
  ↓
Embedding
  ↓
Vector DB
  ↓
Top 5 chunks
```

At this point you have a **retrieval system**.

---

# 9. Connect Retriever to LLM

Now combine retrieval and generation.

```text
User Query
    ↓
Retriever
    ↓
Relevant Context
    ↓
Prompt
    ↓
LLM
    ↓
Answer
```

Prompt structure:

```text
Use the following context to answer the question.

Context:
{retrieved_context}

Question:
{user_question}

Answer:
```

This is your **Basic RAG**.

---

# 10. Add Sources / Citations

Don't only return:

```text
Answer
```

Return:

```text
Answer

Sources:
- document.pdf — page 10
- document.pdf — page 15
- notes.txt
```

Store metadata with every chunk.

```json
{
  "text": "...",
  "source": "document.pdf",
  "page": 10
}
```

---

# 11. RAG v1 Project

Build:

## PDF Question Answering System

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector DB
 ↓
Retriever
 ↓
LLM
 ↓
Answer + Sources
```

Features:

* Upload PDF
* Ask questions
* Retrieve relevant chunks
* Generate answer
* Show sources
* Show page numbers

---

# 12. Improve Chunking

Basic RAG often fails because of poor chunking.

Experiment with:

```text
Chunk Size
├── 200
├── 500
├── 800
└── 1200
```

Experiment with:

```text
Overlap
├── 0
├── 50
├── 100
└── 200
```

Compare retrieval quality.

Learn:

* Small chunks
* Large chunks
* Overlapping chunks
* Semantic chunks
* Parent-child chunks

---

# 13. Retrieval Evaluation

Now stop judging RAG only by looking at answers.

Measure retrieval quality.

Learn:

## Precision@K

How many retrieved documents are relevant?

```text
Precision@K =
Relevant Retrieved Documents
----------------------------
Total Retrieved Documents
```

## Recall@K

How many relevant documents were successfully retrieved?

```text
Recall@K =
Relevant Retrieved Documents
----------------------------
Total Relevant Documents
```

## MRR

Mean Reciprocal Rank.

Measures how early the first relevant result appears.

## Hit Rate

Whether at least one relevant document appears in Top-K.

---

# 14. Build a RAG Evaluation Dataset

Create:

```text
questions.json
```

Example:

```json
[
  {
    "question": "What is supervised learning?",
    "expected_source": "ml.pdf"
  },
  {
    "question": "What is gradient descent?",
    "expected_source": "optimization.pdf"
  }
]
```

Run your retriever against these questions.

Measure:

```text
Recall@5
Precision@5
MRR
Hit Rate
```

---

# 15. Improve Retrieval

Basic vector search is not always enough.

Learn:

## Metadata Filtering

Example:

```text
query
 ↓
filter:
document = "machine_learning.pdf"
 ↓
vector search
```

Other filters:

* page
* author
* date
* category
* document type

---

# 16. Hybrid Search

Combine:

```text
Keyword Search
+
Vector Search
```

Architecture:

```text
                 Query
                   ↓
          ┌────────┴────────┐
          ↓                 ↓
     Vector Search      BM25 Search
          ↓                 ↓
          └────────┬────────┘
                   ↓
             Combined Results
                   ↓
                Reranker
```

Learn:

* BM25
* Dense retrieval
* Sparse retrieval
* Hybrid retrieval
* Reciprocal Rank Fusion (RRF)

---

# 17. Reranking

Initial retrieval might return:

```text
Top 20 chunks
```

Then use a reranker:

```text
Top 20
   ↓
Reranker
   ↓
Top 5
```

Learn:

* Cross-encoder
* Bi-encoder
* Reranker models
* Relevance scoring

Architecture:

```text
Query
 ↓
Retriever
 ↓
Top 20 chunks
 ↓
Reranker
 ↓
Top 5 chunks
 ↓
LLM
```

---

# 18. Query Enhancement

Sometimes the user's question is poorly written.

Example:

```text
User:
"how does it work?"
```

The retriever doesn't know what "it" means.

Use:

## Query Rewriting

```text
Original Query
      ↓
Query Rewriter
      ↓
Improved Query
      ↓
Retriever
```

---

# 19. Multi-Query Retrieval

Generate multiple versions of the query.

```text
Original Query
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
Q1   Q2   Q3
 ↓    ↓    ↓
Retrieval
 └────┼────┘
      ↓
 Combined Results
```

Useful when different wording can refer to the same information.

---

# 20. HyDE

Learn:

> Hypothetical Document Embeddings

Process:

```text
User Query
    ↓
LLM generates hypothetical answer
    ↓
Embedding
    ↓
Vector Search
    ↓
Relevant Documents
```

Instead of embedding only the question, use a hypothetical answer to improve retrieval.

---

# 21. Context Compression

Retrieval may return too much information.

Example:

```text
Retrieved:
20 chunks
5000 tokens
```

Compress:

```text
20 chunks
    ↓
Relevant information
    ↓
1000 tokens
    ↓
LLM
```

Learn:

* Context filtering
* Extractive compression
* LLM-based compression
* Redundancy removal

---

# 22. Context Engineering

Now optimize what is sent to the LLM.

Learn:

* Context ordering
* Deduplication
* Relevant chunk selection
* Context compression
* Token budgeting
* Lost-in-the-middle problem

Architecture:

```text
Retrieved Chunks
      ↓
Remove duplicates
      ↓
Filter irrelevant chunks
      ↓
Compress
      ↓
Order chunks
      ↓
Final Context
      ↓
LLM
```

---

# 23. Improve Generation

Now improve the answer-generation stage.

Learn:

* Prompt templates
* System prompts
* Structured output
* JSON output
* Citation generation
* Answer constraints
* Hallucination reduction

Example:

```text
If the answer is not present in the context,
say:

"I don't have enough information in the provided documents."
```

---

# 24. RAG Evaluation

Evaluate the entire pipeline.

You need to measure:

```text
                 RAG Evaluation
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
     Retrieval      Context       Answer
     Quality        Quality       Quality
```

Important metrics/concepts:

### Retrieval

* Precision@K
* Recall@K
* MRR
* Hit Rate

### Context

* Context relevance
* Context precision
* Context recall

### Answer

* Faithfulness
* Answer relevance
* Answer correctness

Learn tools/frameworks such as:

* RAGAS
* DeepEval
* LangSmith

---

# 25. Hallucination Detection

Test:

```text
Question
 ↓
Retrieved Context
 ↓
Generated Answer
```

Ask:

> Is every important claim supported by the retrieved context?

Example:

```text
Context:
"Python was created by Guido van Rossum."

Answer:
"Python was created by Guido van Rossum in 1989."
```

The year may not be supported by the retrieved context.

Your system should detect this.

---

# 26. RAG Failure Analysis

Learn why RAG fails.

## Failure Type 1 — Bad Chunking

```text
Important information split across chunks
```

## Failure Type 2 — Bad Retrieval

```text
Relevant document not retrieved
```

## Failure Type 3 — Bad Reranking

```text
Relevant chunk ranked too low
```

## Failure Type 4 — Too Much Context

```text
LLM gets unnecessary information
```

## Failure Type 5 — Hallucination

```text
LLM generates information not present in context
```

## Failure Type 6 — Ambiguous Query

```text
Question is unclear
```

---

# 27. Advanced Retrieval

After basic + hybrid retrieval, learn:

### Parent-Child Retrieval

```text
Large Parent Document
        ↓
Small Child Chunks
        ↓
Retrieve Child
        ↓
Return Parent Context
```

### Multi-Vector Retrieval

Represent one document using multiple vectors.

### Hierarchical Retrieval

```text
Document
 ↓
Section
 ↓
Paragraph
 ↓
Sentence
```

Retrieve progressively.

---

# 28. Document Structure-Aware RAG

Don't treat every document as plain text.

Understand:

* Titles
* Headings
* Tables
* Lists
* Code
* Images
* Footnotes
* Page numbers

Example:

```text
PDF
 ├── Title
 ├── Section
 │    ├── Paragraph
 │    └── Table
 ├── Section
 │    └── Code
 └── References
```

This is important for real-world document RAG.

---

# 29. Multimodal RAG

Move beyond text.

Learn retrieval from:

```text
Text
Images
Tables
Charts
PDF pages
```

Architecture:

```text
             Documents
                 ↓
       ┌─────────┼─────────┐
       ↓         ↓         ↓
      Text      Image     Table
       ↓         ↓         ↓
   Embeddings  Embeddings  Processing
       └─────────┼─────────┘
                 ↓
             Retrieval
                 ↓
           Multimodal LLM
```

---

# 30. Agentic RAG

Only after understanding normal RAG.

Instead of:

```text
Query
 ↓
Retriever
 ↓
LLM
```

Use:

```text
                    User Query
                         ↓
                       Agent
                         ↓
              ┌──────────┼──────────┐
              ↓          ↓          ↓
           Search      SQL       Documents
              ↓          ↓          ↓
              └──────────┼──────────┘
                         ↓
                       Agent
                         ↓
                       LLM
                         ↓
                      Answer
```

Learn:

* Tool calling
* Query routing
* Planning
* Multi-step retrieval
* Self-correction
* Agent memory
* Tool selection

---

# 31. Query Routing

Different questions need different sources.

```text
                 Query
                   ↓
                Router
             ┌─────┼─────┐
             ↓     ↓     ↓
           PDF    SQL    Web
             ↓     ↓     ↓
             └─────┼─────┘
                   ↓
                 LLM
```

Example:

```text
"What is in my PDF?"
        ↓
PDF Retriever

"What were sales last month?"
        ↓
SQL Database

"What happened today?"
        ↓
Web Search
```

---

# 32. RAG + SQL

Learn how RAG can work with structured data.

```text
User Question
      ↓
Query Router
      ↓
SQL Generator
      ↓
Database
      ↓
Result
      ↓
LLM
```

Example:

```text
"How many customers purchased product X?"
```

The system generates SQL rather than searching a vector database.

---

# 33. RAG + Web Search

Build:

```text
User Query
    ↓
Query Router
    ↓
Is internal information needed?
    ├── Yes → Vector DB
    └── No  → Web Search
```

You can also combine:

```text
Internal Documents
       +
Web Search
       ↓
Reranking
       ↓
LLM
```

---

# 34. Production RAG

Now move from notebook to application.

Learn:

## Backend

* FastAPI
* REST APIs
* Async programming

## Frontend

* Streamlit
* React

## Database

* PostgreSQL
* Vector database

## Infrastructure

* Docker
* Linux
* Cloud deployment

---

# 35. Production Architecture

```text
                 User
                  ↓
              Frontend
                  ↓
               FastAPI
                  ↓
             Query Router
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
    Retriever            Database
        ↓
    Reranker
        ↓
 Context Builder
        ↓
       LLM
        ↓
     Response
        ↓
      User
```

---

# 36. RAG Performance Optimization

Measure:

```text
Latency
Cost
Accuracy
Throughput
```

Optimize:

* Embedding caching
* LLM caching
* Retrieval caching
* Batch embeddings
* Smaller models
* Smaller context
* Top-K optimization
* Async requests

---

# 37. RAG Security

Learn:

* Prompt injection
* Data leakage
* Unauthorized document access
* PII protection
* Access control
* Tenant isolation
* Malicious documents

Example:

```text
User
 ↓
Authentication
 ↓
Authorization
 ↓
Retriever
 ↓
Only permitted documents
 ↓
LLM
```

---

# 38. Monitoring

Production RAG should record:

```text
Query
 ↓
Retrieved documents
 ↓
Similarity scores
 ↓
Reranker scores
 ↓
Prompt
 ↓
LLM response
 ↓
Latency
 ↓
Token usage
 ↓
User feedback
```

Monitor:

* Retrieval failures
* Hallucinations
* Latency
* Cost
* Error rate
* User feedback

---

# 39. Complete RAG Architecture

The final system can look like:

```text
                         USER
                           │
                           ↓
                     User Query
                           │
                           ↓
                    Query Analyzer
                           │
                           ↓
                    Query Rewriter
                           │
                           ↓
                     Query Router
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
          Vector DB       BM25         SQL
              ↓            ↓            ↓
              └────────────┼────────────┘
                           ↓
                    Hybrid Retrieval
                           ↓
                        Top-K
                           ↓
                       Reranker
                           ↓
                  Context Compression
                           ↓
                   Context Builder
                           ↓
                         LLM
                           ↓
                 Answer + Citations
                           ↓
                    Evaluation
                           ↓
                     Monitoring
```

---

# 40. Projects — Progressive Difficulty

## Project 1 — Beginner

### PDF Chatbot

```text
PDF → Chunks → Embeddings → Vector DB → LLM
```

Features:

* Upload PDF
* Ask questions
* Sources

---

## Project 2 — Intermediate

### Multiple PDF RAG

```text
Multiple PDFs
      ↓
Metadata
      ↓
Vector DB
      ↓
Filtering
      ↓
Retrieval
      ↓
LLM
```

Features:

* Multiple documents
* Document filtering
* Page citations
* Conversation history

---

## Project 3 — Advanced

### Hybrid RAG

```text
Query
 ↓
 ┌───────────────┐
 ↓               ↓
Vector Search   BM25
 ↓               ↓
 └───────┬───────┘
         ↓
       RRF
         ↓
     Reranker
         ↓
        LLM
```

---

## Project 4 — Advanced

### Research Paper RAG

Features:

* PDF ingestion
* Semantic chunking
* Metadata
* Hybrid retrieval
* Reranking
* Citations
* Evaluation
* Query rewriting

---

## Project 5 — Advanced+

### Company Knowledge Assistant

```text
              Company Assistant
                     ↓
              Query Router
           ┌─────────┼─────────┐
           ↓         ↓         ↓
         Docs       SQL       Web
           ↓         ↓         ↓
           └─────────┼─────────┘
                     ↓
                  Reranker
                     ↓
                     LLM
                     ↓
               Answer + Sources
```

---

# 41. Recommended Learning Order

Follow this exact order:

```text
1. RAG Fundamentals
        ↓
2. Document Loading
        ↓
3. Text Cleaning
        ↓
4. Chunking
        ↓
5. Embeddings
        ↓
6. Vector Databases
        ↓
7. Similarity Search
        ↓
8. Basic Retriever
        ↓
9. LLM Integration
        ↓
10. Sources / Citations
        ↓
11. Basic RAG Project
        ↓
12. Chunking Optimization
        ↓
13. Retrieval Evaluation
        ↓
14. Metadata Filtering
        ↓
15. BM25
        ↓
16. Hybrid Search
        ↓
17. Reranking
        ↓
18. Query Rewriting
        ↓
19. Multi-Query
        ↓
20. HyDE
        ↓
21. Context Compression
        ↓
22. Context Engineering
        ↓
23. RAG Evaluation
        ↓
24. Hallucination Detection
        ↓
25. Advanced Retrieval
        ↓
26. Multimodal RAG
        ↓
27. SQL RAG
        ↓
28. Query Routing
        ↓
29. Agentic RAG
        ↓
30. Production RAG
        ↓
31. RAG Security
        ↓
32. Monitoring
        ↓
33. Optimization
```

---

# 42. What You Should Know for NLP/AI Interviews

Focus especially on:

### Fundamentals

* What is RAG?
* Why RAG instead of fine-tuning?
* RAG vs fine-tuning
* Embeddings
* Vector databases
* Cosine similarity

### Retrieval

* Chunking
* Top-K
* Metadata filtering
* BM25
* Hybrid search
* Reranking

### Advanced

* Query rewriting
* Multi-query
* HyDE
* Parent-child retrieval
* Context compression

### Evaluation

* Precision@K
* Recall@K
* MRR
* Hit Rate
* Faithfulness
* Context relevance
* Answer relevance

### Production

* Latency
* Cost
* Caching
* Scaling
* Security
* Monitoring

---

# 43. Final RAG Skill Tree

```text
                         RAG
                          │
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
   Retrieval           Generation        Evaluation
       │                  │                  │
       ├─ Chunking        ├─ Prompting      ├─ Precision
       ├─ Embeddings      ├─ LLM            ├─ Recall
       ├─ Vector DB       ├─ Context        ├─ MRR
       ├─ BM25            └─ Citations      ├─ Faithfulness
       ├─ Hybrid Search                     └─ Relevance
       └─ Reranking
                          │
                          ↓
                    Advanced RAG
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
          Query         Multimodal   Agentic
        Enhancement       RAG          RAG
              │           │           │
              ├─ Rewrite  ├─ Images   ├─ Tools
              ├─ Multi    ├─ Tables   ├─ Routing
              └─ HyDE     └─ PDFs     └─ Planning
                          │
                          ↓
                    Production RAG
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
           Security     Scaling     Monitoring
              │           │           │
              ├─ Auth     ├─ Cache    ├─ Tracing
              ├─ PII      ├─ Async    ├─ Metrics
              └─ Injection└─ Docker   └─ Feedback
```

# 44. The Most Important Transition

If you have already implemented **Basic RAG**, your next target should be:

```text
Basic RAG
    ↓
Better Chunking
    ↓
Retrieval Evaluation
    ↓
Hybrid Search
    ↓
Reranking
    ↓
Query Enhancement
    ↓
Context Engineering
    ↓
RAG Evaluation
    ↓
Production RAG
    ↓
Agentic RAG
```

**Do not jump directly from Basic RAG to Agentic RAG.**

The biggest NLP/AI engineering skill is understanding **why retrieval fails and how to systematically improve it**, not simply connecting an LLM to a vector database.
