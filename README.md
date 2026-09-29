# Document Q&A System (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions from a set of PDF 
documents using local, open-source models — no external API calls or costs.

## Problem
LLMs answer purely from what they memorized during training, which can be outdated, generic, 
or simply not about your specific documents. RAG solves this by retrieving relevant content 
from your own documents first, then having the LLM answer grounded in that retrieved context.

## Architecture
PDF documents → chunked (500 chars, 50 overlap) → embedded (MiniLM)
→ stored in ChromaDB → user question → embedded → top-3 similar chunks retrieved
→ chunks + question sent to local LLM (Llama 3.1 8B via Ollama) → grounded answer

## Tech Stack
- **LLM**: Llama 3.1 8B, served locally via [Ollama](https://ollama.com) (GPU-accelerated, 
  NVIDIA RTX 5070) — no API costs, fully local
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim vectors)
- **Vector store**: ChromaDB (local, persisted to disk)
- **Orchestration**: LangChain (document loading, chunking, retrieval chain)
- **Experiment tracking**: MLflow

## Dataset
3 PDFs spanning very different content types, chosen deliberately to test whether retrieval 
correctly distinguishes between sources rather than blending them:
- A technical research paper (AI Behaviour Trees for NPC decision-making)
- A structured payslip (tabular financial data)
- A novel excerpt (narrative prose)

## Results

**Retrieval accuracy: 3/3** on a test set of questions targeting each document — the system 
correctly identified the right source document as the top result for every query.

**Example — precise extraction from structured data:**
> Q: "What is mentioned in the payslip?"
> A: Correctly extracted Basic Salary, HRA, Gross Earnings, Total Deductions, and Net Salary 
> as specific figures — not a vague summary.

**Example — accurate summarization of technical content:**
> Q: "What is the main contribution of the AI behaviour trees paper?"
> A: Correctly identified the paper's core contribution (Behaviour Trees as a modular, 
> hierarchical alternative for NPC decision-making), sourced from the correct pages.

Every answer included source citations (file + page number), confirmed accurate across all 
test cases.

## Evaluation Approach
Retrieval was evaluated by checking whether the top-ranked retrieved chunk matched the expected 
source document for each test question — a simplified version of what frameworks like RAGAS 
measure as "context precision." A more complete evaluation would also score answer faithfulness 
(whether the generated answer is fully supported by the retrieved context, not just topically 
related) — noted as a next step below.

## Tech Notes
- Running entirely locally (Ollama + local embeddings) means zero API costs, but response time 
  depends on local hardware — this ran on an NVIDIA RTX 5070 GPU
- Chunk size (500 chars) and retrieval count (k=3) were set as reasonable defaults; not yet 
  systematically tuned

## Next Steps
- Add faithfulness/answer-quality evaluation (RAGAS or a custom LLM-as-judge approach)
- Experiment with chunk size and k as tunable hyperparameters
- Add hybrid search (keyword + semantic) for queries with exact terms/numbers
- Build a simple Streamlit or Gradio UI for interactive querying
- Try swapping in an Azure OpenAI embedding/LLM for comparison against the local setup

## Project Structure
\`\`\`
document-qa/
├── data/                       # Source PDFs (not committed — large files)
├── chroma_db/                  # Persisted vector store (not committed)
├── notebooks/
│   └── 01_rag_pipeline.ipynb   # Full pipeline: load → chunk → embed → retrieve → generate → evaluate
├── requirements.txt
└── README.md
\`\`\`

## Setup
\`\`\`bash
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt

# Install Ollama separately: https://ollama.com/download
ollama pull llama3.1:8b
\`\`\`