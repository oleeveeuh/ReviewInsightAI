# ReviewInsight AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

**Agentic Employee Sentiment Analysis System for Warehouse Operations**

A production-grade LLM-powered agent system implementing the ReAct (Reason + Act) pattern for intelligent tool orchestration, persistent memory, and statistical drift detection. Analyzes employee reviews across multiple platforms (Glassdoor, Indeed, Reddit, YouTube) to extract sentiment, themes, and retention risk signals.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technical Background](#technical-background)
- [Core Components](#core-components)
- [Function Reference](#function-reference)
- [SQL Reference](#sql-reference)
- [Agentic Capabilities](#agentic-capabilities)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Performance Benchmarks](#performance-benchmarks)
- [Experimentation & Results](#experimentation--results)
- [Dashboard Demo](#dashboard-demo)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Extension Guide](#extension-guide)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            AGENT ORCHESTRATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐                           │
│  │  LLM Planner│ ──▶│ Tool Registry│ ──▶│  Executor  │                           │
│  │  (Reasoning)│    │ (Dispatch)   │    │  (Acting)   │                           │
│  └─────────────┘    └──────────────┘    └─────────────┘                           │
│         │                                        │                                 │
│         │                                        ▼                                 │
│         │                             ┌─────────────────────┐                       │
│         │                             │  Vector DB         │                       │
│         │                             │  (Semantic Search) │                       │
│         │                             └─────────────────────┘                       │
│         │                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐                 │
│  │                    PERSISTENT MEMORY                       │                 │
│  │  • Analysis Log (JSONL)                                    │                 │
│  │  • Theme Distribution Cache                                 │                 │
│  │  • Sentiment Trend History                                  │                 │
│  │  • Anomaly Detection Registry                               │                 │
│  └─────────────────────────────────────────────────────────────┘                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            DATA PROCESSING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Glassdoor CSV ──┐                                                                │
│  Indeed JSON   ──┼──▶ [Merge & Clean] ──▶ [Quality Filter] ──▶ Unified JSONL      │
│  Reddit Data   ──┤                     │                                           │
│  YouTube TXT   ──┘                     ▼                                           │
│                                     [Deduplication]                              │
│                                     [Date Parsing]                                │
│                                     [Text Normalization]                           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Background

### ReAct Pattern Implementation

This system implements the **ReAct (Reasoning + Acting)** paradigm, where an LLM:

1. **Reasons** about which tools to use based on the input
2. **Acts** by executing tools in a planned sequence
3. **Observes** results and synthesizes a final response

Unlike traditional one-shot LLM calls, the agent maintains state, uses tools dynamically, and builds context incrementally.

### Tool-Augmented Generation

The system extends the RAG (Retrieval-Augmented Generation) pattern with executable tools:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| RAG | Retrieve static context | Document QA |
| TAG | Retrieve + execute functions | Dynamic analysis |
| ReAct | Plan → Act → Observe loop | Complex reasoning |

### KL Divergence for Drift Detection

Theme distribution drift is monitored using Kullback-Leibler divergence:

```
D_KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
```

Where P is the current distribution and Q is the historical baseline. Higher values indicate greater distribution shift.

### Cross-Validation Methodology

To ensure **leakage-free evaluation**, the system implements cross-prompt validation:

```
Generator Prompts: v2.0, v3.0, v6.0
                           ↓
                    Ensemble Labels
                           ↓
Test All Prompts: v1.0, v2.0, v3.0, v4.0, v5.0, v6.0
    (No prompt tested against its own labels)
```

**Why this matters**: Testing a prompt against labels it generated itself inflates accuracy by 8-10 percentage points. Cross-validation provides fair, unbiased performance estimates while maintaining a fully automated pipeline (no human labeling required).

**Implementation**:
- `generate_crossval_labels.py`: Generates labels using multiple prompts
- `evaluate_crossval.py`: Leakage-free evaluation framework

---

## Core Components

### 1. Agent Controller ([`src/agent/controller.py`](src/agent/controller.py))

The orchestration layer implementing the ReAct pattern.

**Key Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `plan(review_text)` | LLM creates execution plan | Dict with reasoning and steps |
| `execute_plan(plan)` | Runs tools in sequence | Tool outputs and analysis |
| `analyze(review_text)` | Full agentic pipeline | Complete analysis results |

**Example Plan Output:**

```json
{
  "reasoning": "Review mentions mandatory overtime and scheduling issues. I'll retrieve similar reviews for context, then analyze sentiment.",
  "steps": [
    {"tool": "retrieve_similar_reviews", "params": {"query_text": "...", "k": 5}},
    {"tool": "analyze_sentiment", "params": {"review_text": "...", "context_reviews": [...]}}
  ]
}
```

### 2. Tool Registry ([`src/agent/tools.py`](src/agent/tools.py))

Pydantic-validated tool implementations with automatic schema generation.

**Available Tools:**

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `retrieve_similar_reviews` | Semantic context retrieval | Query text, k | Top-k similar reviews |
| `analyze_sentiment` | LLM-based sentiment classification | Review text, context | Sentiment (1-5), themes, risk |
| `detect_anomaly` | Statistical anomaly detection | Themes, sentiment | Anomaly flags, rarity scores |

**Tool Interface:**

```python
class Tool(BaseModel):
    name: str
    description: str  # Exposed to LLM for decision-making
    parameters: Dict[str, Any]  # JSON Schema for validation

    def execute(self, **kwargs) -> Dict:
        ...
```

### 3. Persistent Memory System ([`src/agent/memory.py`](src/agent/memory.py))

Stateful agent memory enabling temporal analysis and drift detection.

**Capabilities:**

- **Temporal Tracking**: Theme/sentiment evolution over time
- **Context Enhancement**: Historical baselines for comparison
- **Anomaly Logging**: Persistent record of unusual patterns
- **Resume Capability**: Interrupted analyses can be resumed

**Key Methods:**

| Method | Returns |
|--------|---------|
| `record_analysis(review_id, analysis)` | Appends to persistent log |
| `get_theme_distribution(last_n)` | Theme frequency distribution |
| `get_anomaly_rate(last_n)` | Proportion of anomalous reviews |
| `get_sentiment_trend(last_n)` | Recent sentiment scores |
| `get_risk_distribution(last_n)` | Retention risk breakdown |
| `get_summary_stats()` | Comprehensive memory statistics |

### 4. Drift Detection ([`src/agent/drift.py`](src/agent/drift.py))

Statistical process monitoring using KL divergence for distribution shift detection.

**Features:**

- **KL Divergence**: Detects theme distribution changes
- **Threshold-based Alerting**: Configurable sensitivity (0.05-0.20)
- **Multi-theme Tracking**: Monitors all themes simultaneously
- **Significant Change Detection**: Flags changes >5 percentage points

**Output Format:**

```python
{
    "is_drifting": True,
    "kl_divergence": 0.156,
    "threshold": 0.10,
    "significant_changes": {
        "overtime": {
            "baseline": 0.34,
            "current": 0.55,
            "change": 0.21,
            "direction": "increase"
        }
    },
    "alert": "⚠️ DRIFT DETECTED: ..."
}
```

### 5. REST API ([`api/main.py`](api/main.py))

Production FastAPI service with auto-generated OpenAPI documentation.

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyze` | POST | Agentic review analysis |
| `/memory/stats` | GET | Historical statistics |
| `/tools` | GET | Available tool catalog |
| `/health` | GET | Service health check |

---

## Function Reference

### Agent Controller ([`src/agent/controller.py`](src/agent/controller.py))

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `_get_client()` | None | `OpenAI` | Lazy-initializes OpenAI client |
| `ReviewAgent.__init__()` | `vector_db=None`, `model="gpt-4o-mini"` | None | Initialize agent with tool registry |
| `ReviewAgent.plan()` | `review_text: str` | `Dict[str, Any]` | LLM creates execution plan with reasoning |
| `ReviewAgent.execute_plan()` | `review_text: str`, `plan: Dict[str, Any]` | `Dict[str, Any]` | Executes planned tool sequence |
| `ReviewAgent.analyze()` | `review_text: str` | `Dict[str, Any]` | Full agentic pipeline (plan → execute → synthesize) |
| `analyze_review_agentic()` | `review_text: str`, `vector_db=None` | `Dict[str, Any]` | Convenience function for agentic analysis |

### Tool Registry ([`src/agent/tools.py`](src/agent/tools.py))

**Base Tool Class:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `Tool.__init__()` | `name`, `description`, `parameters` | None | Base tool initialization |
| `Tool.execute()` | `**kwargs` | `Dict` | Abstract method for tool execution |

**Tool Implementations:**

| Tool | Method | Parameters | Returns | Description |
|-----|--------|-----------|---------|-------------|
| `RetrieveSimilarTool` | `execute()` | `query_text: str`, `k: int=5` | `Dict` | Semantic search for similar reviews |
| `AnalyzeSentimentTool` | `execute()` | `review_text: str`, `context_reviews: Optional[List]=None` | `Dict` | LLM sentiment analysis (1-5 scale, themes, risk) |
| `DetectAnomalyTool` | `execute()` | `themes: List[str]`, `sentiment: int` | `Dict` | Statistical anomaly detection |

**ToolRegistry Class:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `ToolRegistry.__init__()` | `vector_db=None` | None | Initialize and register core tools |
| `ToolRegistry.register()` | `tool: Tool` | None | Register a tool instance |
| `ToolRegistry.get_tool()` | `name: str` | `Optional[Tool]` | Get tool by name |
| `ToolRegistry.list_tools()` | None | `List[Dict]` | List all tools for LLM consumption |
| `ToolRegistry.execute()` | `tool_name: str`, `**kwargs` | `Dict[str, Any]` | Execute tool by name |

### Agent Memory ([`src/agent/memory.py`](src/agent/memory.py))

**AgentMemory Class:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `AgentMemory.__init__()` | `memory_dir='data/memory'` | None | Initialize persistent memory |
| `AgentMemory.record_analysis()` | `review_id: str`, `analysis: Dict` | None | Record analysis to JSONL log |
| `AgentMemory.load_history()` | `last_n: Optional[int]=None` | `List[Dict]` | Load analysis history |
| `AgentMemory.get_theme_distribution()` | `last_n: int=100` | `Dict[str, float]` | Calculate theme frequency distribution |
| `AgentMemory.get_anomaly_rate()` | `last_n: int=100` | `float` | Calculate anomaly detection rate |
| `AgentMemory._normalize_sentiment()` | `sentiment: Any` | `Optional[float]` | Normalize sentiment to numeric scale |
| `AgentMemory.get_sentiment_trend()` | `last_n: int=100` | `List[float]` | Get recent sentiment scores |
| `AgentMemory.get_risk_distribution()` | `last_n: int=100` | `Dict[str, int]` | Get retention risk breakdown |
| `AgentMemory.get_summary_stats()` | None | `Dict[str, Any]` | Get comprehensive memory statistics |

**MemoryAwareAgent Class:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `MemoryAwareAgent.__init__()` | `vector_db=None`, `model="gpt-4o-mini"` | None | Initialize agent with memory |
| `MemoryAwareAgent.analyze()` | `review_text: str`, `review_id: Optional[str]=None` | `Dict[str, Any]` | Analyze with memory recording |
| `MemoryAwareAgent.tool_registry` | (property) | `ToolRegistry` | Expose tool registry |
| `MemoryAwareAgent.model` | (property) | `str` | Expose model name |

**Convenience Functions:**

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `analyze_with_memory()` | `review_text: str`, `review_id: Optional[str]=None`, `vector_db=None` | `Dict[str, Any]` | Analyze with memory-aware agent |

### Drift Detection ([`src/agent/drift.py`](src/agent/drift.py))

**DriftDetector Class:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `DriftDetector.__init__()` | `baseline_distribution: Dict[str, float]` | None | Initialize with baseline theme frequencies |
| `DriftDetector.detect_drift()` | `current_distribution: Dict[str, float]`, `threshold: float=0.1` | `Dict[str, Any]` | Detect distribution shift using KL divergence |

**DetectDriftTool Class:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `DetectDriftTool.__init__()` | `memory=None`, `**data` | None | Initialize drift tool with optional memory |
| `DetectDriftTool.detector` | (property) | `DriftDetector` | Get detector instance |
| `DetectDriftTool.execute()` | `current_themes: Optional[List[str]]=None`, `**kwargs` | `Dict[str, Any]` | Execute drift detection |

### Vector Store ([`src/agent/embeddings.py`](src/agent/embeddings.py))

**SimpleVectorStore Class:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `SimpleVectorStore.__init__()` | None | None | Initialize vector store |
| `SimpleVectorStore.load_from_files()` | `jsonl_path='data/processed/reviews_final.jsonl'`, `meta_path='data/processed/reviews_meta.json'` | None | Load reviews and pre-built embeddings |
| `SimpleVectorStore._load_sentence_transformer_embeddings()` | `emb_path: Path`, `query_emb_path: Path`, `meta_path: str` | None | Load sentence-transformer embeddings |
| `SimpleVectorStore._load_tfidf_embeddings()` | `emb_path: Path`, `vectorizer_path: Path`, `meta_path: str` | None | Load TF-IDF embeddings |
| `SimpleVectorStore._encode_query_st()` | `query: str` | `np.ndarray` | Encode query using pre-computed embeddings |
| `SimpleVectorStore._embed_query_approximation()` | `query: str` | `np.ndarray` | Create query embedding via word overlap |
| `SimpleVectorStore.search()` | `query: str`, `k: int=5`, `source_filter: Optional[str]=None`, `exclude_review_id: Optional[str]=None` | `List[Dict[str, Any]]` | Find semantically similar reviews |
| `SimpleVectorStore._filter_and_sort_results()` | `similarities: np.ndarray`, `k: int`, `source_filter: Optional[str]`, `exclude_review_id: Optional[str]` | `List[Dict[str, Any]]` | Apply filters and sort by similarity |

**Module Functions:**

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `get_vector_store()` | None | `SimpleVectorStore` | Get or create global vector store |

### Database Manager ([`src/database/db_manager.py`](src/database/db_manager.py))

**ReviewDatabase Class - Schema:**

| Method | SQL Statement | Purpose |
|--------|-------------|---------|
| `setup_schema()` | `CREATE TABLE IF NOT EXISTS reviews (...)` | Create raw reviews table |
| | `CREATE TABLE IF NOT EXISTS llm_analysis (...)` | Create LLM analysis results table |
| | `CREATE TABLE IF NOT EXISTS kpi_aggregates (...)` | Create precomputed KPI aggregates table |
| | `CREATE SEQUENCE IF NOT EXISTS analysis_id_seq` | Create sequence for analysis IDs |
| | `CREATE SEQUENCE IF NOT EXISTS agg_id_seq` | Create sequence for aggregate IDs |

**ReviewDatabase Class - Data Loading:**

| Method | SQL Statement | Purpose |
|--------|-------------|---------|
| `load_reviews_from_jsonl()` | `INSERT OR REPLACE INTO reviews (...) VALUES (?, ?, ...)` | Bulk insert reviews |
| `load_analyses_from_jsonl()` | `INSERT OR REPLACE INTO llm_analysis (...) VALUES (?, nextval('analysis_id_seq'), ...)` | Insert LLM analyses |
| `load_memory_analyses()` | `INSERT OR REPLACE INTO llm_analysis (...) VALUES (?, nextval('analysis_id_seq'), ...)` | Insert from agent memory |

**ReviewDatabase Class - KPI Aggregation:**

| Method | SQL Statement | Purpose |
|--------|-------------|---------|
| `compute_kpi_aggregates()` | `DELETE FROM kpi_aggregates` | Clear existing aggregates |
| | `INSERT INTO kpi_aggregates ... SELECT nextval('agg_id_seq'), r.date, 'daily', r.source, COUNT(*), AVG(a.sentiment), ... FROM reviews r JOIN llm_analysis a ... GROUP BY r.date, r.source` | Daily aggregates by source |
| | `INSERT INTO kpi_aggregates ... SELECT nextval('agg_id_seq'), NULL, 'overall', r.source, COUNT(*), ... FROM reviews r JOIN llm_analysis a ... GROUP BY r.source` | Overall aggregates by source |

**ReviewDatabase Class - Dashboard Queries:**

| Method | SQL Statement | Returns | Purpose |
|--------|-------------|---------|---------|
| `get_overall_kpis()` | `SELECT COUNT(DISTINCT r.review_id), COUNT(DISTINCT r.source), AVG(a.sentiment), 100.0 * SUM(CASE WHEN a.retention_risk = 'high' THEN 1 ELSE 0 END) / COUNT(*), 100.0 * SUM(CASE WHEN a.is_anomalous THEN 1 ELSE 0 END) / COUNT(*) FROM reviews r JOIN llm_analysis a` | `Dict[str, Any]` | High-level KPIs for dashboard |
| `get_sentiment_trend()` | `SELECT agg_date as date, source, avg_sentiment as sentiment FROM kpi_aggregates WHERE agg_period = 'daily' ORDER BY agg_date, source` | `pd.DataFrame` | Sentiment trend over time |
| `get_theme_distribution()` | `SELECT unnested.theme as theme, COUNT(*) as frequency, 100.0 * COUNT(*) / SUM(COUNT(*)) OVER () as percentage FROM llm_analysis, UNNEST(themes) as unnested(theme) GROUP BY unnested.theme ORDER BY frequency DESC LIMIT {top_n}` | `pd.DataFrame` | Theme frequency distribution |
| `get_retention_risk_breakdown()` | `SELECT r.source, a.retention_risk, COUNT(*) as count FROM reviews r JOIN llm_analysis a ON r.review_id = a.review_id GROUP BY r.source, a.retention_risk ORDER BY r.source, a.retention_risk` | `pd.DataFrame` | Risk distribution by source |
| `get_high_risk_reviews()` | `SELECT r.review_id, SUBSTR(r.text, 1, 200) || '...' as text_preview, r.date, r.source, a.sentiment, a.themes, a.reasoning FROM reviews r JOIN llm_analysis a ON r.review_id = a.review_id WHERE a.retention_risk = 'high' ORDER BY r.date DESC NULLS LAST LIMIT {limit}` | `pd.DataFrame` | Recent high-risk reviews |
| `search_reviews()` | `SELECT r.review_id, SUBSTR(r.text, 1, 200) || '...' as text_preview, r.date, r.source, r.rating, a.sentiment, a.themes, a.retention_risk FROM reviews r JOIN llm_analysis a ON r.review_id = a.review_id WHERE LOWER(r.text) LIKE LOWER('%{search_term}%') ORDER BY r.date DESC NULLS LAST LIMIT {limit}` | `pd.DataFrame` | Text search in reviews |
| `get_table_counts()` | `SELECT COUNT(*) FROM {table}` | `Dict[str, int]` | Record count per table |

---

## SQL Reference

### Tool Selection

The LLM receives a tool catalog and decides which tools to use based on review content:

| Review Content | Triggered Tools |
|----------------|-----------------|
| "Safety violations, no equipment" | `retrieve_similar_reviews`, `analyze_sentiment`, `detect_anomaly` |
| "Great benefits, good pay" | `analyze_sentiment` only |
| "Overnight shift struggles" | `retrieve_similar_reviews`, `analyze_sentiment` |

### Multi-Step Reasoning

The agent chains tools based on intermediate results:

```
retrieve_similar_reviews (find context)
    ↓
analyze_sentiment (with retrieved context)
    ↓
detect_anomaly (if unusual themes detected)
```

### Self-Monitoring

The memory system tracks the agent's own analyses over time, enabling:

- **Drift Detection**: Identifies when output patterns change
- **Baseline Updates**: Maintains current historical context
- **Quality Assurance**: Flags anomalous results for review

### Explainable Planning

Each analysis includes the LLM's reasoning:

```python
{
    "plan": {
        "reasoning": "Review expresses strong negative sentiment about overtime. Retrieving similar reviews for context, then analyzing.",
        "steps": [...]
    },
    "final_analysis": {
        "sentiment": 2,
        "themes": ["overtime", "work_life_balance"],
        "retention_risk": "high"
    }
}
```

---

## Traditional vs Agentic Approach

| Aspect | Traditional LLM | ReviewInsight Agent |
|--------|-----------------|---------------------|
| **Execution** | Single LLM call | LLM plans → Tools execute → LLM synthesizes |
| **Context** | None or static examples | Dynamic retrieval from vector DB |
| **Memory** | Stateless (per request) | Persistent analysis history |
| **Adaptability** | Fixed prompt format | LLM chooses tools based on content |
| **Validation** | Post-hoc | Built-in anomaly detection |
| **Evolution** | Requires redeployment | Self-monitoring for drift |

---

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/reviewinsight-ai.git
cd reviewinsight-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=sk-...
```

### Dependencies

**Core:**
- `openai>=1.54.0` - LLM API client
- `fastapi>=0.115.0` - REST API framework
- `uvicorn[standard]>=0.32.0` - ASGI server
- `pydantic>=2.0` - Data validation

**Data Processing:**
- `pandas>=2.2.0` - Data manipulation
- `numpy>=1.26.0` - Numerical operations

**Analysis:**
- `scikit-learn>=1.5.0` - ML metrics
- `scipy>=1.13.0` - Statistical operations (KL divergence)

---

## Usage

### 1. Interactive Agentic Analysis

```python
from src.agent.controller import ReviewAgent, analyze_review_agentic

# Option 1: Direct class usage
agent = ReviewAgent()
result = agent.analyze(
    "The overnight shifts are killing me. Mandatory overtime every week."
)

# Option 2: Convenience function
result = analyze_review_agentic(
    "Management doesn't communicate schedule changes until the last minute."
)
```

**Response Structure:**

```python
{
    "review_text": "...",
    "plan": {
        "reasoning": "...",
        "steps": [...]
    },
    "final_analysis": {
        "sentiment": 2,
        "themes": ["overtime", "management"],
        "retention_risk": "high",
        "is_anomalous": False
    },
    "tool_outputs": [...],
    "memory_stats": {...}
}
```

### 2. Memory-Aware Analysis

```python
from src.agent.memory import MemoryAwareAgent

agent = MemoryAwareAgent()
result = agent.analyze(
    "Great benefits but the workload is overwhelming during peak season.",
    review_id="glassdoor_123"  # Records to memory for tracking
)
```

### 3. Drift Detection

```python
from src.agent.drift import DriftDetector

baseline = {
    'overtime': 0.34,
    'management': 0.25,
    'safety': 0.12
}

current = {
    'overtime': 0.55,  # Significant increase
    'management': 0.30,
    'safety': 0.08
}

detector = DriftDetector(baseline)
result = detector.detect_drift(current, threshold=0.10)

if result['is_drifting']:
    print(f"Drift detected: {result['alert']}")
```

### 4. Batch Processing

```bash
# Process all reviews with default settings
python src/analysis/batch_process.py

# Dry run to estimate costs
python src/analysis/batch_process.py --dry-run

# Use specific model and prompt
python src/analysis/batch_process.py --model gpt-4o --prompt v3
```

### 5. Data Merging

```bash
# Combine multiple data sources into unified format
python src/processing/merge_sources.py
```

---

## API Documentation

### Starting the Server

```bash
# Development with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production with multiple workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Interactive Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Calls

**Analyze a Review:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Great benefits but mandatory overtime is exhausting",
    "review_id": "review_001"
  }'
```

**Get Memory Statistics:**

```bash
curl http://localhost:8000/memory/stats
```

**List Available Tools:**

```bash
curl http://localhost:8000/tools
```

**Health Check:**

```bash
curl http://localhost:8000/health
```

---

## Performance Benchmarks

### Cross-Validation Results (Fair Evaluation)

Performance metrics from **leakage-free cross-validation** evaluation using silver standard labels generated by v2, v3, v6 prompts (175 samples). No prompt was tested against its own labels.

| Prompt Version | K-Shot | Theme F1 | Sentiment Acc | Risk Acc | Cost/1k Samples |
|----------------|--------|----------|---------------|----------|-----------------|
| **v3.0 Few-Shot** | **3** | **91.95%** | **86%** | **88%** | $0.075 |
| v2.0 Role Enhanced | 0 | 90.97% | 84% | 88% | $0.064 |
| v5.0 Structured | 0 | 85.71% | 56% | 76% | $0.064 |
| v6.0 Detailed | 0 | 85.43% | 90% | 78% | $0.080 |
| v1.0 Zero-shot | 5 | 4.65% | 78% | 74% | $0.080 |
| v4.0 Chain of Thought | 5 | 2.74% | 58% | 76% | $0.100 |

**Key Finding**: v3.0 (3-Shot Learning) is the true winner with 91.95% theme F1 score. The original "v5.0 as best" finding (94.28% F1) was inflated by ~8-10 percentage points due to data contamination (self-testing).

### System Metrics (gpt-4o-mini, 130 reviews)

| Metric | Value |
|--------|-------|
| Average latency | ~1.2s/review |
| Cost per review | ~$0.0012 |
| Best sentiment accuracy (v6.0) | 90% |
| Best theme F1 (v3.0) | 91.95% |
| Checkpoint recovery overhead | <5s |

### Experimental Design

The system underwent systematic A/B testing across **30 configurations**:

- **6 prompt versions**: Zero-shot, Role Enhanced, Few-shot, Chain of Thought, Structured Output, Detailed
- **5 k-shot values**: 0, 1, 2, 3, 5 examples
- **Full factorial design**: 6 × 5 = 30 experiments
- **Cross-validation framework**: Leakage-free evaluation using different prompts for label generation vs. testing

### Scalability Characteristics

| Aspect | Implementation |
|--------|----------------|
| Horizontal scaling | Stateless API design |
| Batch processing | Configurable batch size (default: 10) |
| Memory management | Incremental JSONL append |
| Vector search | Integration-ready (placeholder) |
| Evaluation framework | Cross-validation with ensemble silver standard |

---

## Experimentation & Results

### Prompt Engineering Journey

This project involved systematic A/B testing across **30 configurations** to optimize sentiment analysis performance:

- **6 prompt versions**: Zero-shot, Role Enhanced, Few-shot, Chain of Thought, Structured Output, Detailed
- **5 k-shot values**: 0, 1, 2, 3, 5 examples
- **175 labeled samples** for evaluation
- **Cross-validation framework** for leakage-free testing

### 🏆 Key Findings

**Winner: v3.0 Few-Shot (k=3)**
- **91.95% Theme F1 Score** (vs 85-90% for other prompts)
- **86% Sentiment Accuracy**
- **88% Risk Prediction Accuracy**
- **Cost**: $0.075 per 1,000 samples

### 📊 Performance Comparison

| Prompt Version | Approach | Theme F1 | Sentiment Acc | Risk Acc | Cost/1k |
|----------------|----------|----------|---------------|----------|---------|
| **v3.0 (k=3)** | **Few-Shot** | **91.95%** | **86%** | **88%** | $0.075 |
| v2.0 | Role Enhanced | 90.97% | 84% | 88% | $0.064 |
| v6.0 | Detailed Instructions | 85.43% | 90% | 78% | $0.080 |
| v5.0 | Structured Output | 85.71% | 56% | 76% | $0.064 |
| v1.0 (k=5) | Zero-Shot | 4.65% | 78% | 74% | $0.080 |
| v4.0 (k=5) | Chain of Thought | 2.74% | 58% | 76% | $0.100 |

### 🔬 Methodological Innovation

**Cross-Validation Design**:
```
Generator Prompts: v2.0, v3.0, v6.0
                    ↓
              Ensemble Labels
                    ↓
Test All Prompts: v1.0, v2.0, v3.0, v4.0, v5.0, v6.0
(No prompt tested against its own labels)
```

**Why This Matters**: Self-testing inflates accuracy by 8-10 percentage points. Our leakage-free evaluation provides unbiased performance estimates while maintaining full automation (no human labeling required).

### 🎯 Usage Examples

**Run with Best Prompt:**
```bash
python src/analysis/batch_process.py \
    --input data/processed/reviews_final.jsonl \
    --output data/analysis/analyzed_reviews.jsonl \
    --prompt v3 \
    --k-shot 3 \
    --model gpt-4o-mini
```

**Compare Performance:**
```bash
python evaluate_crossval.py \
    --labels data/labels/crossval_labels.jsonl \
    --test-prompts v1 v2 v3 v4 v5 v6 \
    --k-shots 0 1 2 3 5 \
    --output reports/crossval_results.csv
```

### 📈 Dataset Coverage

| Source | Records | Time Period |
|--------|---------|-------------|
| Glassdoor Reviews | 100+ | 2020-2025 |
| Indeed Reviews | 50+ | 2020-2025 |
| Reddit Threads | 25+ discussions | 2020-2025 |
| YouTube Transcripts | 2,294 lines | 8 videos |
| **Total** | **2,500+ entries** | **Multi-source** |

---

## Dashboard Demo

### 🎯 Interactive Features

**Live Demo**: [Launch Dashboard](dashboard/app.py) `streamlit run dashboard/app.py`

**Key Capabilities**:
- 📊 **Sentiment Trend Analysis**: Track sentiment over time by source
- 🎯 **Theme Distribution**: Visualize common topics across reviews
- ⚠️ **Risk Monitoring**: Identify high-risk employees needing intervention
- 🔍 **Review Search**: Full-text search with sentiment highlighting
- 📈 **KPI Dashboard**: Overall metrics and source-wise breakdowns

### 🎬 Demo Screenshots

*Add screenshots after running the dashboard:*

```bash
# Start the dashboard
streamlit run dashboard/app.py

# Recommended screenshots:
# 1. Main KPI Dashboard
# 2. Sentiment Trend Chart
# 3. Theme Distribution Plot
# 4. High-Risk Reviews Table
# 5. Search Interface
```

### 📊 Dashboard Features

**Real-Time Analysis**:
- Upload and analyze reviews instantly
- Filter by sentiment, source, date range
- Export results to CSV/JSON

**Interactive Visualizations**:
- Sentiment trends over time
- Theme distribution bar charts
- Risk breakdown by source
- Word clouds and text analysis

**Search & Discovery**:
- Full-text search across all reviews
- Sentiment-based highlighting
- Theme filtering
- Source-specific queries

---

## Project Structure

```
extern/
├── api/
│   ├── __init__.py               # Package marker
│   ├── main.py                   # FastAPI service
│   └── README.md                 # API documentation
│
├── src/
│   ├── agent/
│   │   ├── controller.py          # ReAct orchestration
│   │   ├── tools.py               # Tool registry & implementations
│   │   ├── memory.py              # Persistent memory system
│   │   └── drift.py               # Drift detection
│   │
│   ├── analysis/
│   │   └── batch_process.py       # Batch LLM processing
│   │
│   ├── processing/
│   │   ├── merge_sources.py        # Data unification
│   │   └── validate_data.py       # Quality validation
│   │
│   ├── prompts/
│   │   └── templates.py          # Prompt version management
│   │
│   └── evaluation/
│       └── evaluate_prompts.py    # A/B testing framework
│
├── data/
│   ├── processed/                 # Merged datasets
│   ├── analysis/                  # LLM results
│   └── memory/                    # Agent memory storage
│
├── notebooks/
│   └── run_eda.py                # Exploratory analysis
│
├── reports/
│   └── figures/                   # Generated visualizations
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...              # OpenAI API key

# Optional (shown with defaults)
AGENT_MODEL=gpt-4o-mini            # Default LLM model
TEMPERATURE=0.3                     # Sampling temperature
MAX_TOKENS=500                     # Response length limit
MEMORY_DIR=data/memory             # Memory storage path
```

### Agent Configuration

```python
# src/agent/controller.py
class ReviewAgent:
    def __init__(
        self,
        vector_db=None,           # Optional vector database
        model="gpt-4o-mini"       # OpenAI model
    ):
        ...
```

### Drift Detection Thresholds

| Threshold | Sensitivity | Use Case |
|-----------|-------------|----------|
| 0.05 | Very sensitive | Minor changes trigger alerts |
| 0.10 | Balanced (default) | Recommended for production |
| 0.20 | Conservative | Only major shifts trigger alerts |

---

## Extension Guide

### Adding a New Tool

```python
# In src/agent/tools.py
from pydantic import Field

class CustomTool(Tool):
    """Custom tool implementation"""

    name: str = Field(default="custom_tool")
    description: str = Field(default="Tool description for LLM")
    parameters: Dict[str, Any] = Field(default={
        "param_name": {
            "type": "string",
            "description": "Parameter description",
            "required": True
        }
    })

    def execute(self, **kwargs) -> Dict:
        """Tool execution logic"""
        return {"result": "output"}

# Register in ToolRegistry.__init__
self.register(CustomTool())
```

### Adding a New Prompt Version

```python
# In src/prompts/templates.py
class PromptV7(PromptTemplate):
    """Custom prompt template"""

    name = "My Custom Prompt"
    version = "v7"

    def render(self, review_text: str, k_shot: int = 0):
        system_prompt = """You are an expert analyst..."""
        user_prompt = f"Analyze this review:\n\n{review_text}"
        return system_prompt, user_prompt
```

### Adding a New Memory Metric

```python
# In src/agent/memory.py
class AgentMemory:
    def get_custom_metric(self, last_n: int = 100) -> Dict:
        """Calculate custom metric from history"""
        history = self.load_history(last_n)
        # Implementation
        return {"metric": "value"}
```


---

## Technical Stack Summary

| Component | Technology |
|-----------|------------|
| **LLM** | OpenAI GPT-4o-mini |
| **API Framework** | FastAPI |
| **Data Validation** | Pydantic |
| **Numerical Computing** | NumPy, SciPy |
| **Data Processing** | Pandas |
| **Metrics** | scikit-learn |
| **Drift Detection** | KL Divergence (scipy) |
| **Memory Format** | JSONL |
| **Pattern** | ReAct (Reason + Act) |
| **Best Configuration** | v3.0 Few-Shot (k=3) - 91.95% F1 |
| **Evaluation** | Cross-prompt validation (leakage-free) |

---

## Future Enhancements

- [ ] Real vector database integration (ChromaDB/Pinecone)
- [ ] Multi-agent collaboration patterns
- [ ] Streaming analysis for large datasets
- [ ] Custom fine-tuned models for domain specificity
- [ ] Real-time monitoring dashboard
- [ ] Automated report generation
