# ReviewInsight AI - Quick Start Guide

Get up and running with ReviewInsight AI in under 10 minutes.

---

## 🚀 Setup (5 Minutes)

### 1. Install Dependencies

```bash
# Clone and navigate
cd extern

# Install all required packages
pip install -r requirements.txt
```

**Required packages:**
- `duckdb` - Embedded SQL database
- `streamlit` - Dashboard framework
- `plotly` - Interactive charts
- `openai` - LLM API
- `pandas`, `tqdm`, `pydantic` - Data processing

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Get your API key from: https://platform.openai.com/api-keys

---

## 🏗️ Build Database (3-5 minutes)

### Option A: With Existing Data

If you have `data/processed/reviews_final.jsonl`:

```bash
python src/pipelines/build_database.py
```

This will:
- ✅ Load reviews from JSONL
- ✅ Analyze each review with LLM agent
- ✅ Store structured results in SQL
- ✅ Compute KPI aggregates
- ✅ Print summary statistics

### Option B: Without Data (Demo Mode)

The system will use mock data from agent memory.

---

## 📊 Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to: **http://localhost:8501**

**Dashboard Features:**
- 📊 **KPI Dashboard** - Fast SQL-powered analytics
- 🤖 **AI Agent** - On-demand LLM analysis
- 🔍 **Search & Explore** - Filter and search reviews

---

## 📁 Project Structure

```
extern/
├── data/
│   ├── processed/
│   │   └── reviews_final.jsonl       # Input: merged reviews
│   ├── database/
│   │   └── reviews.duckdb            # Output: SQL database
│   └── memory/
│       └── analysis_log.jsonl        # Agent memory log
│
├── src/
│   ├── database/
│   │   └── db_manager.py             # SQL layer
│   ├── agent/
│   │   ├── tools.py                  # Agent tools
│   │   ├── controller.py             # Agent orchestration
│   │   ├── memory.py                 # Memory system
│   │   └── drift.py                  # Drift detection
│   └── pipelines/
│       └── build_database.py         # Batch processor
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard
│
├── api/
│   └── main.py                        # REST API
│
├── requirements.txt
└── README.md                          # Full documentation
```

---

## 🔧 Common Workflows

### Analyze a Single Review

```python
from src.agent.controller import analyze_review_agentic

result = analyze_review_agentic(
    "The benefits are good but mandatory overtime is exhausting."
)

print(result['final_analysis'])
# {'sentiment': 2, 'themes': ['overtime', 'work_life_balance'], 'retention_risk': 'high'}
```

### Query the Database

```python
from src.database.db_manager import ReviewDatabase

db = ReviewDatabase()

# Get overall KPIs
kpis = db.get_overall_kpis()
print(f"Average sentiment: {kpis['avg_sentiment']}/5")

# Get theme distribution
themes = db.get_theme_distribution(top_n=5)
print(themes)
```

### Start the API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `POST /analyze` - Agentic review analysis
- `GET /memory/stats` - Historical statistics
- `GET /tools` - Available tool catalog
- `GET /health` - Service health check

---

## 📊 Expected Results

After running `build_database.py`:

```
============================================================
BUILDING REVIEWINSIGHT DATABASE
============================================================

📥 STEP 1: Loading reviews into database...
  Loaded 108 reviews into database

🔍 STEP 2: Identifying reviews to analyze...
Loaded 108 total reviews
Already analyzed: 0 reviews
Need to analyze: 108 reviews

🤖 STEP 3: Analyzing 108 reviews with agent...
  (This may take several minutes...)
Analyzing: 100%|████████| 108/108

  ✅ Successfully analyzed: 108
  ❌ Errors: 0

📊 STEP 4: Computing KPI aggregates...
  Computed 15 KPI aggregates

============================================================
DATABASE BUILD COMPLETE
============================================================

📊 OVERALL STATISTICS:
  Total reviews: 108
  Data sources: 3
  Avg sentiment: 2.87/5
  High risk: 35.2%
  Anomaly rate: 0.0%

🏷️  TOP 5 THEMES:
  overtime: 59 (54.6%)
  management: 43 (39.8%)
  pay_benefits: 39 (36.1%)
  workload: 36 (33.3%)
  work_life_balance: 28 (25.9%)
```

---

## 🐛 Troubleshooting

### "Module not found: duckdb"

```bash
pip install duckdb
```

### "OPENAI_API_KEY not found"

Create `.env` file with your API key:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### "No reviews found"

Ensure `data/processed/reviews_final.jsonl` exists:
```bash
python src/processing/merge_sources.py
```

### Dashboard not loading

```bash
pip install streamlit plotly
streamlit run dashboard/app.py
```

---

## 📚 Next Steps

1. **Explore the Dashboard** - Navigate all three tabs
2. **Try the AI Agent** - Paste a review and see the analysis
3. **Search Reviews** - Look for specific keywords (e.g., "overtime")
4. **Read Full Documentation** - See [README.md](README.md)

---

## 💡 Tips

- **Resume capability**: The pipeline skips already-analyzed reviews
- **Cost estimation**: ~$0.01-0.02 per review analysis
- **Database location**: `data/database/reviews.duckdb` (can be deleted to rebuild)
- **Memory persistence**: Agent memory stored in `data/memory/`

---

## ⚡ Quick Commands

```bash
# Install everything
pip install -r requirements.txt

# Build database from scratch
python src/pipelines/build_database.py

# Launch dashboard
streamlit run dashboard/app.py

# Start API server
uvicorn api.main:app --reload --port 8000

# Run agent on single review
python -c "
from src.agent.controller import analyze_review_agentic
result = analyze_review_agentic('Great benefits but too much overtime')
print(result['final_analysis'])
"
```

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.
