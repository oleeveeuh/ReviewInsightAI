#!/usr/bin/env python3
"""
Streamlit dashboard for ReviewInsight AI.

Hybrid dashboard combining:
- SQL-powered KPI analytics (fast, aggregated)
- AI Agent for on-demand deep analysis
- Search and exploration functionality

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import json

from database.db_manager import ReviewDatabase
from agent.memory import MemoryAwareAgent
from agent.drift import DriftDetector


# Page config
st.set_page_config(
    page_title="ReviewInsight AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_real_aggregates():
    """Frozen aggregates of the real 175-review corpus — counts only, no
    review text (see dashboard/build_real_aggregates.py)."""
    path = Path(__file__).resolve().parent / "real_aggregates.json"
    return json.loads(path.read_text())


REAL = load_real_aggregates()

# Custom CSS - fieldnotes theme (matches oleeveeuh.github.io)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;500;600&family=Spectral:wght@400;500&display=swap');

    :root {
        --paper: #f5f1e8;
        --cream: #fbf9f4;
        --ink: #25251f;
        --ink2: #52514e;
        --muted: #77776c;
        --grid: #d8d1c4;
        --sage: #89927a;
        --crimson: #a43b36;
        --wine: #482728;
        --sumi: #1e342e;
    }

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--paper);
        color: var(--ink);
    }
    /* the theme layer paints this full-page container — keep it paper even
       when a viewer's system dark mode flips streamlit's default theme */
    .stApp {
        background: var(--paper);
        color: var(--ink);
    }

    /* lowercase voice, like the site — but keep typed input as typed */
    .stApp, .stMarkdown, .stText, [data-testid="stMarkdownContainer"] {
        text-transform: lowercase;
    }
    input, textarea { text-transform: none; }

    /* Hide streamlit footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Header styles */
    .main-title {
        font-family: 'Spectral', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 500;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        color: var(--crimson);
        margin-bottom: 0.4rem;
    }

    .subtitle {
        font-size: 0.98rem;
        color: var(--ink2);
        font-weight: 400;
    }

    .honesty {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--wine);
        border: 1px dashed var(--wine);
        background: var(--cream);
        padding: 0.65rem 0.9rem;
        margin: 0.9rem 0 0.3rem;
        max-width: 640px;
    }

    /* Metric cards */
    .metric-container {
        background: var(--cream);
        border: 1px solid var(--grid);
        border-radius: 2px;
        padding: 1.15rem 1.25rem;
        transition: border-color 0.2s;
    }
    .metric-container:hover {
        border-color: var(--sage);
    }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        font-size: 1.6rem;
        color: var(--sumi);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
        padding: 0;
        border-radius: 0;
        border: none;
        border-bottom: 1px solid var(--grid);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 10px 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.1em;
        font-weight: 400;
        color: var(--muted);
        transition: all 0.15s;
        border: none;
    }

    .stTabs [data-baseweb="tab-selected"] {
        background: transparent;
        color: var(--crimson);
        box-shadow: inset 0 -2px 0 var(--crimson);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 2px;
        border: 1px solid var(--grid);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 400;
        background: var(--cream);
        color: var(--ink);
        transition: all 0.15s;
    }

    .stButton > button[kind="primary"] {
        background: var(--crimson);
        border: 1px solid var(--crimson);
        color: #fbf9f4;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--wine);
        border-color: var(--wine);
        transform: none;
        box-shadow: none;
    }

    /* Info boxes */
    .stInfo, .stSuccess, .stWarning {
        border-radius: 2px;
        border: 1px solid;
    }

    .stInfo {
        background: var(--cream);
        border-color: var(--grid);
        color: var(--ink2);
    }

    .stSuccess {
        background: #e9ede3;
        border-color: var(--sage);
    }

    .stWarning {
        background: #f2e9e2;
        border-color: var(--crimson);
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 2px;
        overflow: hidden;
        border: 1px solid var(--grid);
    }

    /* Section dividers */
    hr {
        border: none;
        border-top: 1px solid var(--grid);
        margin: 2rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--cream);
        border-right: 1px solid var(--grid);
    }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        color: var(--ink2);
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--ink);
        font-weight: 600;
    }
    h3 {
        font-family: 'Spectral', Georgia, serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize systems (cached)
@st.cache_resource
def init_systems():
    """Initialize database and agent (cached for performance)"""
    try:
        # cloud-safe bootstrap: build from the tracked fixtures when the
        # database is missing or empty; an existing local build is untouched
        from prepare_database import ensure_database
        boot = ensure_database()
        print(f"dashboard db bootstrap: {boot['action']}")
        db = ReviewDatabase()
        agent = MemoryAwareAgent()
        return db, agent, None
    except Exception as e:
        return None, None, str(e)


def answer_pattern_question(question: str, db, agent) -> dict:
    """Rule-based pattern summary from local aggregates (offline, no LLM).

    Matches keywords in the question against theme/sentiment/risk aggregates
    and returns a templated answer with its data sources. Honest scope: this
    is aggregation, not natural-language question answering.
    """
    q = question.lower()
    sources = []

    theme_dist = db.get_theme_distribution(top_n=10)
    kpis = db.get_overall_kpis()

    matched_theme = None
    if not theme_dist.empty:
        for theme in theme_dist['theme']:
            if str(theme).lower().replace('_', ' ') in q or str(theme).lower() in q:
                matched_theme = theme
                break

    lines = []
    total = kpis.get('total_reviews', 0)
    lines.append(f"Based on {total} analyzed reviews in the local database:")

    if matched_theme:
        row = theme_dist[theme_dist['theme'] == matched_theme].iloc[0]
        pct = row['percentage']
        lines.append(
            f"- The theme **{matched_theme}** appears in about {pct:.0f}% of "
            f"theme mentions across analyzed reviews.")
        sources.append("get_theme_distribution()")

        try:
            hits = db.search_reviews(search_term=str(matched_theme).replace('_', ' '),
                                     limit=3)
            if not hits.empty:
                examples = ", ".join(hits['review_id'].tolist())
                lines.append(f"- Example reviews mentioning it: {examples}.")
                sources.append("search_reviews()")
        except Exception:
            pass
    else:
        top = theme_dist.iloc[0] if not theme_dist.empty else None
        if top is not None:
            lines.append(
                f"- The most frequent theme is **{top['theme']}** "
                f"(~{top['percentage']:.0f}% of theme mentions).")
            sources.append("get_theme_distribution()")

    avg = kpis.get('avg_sentiment')
    if avg is not None:
        lines.append(f"- Average LLM-assigned sentiment is {avg:.2f} on a 1-5 scale.")
        sources.append("get_overall_kpis()")

    risk = kpis.get('high_risk_pct')
    if risk is not None:
        lines.append(
            f"- {risk:.0f}% of reviews carry a **high** LLM-assigned review-risk "
            f"class (an LLM-generated classification, not observed attrition).")
        sources.append("get_overall_kpis()")

    lines.append(
        "_Note: this is a rule-based summary of local database aggregates, "
        "not a language-model answer._")

    return {'answer': "\n".join(lines), 'sources': sources}

@st.cache_data
def get_drift_status(_db):
    """Check for drift in theme distribution"""
    try:
        baseline = {
            'overtime': 0.55,
            'management': 0.40,
            'pay_benefits': 0.36,
            'workload': 0.33,
            'work_life_balance': 0.26,
            'culture': 0.15,
            'safety': 0.12,
            'career_growth': 0.10,
            'training': 0.08,
            'other': 0.20
        }

        current = _db.get_theme_distribution(top_n=50)
        if current.empty:
            return None

        current_dist = dict(zip(current['theme'], current['frequency'] / 100))
        detector = DriftDetector(baseline)
        return detector.detect_drift(current_dist, threshold=0.1)
    except Exception:
        return None

@st.cache_data
def get_kpis(_db):
    """Calculate KPI metrics"""
    try:
        stats = _db.get_statistics()
        sentiment_dist = _db.get_sentiment_distribution()

        avg_sentiment = stats.get('avg_sentiment', 0)
        high_risk_pct = stats.get('high_risk_pct', 0)
        anomaly_rate = stats.get('anomaly_rate', 0)

        # Calculate positive sentiment percentage
        if not sentiment_dist.empty:
            positive_pct = sentiment_dist[sentiment_dist['sentiment'] >= 4]['count'].sum() / \
                          sentiment_dist['count'].sum() * 100
        else:
            positive_pct = 0

        return {
            'avg_sentiment': avg_sentiment,
            'positive_pct': positive_pct,
            'high_risk_pct': high_risk_pct,
            'anomaly_rate': anomaly_rate
        }
    except Exception:
        return {
            'avg_sentiment': 0,
            'positive_pct': 0,
            'high_risk_pct': 0,
            'anomaly_rate': None
        }

# Initialize
db, agent, error = init_systems()

if error or db is None:
    st.error(f"Failed to initialize: {error}")
    st.stop()

# Custom header
st.markdown("""
<div style="margin-bottom: 1.4rem;">
    <p class="eyebrow">reviewinsight ai · a field-notes dashboard</p>
    <h1 class="main-title">warehouse reviews, read by an llm pipeline</h1>
    <p class="subtitle">sql-powered kpis, a plan-and-execute agent, and prompt a/b evaluation over the merged
    glassdoor · reddit · youtube corpus. headline numbers are aggregates of the real 175-review corpus
    (counts only — the text stays private); review-level views run on the tracked synthetic fixtures.</p>
    <div class="honesty">every metric on this page is agreement with an llm labeler — not accuracy against
    human ground truth. the word accuracy is reserved in this codebase for a human-labeled sample that
    does not exist yet.</div>
</div>
""", unsafe_allow_html=True)

# Drift status in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Distribution Drift")

    drift_result = get_drift_status(db)

    if drift_result is None or not isinstance(drift_result, dict):
        st.caption("No drift data available")
    elif drift_result.get('has_drift'):
        st.error(
            f"**Drift Detected**\n\n"
            f"KL Divergence: {drift_result['kl_divergence']:.3f}\n\n"
            f"**Top changes:**"
        )
        for theme, info in list(drift_result['significant_changes'].items())[:3]:
            direction_arrow = "^" if info["direction"] == "increase" else "v"
            change_val = abs(info["change"] * 100)
            st.caption(f"{direction_arrow} {theme}: {change_val:.1f}%")
        with st.expander("What is drift?"):
            st.markdown("""
            **Distribution drift** occurs when the theme distribution in recent
            reviews significantly differs from historical patterns.

            **KL Divergence** measures this difference (higher = more drift).

            **Significant change**: Any theme that shifted by >5 percentage points.
            """)
    else:
        st.success(
            f"**No Drift**\n\n"
            f"KL Divergence: {drift_result['kl_divergence']:.3f}\n\n"
            f"Recent themes match historical patterns."
        )

    st.markdown("---")
    st.markdown("### Filters")
    date_range = st.selectbox("Time Period", ["All Time", "Last 30 Days", "Last 7 Days"])
    source_filter = st.multiselect("Source", ["Amazon", "Reddit", "Glassdoor"], default=["Amazon", "Reddit", "Glassdoor"])

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Analysis", "Search", "Experiments"])

# ===== TAB 1: DASHBOARD =====
with tab1:
    st.caption("headline numbers: the real 175-review corpus, aggregates only — review text "
               "stays private (DATASET_CARD.md) · review-level views further down run on the 9 "
               "tracked synthetic fixtures")

    # KPI cards — real corpus aggregates
    kpis = REAL["kpis"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">avg sentiment · real corpus</div>
            <div class="metric-value">{kpis['avg_sentiment']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">positive sentiment · real corpus</div>
            <div class="metric-value" style="color: #89927a;">{kpis['positive_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        risk_color = "#a43b36" if kpis['high_risk_pct'] > 20 else "#b86a62"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">high retention risk · real corpus</div>
            <div class="metric-value" style="color: {risk_color};">{kpis['high_risk_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">anomaly rate · real corpus (rule-flagged)</div>
            <div class="metric-value" style="color: #25251f;">{kpis['anomaly_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row — real corpus aggregates
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Sentiment Trend")
        df_trend = pd.DataFrame(REAL["trend"])
        pivot = df_trend.pivot(index="month", columns="source", values="avg_sentiment")

        fig = px.line(
            pivot,
            markers=True,
            labels={'value': 'Average Sentiment', 'month': 'Month', 'source': 'Source'},
            line_shape='spline',
            color_discrete_map={"glassdoor": "#1E342E", "reddit": "#89927a",
                                "youtube": "#d4708c"},
        )

        fig.add_hline(y=3.0, line_dash="dash", line_color="#b86a62",
                     annotation_text="Neutral")

        fig.update_layout(
            yaxis_range=[1, 5],
            hovermode='x unified',
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='#fbf9f4',
            plot_bgcolor='#fbf9f4',
            font=dict(size=12, color="#52514e"),
            legend_title_text=''
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Theme Distribution")
        themes = pd.DataFrame(
            sorted(REAL["themes"].items(), key=lambda kv: kv[1]),
            columns=['theme', 'mentions']
        )

        if not themes.empty:
            fig = px.bar(
                themes,
                x='mentions',
                y='theme',
                orientation='h',
                labels={'mentions': 'Mentions', 'theme': ''},
                text_auto='.0f'
            )

            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='#fbf9f4',
                plot_bgcolor='#fbf9f4',
                font=dict(size=12, color="#52514e")
            )
            fig.update_traces(marker_color="#89927a", textposition='outside')

            st.plotly_chart(fig, use_container_width=True)

    st.caption("labels: v5 silver labels, one per review across the real corpus — agreement "
               "with an llm labeler, not accuracy · risk is an llm classification of text, not "
               "observed attrition · anomaly rate: the repo's detect_anomaly rule "
               "(negative sentiment + safety, or pay_benefits + management at negative sentiment)")

    st.markdown("---")
    st.caption("review-level views — the 9 tracked synthetic fixtures (the real corpus text stays private)")

    # Drift detection section
    st.markdown("### Distribution Drift Analysis")

    if drift_result is None or not isinstance(drift_result, dict):
        st.info("Drift analysis requires historical data.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            has_drift = drift_result.get('has_drift', False)
            status = "Detected" if has_drift else "No Drift"
            status_color = "#a43b36" if has_drift else "#89927a"

            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.85rem; color: #77776c; margin-bottom: 0.5rem;">Drift Status</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {status_color};">
                    {status}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            kl_div = drift_result.get('kl_divergence', 0)
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.85rem; color: #77776c; margin-bottom: 0.5rem;">KL Divergence</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #25251f;">
                    {kl_div:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            changes = drift_result.get('significant_changes', {})
            change_count = len(changes)
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.85rem; color: #77776c; margin-bottom: 0.5rem;">Significant Changes</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #25251f;">
                    {change_count}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if drift_result.get('significant_changes'):
            st.markdown("#### Significant Theme Changes")

            changes_data = []
            for theme, info in drift_result['significant_changes'].items():
                direction_icon = "^" if info['direction'] == 'increase' else "v"
                change_pct = abs(info['change'] * 100)
                baseline_pct = info['baseline'] * 100
                current_pct = info['current'] * 100

                if info['direction'] == 'increase':
                    interpretation = "Increasing mentions"
                else:
                    interpretation = "Decreasing mentions"

                changes_data.append({
                    'Theme': f"{direction_icon} {theme}",
                    'Baseline': f"{baseline_pct:.1f}%",
                    'Current': f"{current_pct:.1f}%",
                    'Change': f"{change_pct:+.1f}%",
                    'Meaning': interpretation
                })

            changes_df = pd.DataFrame(changes_data)
            changes_df = changes_df.sort_values('Change', key=lambda x: abs(x.str.rstrip('%').astype(float)), ascending=False)
            st.dataframe(changes_df, use_container_width=True, hide_index=True)

            # Visual comparison chart
            chart_data = []
            for theme, info in drift_result.get('significant_changes', {}).items():
                chart_data.append({
                    'Theme': theme,
                    'Baseline': info['baseline'] * 100,
                    'Current': info['current'] * 100,
                    'Direction': info['direction']
                })

            chart_df = pd.DataFrame(chart_data)
            chart_df = chart_df.sort_values('Current', ascending=False)

            fig = go.Figure()
            for theme in chart_df['Theme']:
                row = chart_df[chart_df['Theme'] == theme].iloc[0]
                fig.add_trace(go.Bar(
                    name=f"{theme} (baseline)",
                    x=[row['Baseline']],
                    y=[theme],
                    orientation='h',
                    marker_color='#d8d1c4',
                    showlegend=False
                ))
                fig.add_trace(go.Bar(
                    name=f"{theme} (current)",
                    x=[row['Current']],
                    y=[theme],
                    orientation='h',
                    marker_color='#a43b36' if row['Direction'] == 'increase' else '#89927a',
                    showlegend=False
                ))

            fig.update_layout(
                barmode='overlay',
                xaxis_title='Frequency (%)',
                height=max(200, len(chart_df) * 40),
                hovermode='x unified',
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='#fbf9f4',
                plot_bgcolor='#fbf9f4'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant theme changes detected from baseline.")

    # High risk reviews
    st.markdown("---")
    st.markdown("### High Retention Risk Reviews")

    high_risk = db.get_high_risk_reviews(limit=5)

    if not high_risk.empty:
        for _, row in high_risk.iterrows():
            with st.expander(f"Sentiment: {row['sentiment']}/5 | {row['source'].upper()}"):
                # the full review, not a preview — line breaks preserved
                full_text = row.get('text') or row.get('text_preview') or 'No text available'
                st.markdown(str(full_text).replace("\n", "  \n"))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Source: {row.get('source', 'N/A')}")
                with col2:
                    review_date = row.get('date')
                    st.caption(f"Date: {str(review_date)[:10] if review_date is not None else 'N/A'}")
                with col3:
                    retention_risk = row.get('retention_risk', 'unknown')
                    risk_badge = "HIGH" if retention_risk == 'high' else "MEDIUM" if retention_risk == 'medium' else "LOW"
                    st.caption(f"Risk: {risk_badge}")

                themes = row.get('themes')
                if themes is not None and len(themes) > 0:
                    st.caption(f"Themes: {', '.join(themes)}")

                reasoning = row.get('reasoning')
                if reasoning:
                    st.caption(f"Label provenance: {reasoning}")
    else:
        st.info("No high-risk reviews found")

# ===== TAB 2: ANALYSIS =====
with tab2:
    st.markdown("### AI-Powered Analysis")
    st.markdown("*On-demand analysis using LLM agent with tool orchestration*")
    st.caption("needs an openai api key at request time, so this public deployment stays keyless "
               "by design — run the dashboard locally with your own key to use this tab.")

    st.markdown("---")

    analysis_mode = st.radio(
        "Choose analysis mode:",
        ["Analyze New Review", "Ask About Patterns"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if analysis_mode == "Analyze New Review":
        st.markdown("#### Analyze a New Employee Review")

        review_input = st.text_area(
            "Paste employee review here:",
            height=120,
            placeholder="Example: The benefits are good but the mandatory overtime is exhausting...",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            analyze_button = st.button("Analyze", type="primary", use_container_width=True)

        if analyze_button and review_input:
            with st.spinner("Analyzing review..."):
                try:
                    result = agent.analyze(review_input)
                    st.success("Analysis Complete")

                    analysis = result['final_analysis']

                    # Results cards
                    col1, col2, col3 = st.columns(3)

                    sentiment_color = "#89927a" if analysis['sentiment'] >= 4 else "#b86a62" if analysis['sentiment'] == 3 else "#a43b36"
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div style="font-size: 0.85rem; color: #77776c; margin-bottom: 0.5rem;">Sentiment</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: {sentiment_color};">
                                {analysis['sentiment']}/5
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        risk = analysis['retention_risk'].title()
                        risk_color = {"High": "#a43b36", "Medium": "#b86a62", "Low": "#89927a"}.get(risk, "#6b7280")
                        st.markdown(f"""
                        <div class="metric-container">
                            <div style="font-size: 0.85rem; color: #77776c; margin-bottom: 0.5rem;">Retention Risk</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: {risk_color};">
                                {risk}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        theme_count = len(analysis.get('themes', []))
                        st.markdown(f"""
                        <div class="metric-container">
                            <div style="font-size: 0.85rem; color: #77776c; margin-bottom: 0.5rem;">Themes Found</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #25251f;">
                                {theme_count}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Identified Themes")
                        themes = analysis.get('themes', [])
                        if themes:
                            for theme in themes:
                                st.markdown(f"- {theme.replace('_', ' ').title()}")
                        else:
                            st.caption("No themes identified")

                    with col2:
                        st.markdown("#### Agent Reasoning")
                        reasoning = result.get('reasoning', [])
                        if reasoning:
                            for step in reasoning:
                                st.caption(f"• {step}")
                        else:
                            st.caption("No reasoning available")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    else:
        st.markdown("#### Ask About Patterns in the Data")

        question = st.text_input(
            "Your question:",
            placeholder="Example: What are the main complaints about overtime?",
            label_visibility="collapsed"
        )

        if st.button("Get Answer", type="primary"):
            if question:
                with st.spinner("Summarizing patterns..."):
                    try:
                        # Rule-based pattern summary from local aggregates
                        # (no LLM call, works offline)
                        response = answer_pattern_question(question, db, agent)

                        st.markdown("#### Answer (rule-based summary of local aggregates)")
                        st.markdown(response['answer'])

                        if response.get('sources'):
                            st.markdown("**Sources:**")
                            for source in response['sources']:
                                st.caption(f"- {source}")
                    except Exception as e:
                        st.error(f"Query failed: {type(e).__name__}")
            else:
                st.warning("Please enter a question")

# ===== TAB 3: SEARCH =====
with tab3:
    st.markdown("### Search & Explore Reviews")
    st.markdown("*Search through analyzed reviews with filters*")
    st.caption("searching the 9 tracked synthetic fixtures — the real corpus text stays private")

    st.markdown("---")

    # Search interface
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_term = st.text_input("Search reviews:", placeholder="Keyword or phrase...")

    with col2:
        sentiment_filter = st.selectbox("Sentiment", ["All", "1", "2", "3", "4", "5"], label_visibility="collapsed")

    with col3:
        risk_filter = st.selectbox("Risk", ["All", "High", "Medium", "Low"], label_visibility="collapsed")

    if search_term or st.button("Search", type="primary", key="search_btn"):
        if search_term:
            results = db.search_reviews(
                search_term=search_term,
                sentiment=int(sentiment_filter) if sentiment_filter != "All" else None,
                retention_risk=risk_filter.lower() if risk_filter != "All" else None,
                limit=20
            )

            st.markdown(f"#### Found {len(results)} results")

            if not results.empty:
                for _, row in results.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.markdown(f"**{row['source'].upper()}** | {row['date'] if pd.notna(row['date']) else 'No date'}")
                        st.markdown(row['text_preview'])

                    with col2:
                        st.markdown(f"**{row['sentiment']}/5**")

                    with col3:
                        risk_badge = "HIGH" if row['retention_risk'] == 'high' else "MED" if row['retention_risk'] == 'medium' else "LOW"
                        st.markdown(f"**{risk_badge}**")

                    themes = row.get('themes')
                    if themes is not None and len(themes) > 0:
                        st.caption(f"Themes: {', '.join(themes)}")

                    st.markdown("---")
            else:
                st.info("No results found")

# ===== TAB 4: EXPERIMENTS =====
with tab4:
    st.markdown("### Prompt A/B Testing Results")
    st.markdown("*Compare performance across different prompt versions and configurations*")

    st.markdown("---")

    # Evaluation caveats
    st.info("""
    **What these numbers are**: agreement with **LLM-generated silver labels**
    (gpt-4o-mini) — not accuracy against human ground truth. The historical
    grid runs were scored against labels produced by v5 itself, so absolute
    values (especially v5.0's) are inflated by self-testing. See
    docs/EVALUATION.md for the corrected leave-one-prompt-out re-scoring.
    """)

    import json

    eval_dir = Path(__file__).parent.parent / 'data' / 'evaluation'
    # rescored_crossval_summary.json is the only evaluation artifact the repo
    # tracks (DATASET_CARD); raw grid files stay local-only. Prefer it, then
    # any local grid/crossval files for offline use.
    result_files = list(eval_dir.glob('rescored_crossval_summary.json')) + \
                   list(eval_dir.glob('experiment_grid_*.json')) + \
                   list(eval_dir.glob('experiment_*.json')) + \
                   list(eval_dir.glob('experiment_crossval*.json'))
    result_files = list(dict.fromkeys(result_files))

    if not result_files:
        st.info("No experiment results found. Run `python run_experiments.py --run-all` to generate results.")
    else:
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r') as f:
            data = json.load(f)

        results = data if isinstance(data, list) else data.get('results', [])

        if not results:
            st.warning("No results in file")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Data source: `{latest_file.name}`")
            with col2:
                st.caption(f"{len(results)} experiments completed")

            # Convert to DataFrame (handles both the rescored summary's
            # rescored_agreement schema and the raw grid's metrics schema)
            df_results = []
            for r in results:
                metrics = r.get('rescored_agreement', r.get('metrics', {}))
                df_results.append({
                    'prompt': r.get('prompt_name', r.get('prompt_version', 'Unknown')),
                    'version': r.get('prompt_version', 'v?'),
                    'k_shot': r.get('k_shot', 0),
                    'model': r.get('model', 'unknown'),
                    'sentiment_mae': metrics.get('sentiment_mae', 0),
                    'sentiment_acc': metrics.get('sentiment_exact_accuracy',
                                                 metrics.get('sentiment_exact_agreement', 0)),
                    'theme_f1': metrics.get('theme_f1', 0),
                    'theme_precision': metrics.get('theme_precision', 0),
                    'theme_recall': metrics.get('theme_recall', 0),
                    'risk_acc': metrics.get('risk_accuracy',
                                            metrics.get('risk_agreement', 0))
                })

            df = pd.DataFrame(df_results)
            df = df[df['theme_f1'] > 0].copy()

            if df.empty:
                st.warning("No valid experiments with data")
            else:
                # Summary metrics
                st.markdown("#### Performance Summary")

                # Highlight cross-validation results
                is_crossval = any('crossval' in latest_file.name for r in results if r.get('leakage_free'))
                if is_crossval:
                    st.success("**Cross-Validation Mode**: Fair, leakage-free evaluation")

                col1, col2, col3 = st.columns(3)

                with col1:
                    best_idx = df['sentiment_acc'].idxmax()
                    st.metric(
                        "Best Sentiment Acc",
                        f"{df.loc[best_idx, 'sentiment_acc']:.1%}",
                        help=f"{df.loc[best_idx, 'prompt']} (k={df.loc[best_idx, 'k_shot']})"
                    )

                with col2:
                    best_idx = df['theme_f1'].idxmax()
                    st.metric(
                        "Best Theme F1",
                        f"{df.loc[best_idx, 'theme_f1']:.1%}",
                        help=f"{df.loc[best_idx, 'prompt']} (k={df.loc[best_idx, 'k_shot']})",
                        delta_color="normal"
                    )
                    # Add crown icon for winner
                    if df.loc[best_idx, 'prompt'] == 'v3.0 Few-Shot Learning':
                        st.caption("🏆 True Winner (Cross-Validation)")

                with col3:
                    best_idx = df['risk_acc'].idxmax()
                    st.metric(
                        "Best Risk Acc",
                        f"{df.loc[best_idx, 'risk_acc']:.1%}",
                        help=f"{df.loc[best_idx, 'prompt']} (k={df.loc[best_idx, 'k_shot']})"
                    )

                st.markdown("---")

                # Chart selection
                chart_type = st.radio(
                    "Select view:",
                    ["Leaderboard", "Theme Metrics"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                if chart_type == "Leaderboard":
                    st.markdown("#### Experiment Leaderboard")

                    df_sorted = df.sort_values('theme_f1', ascending=False).reset_index(drop=True)
                    display_df = df_sorted[['prompt', 'k_shot', 'sentiment_acc', 'theme_f1', 'risk_acc']].copy()
                    display_df.columns = ['Prompt', 'K-shot', 'Sentiment Acc', 'Theme F1', 'Risk Acc']
                    display_df['Sentiment Acc'] = (display_df['Sentiment Acc'] * 100).round(1).astype(str) + '%'
                    display_df['Theme F1'] = (display_df['Theme F1'] * 100).round(1).astype(str) + '%'
                    display_df['Risk Acc'] = (display_df['Risk Acc'] * 100).round(1).astype(str) + '%'

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                elif chart_type == "Theme Metrics":
                    st.markdown("#### Theme Classification Metrics")

                    df_melted = df.melt(
                        id_vars=['prompt', 'k_shot'],
                        value_vars=['theme_precision', 'theme_recall', 'theme_f1'],
                        var_name='metric',
                        value_name='score'
                    )

                    df_melted['label'] = df_melted['prompt'] + ' (k=' + df_melted['k_shot'].astype(str) + ')'
                    df_melted['metric'] = df_melted['metric'].replace({
                        'theme_precision': 'Precision',
                        'theme_recall': 'Recall',
                        'theme_f1': 'F1 Score'
                    })

                    fig = px.bar(
                        df_melted,
                        x='label',
                        y='score',
                        color='metric',
                        barmode='group',
                        labels={'label': 'Experiment', 'score': 'Score'},
                        title='Precision, Recall, and F1 by Experiment'
                    )

                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=500,
                        yaxis_range=[0, 1],
                        paper_bgcolor='#fbf9f4',
                        plot_bgcolor='#fbf9f4',
                        font=dict(color="#52514e")
                    )

                    st.plotly_chart(fig, use_container_width=True)



# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #b86a62; font-size: 0.85rem;'>
    ReviewInsight AI | Hybrid LLM + SQL Analytics Platform
</div>
""", unsafe_allow_html=True)
