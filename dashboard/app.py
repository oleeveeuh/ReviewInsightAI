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

from database.db_manager import ReviewDatabase
from agent.memory import MemoryAwareAgent
from agent.drift import DriftDetector

# Page config
st.set_page_config(
    page_title="ReviewInsight AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern, clean design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide streamlit footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Header styles */
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .metric-container:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f9fafb;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        color: #6b7280;
        transition: all 0.15s;
    }

    .stTabs [data-baseweb="tab-selected"] {
        background: white;
        color: #111827;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.15s;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Info boxes */
    .stInfo, .stSuccess, .stWarning {
        border-radius: 10px;
        border: 1px solid;
    }

    .stInfo {
        background: #f0f9ff;
        border-color: #bae6fd;
    }

    .stSuccess {
        background: #f0fdf4;
        border-color: #bbf7d0;
    }

    .stWarning {
        background: #fffbeb;
        border-color: #fde68a;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Section dividers */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }

    /* Headings */
    h1, h2, h3 {
        color: #111827;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize systems (cached)
@st.cache_resource
def init_systems():
    """Initialize database and agent (cached for performance)"""
    try:
        db = ReviewDatabase()
        agent = MemoryAwareAgent()
        return db, agent, None
    except Exception as e:
        return None, None, str(e)

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
    except Exception as e:
        return None

@st.cache_data
def get_kpis(_db):
    """Calculate KPI metrics"""
    try:
        stats = _db.get_statistics()
        sentiment_dist = _db.get_sentiment_distribution()

        avg_sentiment = stats.get('avg_sentiment', 0)
        high_risk_pct = stats.get('high_risk_percentage', 0)
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
    except Exception as e:
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
<div style="margin-bottom: 2rem;">
    <h1 class="main-title">ReviewInsight AI</h1>
    <p class="subtitle">Employee sentiment analysis powered by LLM agents and SQL analytics</p>
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
    # KPI cards
    kpis = get_kpis(db)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Avg Sentiment</div>
            <div style="font-size: 1.75rem; font-weight: 600; color: #111827;">
                {kpis['avg_sentiment']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Positive Sentiment</div>
            <div style="font-size: 1.75rem; font-weight: 600; color: #059669;">
                {kpis['positive_pct']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        risk_color = "#dc2626" if kpis['high_risk_pct'] > 20 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">High Retention Risk</div>
            <div style="font-size: 1.75rem; font-weight: 600; color: {risk_color};">
                {kpis['high_risk_pct']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        anomaly_display = f"{kpis['anomaly_rate']:.1f}%" if kpis['anomaly_rate'] is not None else "N/A"
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Anomaly Rate</div>
            <div style="font-size: 1.75rem; font-weight: 600; color: #111827;">
                {anomaly_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Sentiment Trend")
        df_trend = db.get_sentiment_trend()

        if not df_trend.empty and 'date' in df_trend.columns:
            df_trend_agg = df_trend.groupby('date')['sentiment'].mean().reset_index()

            fig = px.line(
                df_trend_agg,
                x='date',
                y='sentiment',
                markers=True,
                labels={'sentiment': 'Average Sentiment', 'date': 'Date'},
                line_shape='spline'
            )

            fig.add_hline(y=3.0, line_dash="dash", line_color="#9ca3af",
                         annotation_text="Neutral")

            fig.update_layout(
                yaxis_range=[1, 5],
                hovermode='x unified',
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(size=12, color="#374151")
            )
            fig.update_traces(line_color="#667eea", marker_size=8)

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Theme Distribution")
        themes = db.get_theme_distribution(top_n=10)

        if not themes.empty:
            fig = px.bar(
                themes,
                x='frequency',
                y='theme',
                orientation='h',
                labels={'frequency': 'Percentage', 'theme': ''},
                text_auto='.1f%'
            )

            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(size=12, color="#374151")
            )
            fig.update_traces(
                marker_color="#667eea",
                textposition='outside'
            )
            fig.update_yaxes(categoryorder='total ascending')

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Drift detection section
    st.markdown("### Distribution Drift Analysis")

    if drift_result is None or not isinstance(drift_result, dict):
        st.info("Drift analysis requires historical data.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            has_drift = drift_result.get('has_drift', False)
            status = "Detected" if has_drift else "No Drift"
            status_color = "#dc2626" if has_drift else "#059669"

            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Drift Status</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {status_color};">
                    {status}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            kl_div = drift_result.get('kl_divergence', 0)
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">KL Divergence</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #111827;">
                    {kl_div:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            changes = drift_result.get('significant_changes', {})
            change_count = len(changes)
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Significant Changes</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #111827;">
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
                    interpretation = f"Increasing mentions"
                else:
                    interpretation = f"Decreasing mentions"

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
                    marker_color='#d1d5db',
                    showlegend=False
                ))
                fig.add_trace(go.Bar(
                    name=f"{theme} (current)",
                    x=[row['Current']],
                    y=[theme],
                    orientation='h',
                    marker_color='#dc2626' if row['Direction'] == 'increase' else '#059669',
                    showlegend=False
                ))

            fig.update_layout(
                barmode='overlay',
                xaxis_title='Frequency (%)',
                height=max(200, len(chart_df) * 40),
                hovermode='x unified',
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='white',
                plot_bgcolor='white'
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
                st.markdown(f"**{row.get('text_preview', row.get('text', 'No text available'))}**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Source: {row.get('source', 'N/A')}")
                with col2:
                    st.caption(f"Date: {row.get('date', 'N/A')}")
                with col3:
                    retention_risk = row.get('retention_risk', 'unknown')
                    risk_badge = "HIGH" if retention_risk == 'high' else "MEDIUM" if retention_risk == 'medium' else "LOW"
                    st.caption(f"Risk: {risk_badge}")

                themes = row.get('themes')
                if themes is not None and len(themes) > 0:
                    st.caption(f"Themes: {', '.join(themes)}")
    else:
        st.info("No high-risk reviews found")

# ===== TAB 2: ANALYSIS =====
with tab2:
    st.markdown("### AI-Powered Analysis")
    st.markdown("*On-demand analysis using LLM agent with tool orchestration*")

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

                    sentiment_color = "#059669" if analysis['sentiment'] >= 4 else "#f59e0b" if analysis['sentiment'] == 3 else "#dc2626"
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Sentiment</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: {sentiment_color};">
                                {analysis['sentiment']}/5
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        risk = analysis['retention_risk'].title()
                        risk_color = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#059669"}.get(risk, "#6b7280")
                        st.markdown(f"""
                        <div class="metric-container">
                            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Retention Risk</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: {risk_color};">
                                {risk}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        theme_count = len(analysis.get('themes', []))
                        st.markdown(f"""
                        <div class="metric-container">
                            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem;">Themes Found</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #111827;">
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
                with st.spinner("Analyzing data..."):
                    try:
                        response = agent.query(question)

                        st.markdown("#### Answer")
                        st.markdown(response['answer'])

                        if response.get('sources'):
                            st.markdown("**Sources:**")
                            for source in response['sources']:
                                st.caption(f"- {source}")
                    except Exception as e:
                        st.error(f"Query failed: {e}")
            else:
                st.warning("Please enter a question")

# ===== TAB 3: SEARCH =====
with tab3:
    st.markdown("### Search & Explore Reviews")
    st.markdown("*Search through analyzed reviews with filters*")

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

    # Cross-validation notice
    st.info("""
    **Cross-Validation Results**: The table below shows fair, leakage-free evaluation where no prompt
    was tested against its own labels. Labels were generated by v2, v3, v6 ensemble, then all prompts
    were tested against these labels.

    **True Winner**: v3.0 (3-Shot Learning) with 91.95% Theme F1
    """)

    import json

    eval_dir = Path(__file__).parent.parent / 'data' / 'evaluation'
    result_files = list(eval_dir.glob('experiment_grid_*.json')) + list(eval_dir.glob('experiment_*.json')) + list(eval_dir.glob('experiment_crossval*.json'))

    # Prioritize cross-validation results
    crossval_files = [f for f in result_files if 'crossval' in f.name]
    if crossval_files:
        result_files = crossval_files + [f for f in result_files if f not in crossval_files]

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

            # Convert to DataFrame
            df_results = []
            for r in results:
                metrics = r.get('metrics', {})
                df_results.append({
                    'prompt': r.get('prompt_name', 'Unknown'),
                    'version': r.get('prompt_version', 'v?'),
                    'k_shot': r.get('k_shot', 0),
                    'model': r.get('model', 'unknown'),
                    'sentiment_mae': metrics.get('sentiment_mae', 0),
                    'sentiment_acc': metrics.get('sentiment_exact_accuracy', 0),
                    'theme_f1': metrics.get('theme_f1', 0),
                    'theme_precision': metrics.get('theme_precision', 0),
                    'theme_recall': metrics.get('theme_recall', 0),
                    'risk_acc': metrics.get('risk_accuracy', 0),
                    'cost': metrics.get('cost_per_1k_samples', 0)
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

                col1, col2, col3, col4 = st.columns(4)

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

                with col4:
                    best_idx = df['cost'].idxmin()
                    st.metric(
                        "Lowest Cost",
                        f"${df.loc[best_idx, 'cost']:.3f}/1k",
                        help=f"{df.loc[best_idx, 'prompt']} (k={df.loc[best_idx, 'k_shot']})"
                    )

                st.markdown("---")

                # Chart selection
                chart_type = st.radio(
                    "Select view:",
                    ["Leaderboard", "Performance vs Cost", "Theme Metrics", "Cost Analysis"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                if chart_type == "Leaderboard":
                    st.markdown("#### Experiment Leaderboard")

                    df_sorted = df.sort_values('theme_f1', ascending=False).reset_index(drop=True)
                    display_df = df_sorted[['prompt', 'k_shot', 'sentiment_acc', 'theme_f1', 'risk_acc', 'cost']].copy()
                    display_df.columns = ['Prompt', 'K-shot', 'Sentiment Acc', 'Theme F1', 'Risk Acc', 'Cost/1k']
                    display_df['Sentiment Acc'] = (display_df['Sentiment Acc'] * 100).round(1).astype(str) + '%'
                    display_df['Theme F1'] = (display_df['Theme F1'] * 100).round(1).astype(str) + '%'
                    display_df['Risk Acc'] = (display_df['Risk Acc'] * 100).round(1).astype(str) + '%'
                    display_df['Cost/1k'] = '$' + display_df['Cost/1k'].round(3).astype(str)

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                elif chart_type == "Performance vs Cost":
                    st.markdown("#### Performance vs Cost Trade-off")

                    fig = px.scatter(
                        df,
                        x='cost',
                        y='theme_f1',
                        size='sentiment_acc',
                        color='prompt',
                        hover_data=['k_shot'],
                        labels={
                            'cost': 'Cost per 1k Samples ($)',
                            'theme_f1': 'Theme F1 Score',
                            'sentiment_acc': 'Sentiment Accuracy',
                            'prompt': 'Prompt'
                        },
                        title='Theme F1 vs Cost (bubble size = Sentiment Accuracy)'
                    )

                    fig.update_layout(
                        height=500,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color="#374151")
                    )

                    st.plotly_chart(fig, use_container_width=True)

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
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color="#374151")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Cost Analysis":
                    st.markdown("#### Cost per Sample by Experiment")

                    df_cost = df.sort_values('cost').reset_index(drop=True)

                    fig = px.bar(
                        df_cost,
                        x='prompt',
                        y='cost',
                        color='k_shot',
                        labels={'prompt': 'Prompt Version', 'cost': 'Cost per 1k Samples ($)', 'k_shot': 'K-shot'},
                        title='Cost per 1k Samples (lower is better)',
                        text='cost'
                    )

                    fig.update_traces(texttemplate='$%{y:.3f}', textposition='outside')
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        height=500,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color="#374151")
                    )

                    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>
    ReviewInsight AI | Hybrid LLM + SQL Analytics Platform
</div>
""", unsafe_allow_html=True)
