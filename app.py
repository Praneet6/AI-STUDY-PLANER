"""
app.py — Streamlit UI for Smart Study Path Generator
=====================================================

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit plotly pandas
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

from astar import Topic, AStarStudyPlanner
from utils import (
    build_daily_schedule,
    compute_stats,
    difficulty_progression,
    export_plan_text,
    export_plan_csv,
    get_sample_topics,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Study Path Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #64748b; font-size: 1rem; margin-top: 0; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .topic-card {
        background: #fafafa; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    }
    .day-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0284c7; border-radius: 8px;
        padding: 0.75rem 1rem; margin-bottom: 0.75rem;
    }
    .algo-info {
        background: #fefce8; border: 1px solid #fde047;
        border-radius: 10px; padding: 1rem; margin: 1rem 0;
    }
    .priority-badge {
        background: #fbbf24; color: white;
        border-radius: 20px; padding: 2px 8px; font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialisation ────────────────────────────────────────────

def init_session_state():
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'plan_generated' not in st.session_state:
        st.session_state.plan_generated = False
    if 'result' not in st.session_state:
        st.session_state.result = None


init_session_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Constraints")

    hours_per_day = st.slider(
        "Available hours per day", min_value=1.0, max_value=12.0,
        value=4.0, step=0.5, format="%.1fh"
    )

    deadline_days = st.number_input(
        "Deadline (days)", min_value=1, max_value=365, value=14
    )

    st.markdown("---")
    st.markdown("## 🔬 Algorithm Info")
    st.markdown("""
    **A\\* Search Algorithm**

    `f(n) = g(n) + h(n)`

    - `g(n)` — Adjusted time spent so far  
      *(includes cognitive-load penalties)*
    - `h(n)` — Admissible heuristic  
      *(sum of remaining adjusted times)*
    - Guarantees **optimal ordering** ✓

    **Cost Adjustments**
    - Hard→Hard consecutive: +20%
    - Big difficulty jump: +10%
    - Priority topic: −10% discount
    """)

    st.markdown("---")
    if st.button("🗂 Load Sample Data", use_container_width=True):
        st.session_state.topics = get_sample_topics()
        st.session_state.plan_generated = False
        st.rerun()

    if st.button("🗑 Clear All Topics", use_container_width=True):
        st.session_state.topics = []
        st.session_state.plan_generated = False
        st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">🧠 Smart Study Path Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by A* Search Algorithm — finds the optimal study order for your topics</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📥 Add Topics", "📊 Study Plan"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### ➕ Add a New Topic")

    with st.form("add_topic_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            topic_name = st.text_input("Topic Name *", placeholder="e.g. Graph Algorithms")
            difficulty = st.selectbox(
                "Difficulty",
                options=["Easy", "Medium", "Hard"],
                index=1
            )
        with col2:
            base_time = st.number_input(
                "Estimated Time (hours)", min_value=0.5, max_value=40.0, value=3.0, step=0.5
            )
            familiarity = st.slider(
                "Your Familiarity (%)", min_value=0, max_value=100, value=20,
                help="0 = never seen it | 50 = somewhat know it | 100 = already mastered"
            )
        is_priority = st.checkbox("⭐ Mark as Priority Topic")
        submitted = st.form_submit_button("Add Topic", use_container_width=True, type="primary")

    if submitted:
        if not topic_name.strip():
            st.error("Please enter a topic name.")
        else:
            st.session_state.topics.append(
                Topic(
                    name=topic_name.strip(),
                    difficulty=difficulty,
                    base_time=base_time,
                    familiarity=familiarity,
                    is_priority=is_priority,
                )
            )
            st.session_state.plan_generated = False
            st.success(f"✅ Added: **{topic_name}**")

    # ── Topic List ────────────────────────────────────────────────────────

    if st.session_state.topics:
        st.markdown(f"### 📚 Topics ({len(st.session_state.topics)})")

        for i, t in enumerate(st.session_state.topics):
            cols = st.columns([4, 2, 2, 2, 1])
            with cols[0]:
                label = f"{'⭐ ' if t.is_priority else ''}{t.name}"
                st.markdown(f"**{label}**")
            with cols[1]:
                color = {'Easy': '🟢', 'Medium': '🟡', 'Hard': '🔴'}[t.difficulty]
                st.write(f"{color} {t.difficulty}")
            with cols[2]:
                st.write(f"⏱ {t.base_time}h base")
            with cols[3]:
                adj = t.adjusted_time
                st.write(f"📐 {adj:.1f}h adj | {t.familiarity}% known")
            with cols[4]:
                if st.button("🗑", key=f"del_{i}", help="Remove topic"):
                    st.session_state.topics.pop(i)
                    st.session_state.plan_generated = False
                    st.rerun()

        st.markdown("---")

        # ── Generate Button ────────────────────────────────────────────
        deadline_warning = ""
        total_adj = sum(t.adjusted_time for t in st.session_state.topics)
        available = hours_per_day * deadline_days
        if total_adj > available:
            deadline_warning = (
                f"⚠️ **Warning:** Plan needs ~{total_adj:.1f}h but deadline allows "
                f"{available:.0f}h ({hours_per_day}h/day × {deadline_days} days)."
            )

        if deadline_warning:
            st.warning(deadline_warning)

        if st.button(
            "🚀 Generate Optimal Study Plan (A*)",
            use_container_width=True,
            type="primary",
            disabled=len(st.session_state.topics) == 0
        ):
            with st.spinner("Running A* search..."):
                start = time.perf_counter()
                planner = AStarStudyPlanner(st.session_state.topics)
                optimal_path, metadata = planner.solve()
                elapsed = time.perf_counter() - start

                daily = build_daily_schedule(
                    st.session_state.topics, optimal_path, hours_per_day
                )
                stats = compute_stats(st.session_state.topics, optimal_path)
                progression = difficulty_progression(st.session_state.topics, optimal_path)

                st.session_state.result = {
                    'optimal_path': optimal_path,
                    'metadata': metadata,
                    'daily': daily,
                    'stats': stats,
                    'progression': progression,
                    'elapsed_ms': round(elapsed * 1000, 2),
                }
                st.session_state.plan_generated = True

            st.success("✅ Optimal study path found! Switch to the **📊 Study Plan** tab.")
    else:
        st.info("👆 Add topics above or click **Load Sample Data** in the sidebar to get started.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    if not st.session_state.plan_generated or st.session_state.result is None:
        st.info("📥 Add topics and click **Generate** on the first tab to see your study plan.")
    else:
        res = st.session_state.result
        stats = res['stats']
        meta = res['metadata']

        # ── Algorithm Banner ──────────────────────────────────────────────

        st.markdown(f"""
        <div class="algo-info">
        <b>🔬 A* Search Complete</b> &nbsp;|&nbsp;
        Algorithm: <code>{meta.get('algorithm', 'A* exact')}</code> &nbsp;|&nbsp;
        Nodes expanded: <b>{meta.get('nodes_expanded', '—')}</b> &nbsp;|&nbsp;
        Runtime: <b>{res['elapsed_ms']} ms</b>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI Metrics ───────────────────────────────────────────────────

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📚 Topics", stats['topic_count'])
        c2.metric("⏱ Total Hours", f"{stats['total_adjusted_hours']}h")
        c3.metric("📅 Days Needed", len(res['daily']))
        c4.metric("💡 Avg Familiarity", f"{stats['avg_familiarity_pct']}%")
        c5.metric("⚡ Difficulty Easy/Med/Hard",
                  f"{stats['difficulty_distribution']['Easy']}/"
                  f"{stats['difficulty_distribution']['Medium']}/"
                  f"{stats['difficulty_distribution']['Hard']}")

        st.markdown("---")
        col_left, col_right = st.columns([1, 1])

        # ── Optimal Study Order ───────────────────────────────────────────

        with col_left:
            st.markdown("### 🏆 Optimal Study Order")
            st.caption("Topics ranked by A* — minimises cognitive load and total time")

            topics = st.session_state.topics
            for rank, idx in enumerate(res['optimal_path'], 1):
                t = topics[idx]
                diff_color = {'Easy': '#22c55e', 'Medium': '#f59e0b', 'Hard': '#ef4444'}[t.difficulty]
                priority_html = '<span style="background:#fbbf24;color:white;border-radius:20px;padding:2px 8px;font-size:0.75rem;margin-left:6px">⭐ Priority</span>' if t.is_priority else ''

                st.markdown(f"""
                <div class="topic-card">
                  <b>#{rank}</b> &nbsp; {t.name} {priority_html} <br>
                  <small>
                    <span style="color:{diff_color};font-weight:600">{t.difficulty}</span>
                    &nbsp;·&nbsp; ⏱ {t.adjusted_time:.1f}h adjusted
                    &nbsp;·&nbsp; 🧠 {t.familiarity}% familiar
                  </small>
                </div>
                """, unsafe_allow_html=True)

        # ── Day-Wise Schedule ─────────────────────────────────────────────

        with col_right:
            st.markdown("### 📅 Day-Wise Schedule")
            st.caption(f"Based on {hours_per_day}h/day availability")

            for day in res['daily']:
                topics_html = "".join(
                    f"<li><b>{ti['name']}</b> — {ti['difficulty']}, {ti['adjusted_time']:.1f}h</li>"
                    for ti in day['topics']
                )
                st.markdown(f"""
                <div class="day-card">
                  <b>📆 Day {day['day']}</b>
                  <span style="float:right;color:#0284c7;font-weight:600">{day['total_hours']:.1f}h</span><br>
                  <ul style="margin:4px 0 0 0;padding-left:18px;font-size:0.9rem">{topics_html}</ul>
                </div>
                """, unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────

        st.markdown("---")
        st.markdown("### 📈 Visualisations")

        chart1, chart2 = st.columns(2)

        with chart1:
            # Difficulty progression bar chart
            df_prog = pd.DataFrame(res['progression'])
            color_map = {'Easy': '#22c55e', 'Medium': '#f59e0b', 'Hard': '#ef4444'}
            fig_bar = px.bar(
                df_prog, x='topic', y='adjusted_time',
                color='difficulty',
                color_discrete_map=color_map,
                title='Topic Effort Breakdown',
                labels={'adjusted_time': 'Adjusted Hours', 'topic': 'Topic'},
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend_title='Difficulty',
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart2:
            # Cumulative time line
            df_prog = pd.DataFrame(res['progression'])
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_prog['topic'],
                y=df_prog['cumulative_hours'],
                mode='lines+markers+text',
                text=df_prog['cumulative_hours'].apply(lambda x: f"{x:.1f}h"),
                textposition='top center',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=8, color='#8b5cf6'),
                name='Cumulative'
            ))
            fig_line.update_layout(
                title='Cumulative Study Progress',
                xaxis_title='Topic (in study order)',
                yaxis_title='Hours Completed',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # Difficulty distribution pie
        diff_dist = stats['difficulty_distribution']
        fig_pie = go.Figure(go.Pie(
            labels=list(diff_dist.keys()),
            values=list(diff_dist.values()),
            marker_colors=['#22c55e', '#f59e0b', '#ef4444'],
            hole=0.4,
        ))
        fig_pie.update_layout(
            title='Difficulty Distribution',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        col_pie, col_info = st.columns([1, 1])
        with col_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_info:
            st.markdown("### 📋 A* Search Statistics")
            st.json({
                "Algorithm": meta.get("algorithm", "A* exact"),
                "Nodes Expanded": meta.get("nodes_expanded", "—"),
                "Nodes Generated": meta.get("nodes_generated", "—"),
                "Total Adjusted Hours": stats["total_adjusted_hours"],
                "Topics": stats["topic_count"],
                "Days Required": len(res["daily"]),
                "Search Time (ms)": res["elapsed_ms"],
            })

        # ── Export ────────────────────────────────────────────────────────

        st.markdown("---")
        st.markdown("### 💾 Export Plan")
        exp1, exp2 = st.columns(2)

        with exp1:
            text_plan = export_plan_text(
                st.session_state.topics,
                res['optimal_path'],
                res['daily'],
                stats,
                meta,
            )
            st.download_button(
                "📄 Download as Text",
                data=text_plan,
                file_name="study_plan.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with exp2:
            csv_plan = export_plan_csv(
                st.session_state.topics,
                res['optimal_path'],
                res['daily'],
            )
            st.download_button(
                "📊 Download as CSV",
                data=csv_plan,
                file_name="study_plan.csv",
                mime="text/csv",
                use_container_width=True,
            )
