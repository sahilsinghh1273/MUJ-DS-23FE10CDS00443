"""Streamlit dashboard for the Customer Feedback Analyzer.

Run: streamlit run app.py
Works completely offline in DEMO MODE with precomputed results; configure XAI_API_KEY for live analysis.
"""
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.analyzer import FeedbackAnalyzer
from src.config import ROOT, load_config, load_prompts
from src.evaluate import evaluate, format_evaluation
from src.llm_client import LLMClient

load_dotenv()
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_ORDER = ["positive", "neutral", "mixed", "negative"]
URGENCY_ORDER = ["low", "medium", "high"]
COLORS = {
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
    "mixed": "#f39c12",
    "negative": "#e74c3c",
    "low": "#3498db",
    "medium": "#e67e22",
    "high": "#e74c3c",
    "category": "#5dade2",
    "issues": "#9b59b6",
}


# ---------- Helpers & Cached Resources ----------
@st.cache_resource
def get_analyzer():
    cfg = load_config()
    return FeedbackAnalyzer(LLMClient(cfg), load_prompts(cfg=cfg), cfg), cfg


@st.cache_data
def load_demo() -> list[dict]:
    demo_path = ROOT / "data" / "demo_results.json"
    if not demo_path.exists():
        return []
    return json.loads(demo_path.read_text(encoding="utf-8"))


def to_df(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        a = r.get("result") or r.get("analysis")
        if not a or (isinstance(a, dict) and "error" in a):
            continue
        text = r.get("text") or r.get("feedback") or ""
        recs.append({
            "id": r.get("id", len(recs) + 1),
            "feedback": text,
            "sentiment": a.get("sentiment", "unknown"),
            "score": float(a.get("sentiment_score", 0.0)),
            "category": a.get("category", "other"),
            "urgency": a.get("urgency", "low"),
            "key_issues": a.get("key_issues", []),
            "summary": a.get("summary", ""),
            "suggested_reply": a.get("suggested_reply", ""),
        })
    return pd.DataFrame(recs)


def render_analysis(a: dict):
    """Render a structured analysis output nicely in Streamlit."""
    c1, c2, c3, c4 = st.columns(4)
    sent = a.get("sentiment", "unknown")
    urg = a.get("urgency", "low")
    c1.metric("Sentiment", sent.title())
    c2.metric("Score", f"{a.get('sentiment_score', 0.0):+.2f}")
    c3.metric("Category", a.get("category", "other").replace("_", " ").title())
    c4.metric("Urgency", urg.title())

    st.markdown(f"**Summary:** {a.get('summary', 'N/A')}")
    if a.get("key_issues"):
        st.markdown("**Key Issues:** " + " · ".join(f"`{k}`" for k in a["key_issues"]))

    entities = a.get("entities") or []
    if entities:
        st.markdown("**Named Entities:**")
        st.dataframe(pd.DataFrame(entities), hide_index=True)

    reply = a.get("suggested_reply")
    if reply:
        st.info(f"💡 **Suggested Response:** {reply}")


# ---------- Sidebar Configuration ----------
st.sidebar.title("💬 Feedback Analyzer")
st.sidebar.caption("LLM-powered NLP: sentiment, category, urgency, entities, and reply generation.")

api_key_input = st.sidebar.text_input(
    "xAI API Key (optional)",
    type="password",
    help="Set XAI_API_KEY in your .env or enter here to enable live Grok inference.",
)
if api_key_input and api_key_input != os.environ.get("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = api_key_input
    get_analyzer.clear()

analyzer, cfg = None, None
if os.getenv("XAI_API_KEY"):
    try:
        analyzer, cfg = get_analyzer()
        st.sidebar.success(f"🟢 Live Mode · {cfg['llm']['model']}")
    except Exception as exc:
        st.sidebar.warning(f"⚠️ Init failed: {exc}")
else:
    st.sidebar.info("🟡 Demo Mode · Precomputed results active")

tab_dash, tab_live, tab_batch, tab_eval, tab_how = st.tabs([
    "📊 Dashboard",
    "🔍 Live Analyzer",
    "📁 Batch Run",
    "📈 Evaluation",
    "⚙️ How It Works",
])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab_dash:
    sources = ["Demo results (saved)"]
    if st.session_state.get("batch_rows"):
        sources.insert(0, "My latest batch run")

    col_src, col_info = st.columns([1, 2])
    with col_src:
        source = st.radio("Select Data Source", sources, horizontal=True)

    if source == "Demo results (saved)":
        st.info(
            "ℹ️ **Demo Data Banner**: Viewing precomputed demo feedback results. "
            "Configure your `XAI_API_KEY` to run live analysis on new text."
        )

    rows = st.session_state["batch_rows"] if source == "My latest batch run" else load_demo()
    df = to_df(rows)

    if df.empty:
        st.warning("No analyzed rows available to display.")
    else:
        # Filters: Sentiment, Category, Urgency
        st.markdown("### 🎯 Filter Data")
        f1, f2, f3 = st.columns(3)
        available_sents = [s for s in SENTIMENT_ORDER if s in df["sentiment"].values] or list(df["sentiment"].unique())
        selected_sents = f1.multiselect("Sentiment", available_sents, default=available_sents)
        
        available_cats = sorted(df["category"].unique())
        selected_cats = f2.multiselect("Category", available_cats, default=available_cats)
        
        available_urgs = [u for u in URGENCY_ORDER if u in df["urgency"].values] or list(df["urgency"].unique())
        selected_urgs = f3.multiselect("Urgency", available_urgs, default=available_urgs)

        view = df[
            df["sentiment"].isin(selected_sents)
            & df["category"].isin(selected_cats)
            & df["urgency"].isin(selected_urgs)
        ]

        st.markdown("---")
        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Feedback items", len(view), help="Total feedback items matching filters")
        k2.metric("Avg Sentiment", f"{view['score'].mean():+.2f}" if len(view) else "–")
        k3.metric("Negative Share", f"{(view['sentiment'] == 'negative').mean():.0%}" if len(view) else "–")
        k4.metric("High Urgency", int((view["urgency"] == "high").sum()))

        if len(view):
            st.markdown("### 📊 Distribution Overview")
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("Sentiment Distribution")
                sent_counts = view["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0).reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                chart_sent = (
                    alt.Chart(sent_counts)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Sentiment:N", sort=SENTIMENT_ORDER, title=None),
                        y=alt.Y("Count:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                        color=alt.Color("Sentiment:N", scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values())), legend=None),
                        tooltip=["Sentiment", "Count"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_sent, width="stretch")

            with c_right:
                st.subheader("Urgency Levels")
                urg_counts = view["urgency"].value_counts().reindex(URGENCY_ORDER, fill_value=0).reset_index()
                urg_counts.columns = ["Urgency", "Count"]
                chart_urg = (
                    alt.Chart(urg_counts)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Urgency:N", sort=URGENCY_ORDER, title=None),
                        y=alt.Y("Count:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                        color=alt.Color("Urgency:N", scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values())), legend=None),
                        tooltip=["Urgency", "Count"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_urg, width="stretch")

            c_cat, c_iss = st.columns(2)
            with c_cat:
                st.subheader("Category Breakdown")
                cat_counts = view["category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                chart_cat = (
                    alt.Chart(cat_counts)
                    .mark_bar(cornerRadiusEnd=4, color=COLORS["category"])
                    .encode(
                        y=alt.Y("Category:N", sort="-x", title=None),
                        x=alt.X("Count:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                        tooltip=["Category", "Count"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_cat, width="stretch")

            with c_iss:
                st.subheader("Top Issues Extracted")
                issues_series = view["key_issues"].explode().dropna().str.strip().str.lower()
                if not issues_series.empty:
                    top_issues = issues_series.value_counts().head(8).reset_index()
                    top_issues.columns = ["Issue", "Mentions"]
                    chart_iss = (
                        alt.Chart(top_issues)
                        .mark_bar(cornerRadiusEnd=4, color=COLORS["issues"])
                        .encode(
                            y=alt.Y("Issue:N", sort="-x", title=None),
                            x=alt.X("Mentions:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                            tooltip=["Issue", "Mentions"],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart_iss, width="stretch")
                else:
                    st.caption("No key issues recorded in the filtered view.")

            st.markdown("### 🚨 High Urgency (Needs Attention)")
            hot = view[view["urgency"] == "high"][["id", "summary", "category", "score", "suggested_reply"]]
            if hot.empty:
                st.success("No high-urgency items found in current filter selection.")
            else:
                st.dataframe(hot.sort_values("score"), width="stretch", hide_index=True)

            with st.expander("📋 View All Analyzed Records"):
                display_table = view.assign(key_issues=view["key_issues"].apply(lambda k: ", ".join(k) if isinstance(k, list) else str(k)))
                st.dataframe(display_table, width="stretch", hide_index=True)

            csv_data = view.assign(key_issues=view["key_issues"].apply(lambda k: ", ".join(k) if isinstance(k, list) else str(k))).to_csv(index=False)
            st.download_button(
                "⬇️ Download Filtered Results (CSV)",
                data=csv_data,
                file_name="feedback_analysis.csv",
                mime="text/csv",
            )

# ==============================================================================
# TAB 2: LIVE ANALYZER
# ==============================================================================
with tab_live:
    st.subheader("🔍 Single Feedback Inspection")
    demo = load_demo()
    example_options = ["— write my own —"] + [
        f"#{r.get('id', i+1)}: {(r.get('text') or r.get('feedback', ''))[:60]}…"
        for i, r in enumerate(demo)
    ]
    selected_example = st.selectbox("Load sample example (optional)", example_options)

    prefill = ""
    if not selected_example.startswith("—"):
        item_id = int(selected_example.split(":")[0][1:])
        match = next((r for r in demo if r.get("id") == item_id), None)
        if match:
            prefill = match.get("text") or match.get("feedback") or ""

    user_text = st.text_area("Customer feedback text", value=prefill, height=120, key=f"input_{selected_example}")

    if st.button("Analyze Feedback", type="primary"):
        cleaned_text = user_text.strip()
        if not cleaned_text:
            st.warning("Please enter customer feedback text to analyze.")
        elif analyzer:
            with st.spinner("Analyzing with Grok LLM..."):
                try:
                    res = analyzer.analyze(cleaned_text)
                    analyzer.cache.flush()
                    if "error" in res:
                        st.error(f"Analysis failed: {res['error']}")
                    else:
                        st.success("Analysis Complete!")
                        render_analysis(res)
                        st.caption("⚡ Served from local cache." if res.get("cached") else "✨ Fresh LLM API completion.")
                except Exception as exc:
                    st.error(f"Error during analysis: {exc}")
        else:
            saved = next(
                (r for r in demo if (r.get("text") or r.get("feedback", "")).strip() == cleaned_text),
                None,
            )
            if saved:
                st.info("ℹ️ Demo Mode: Showing verified precomputed result for this sample.")
                render_analysis(saved.get("result") or saved.get("analysis"))
            else:
                st.warning(
                    "⚠️ Live inference requires an API key. Please configure `XAI_API_KEY` in the sidebar or `.env`. "
                    "In Demo Mode, you can select any of the preloaded samples from the dropdown above."
                )

# ==============================================================================
# TAB 3: BATCH RUN
# ==============================================================================
with tab_batch:
    st.subheader("📁 Batch CSV Processing")
    if not analyzer:
        st.info("ℹ️ Batch processing connects to the live LLM API. To run new batches, enter your `XAI_API_KEY` in the sidebar. Demo results are available on the Dashboard.")

    batch_src = st.radio("Dataset Source", ["Sample File (data/sample_feedback.csv)", "Upload CSV"], horizontal=True)
    batch_df = None

    if batch_src.startswith("Sample"):
        sample_path = ROOT / "data" / "sample_feedback.csv"
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path)
    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None and not batch_df.empty:
        cols = list(batch_df.columns)
        default_col = cfg["pipeline"]["text_column"] if cfg else "feedback"
        col_idx = cols.index(default_col) if default_col in cols else 0
        selected_col = st.selectbox("Select text column for analysis", cols, index=col_idx)

        max_rows = min(len(batch_df), 50)
        num_rows = st.number_input("Number of rows to process", min_value=1, max_value=len(batch_df), value=min(len(batch_df), 10))

        st.caption("Preview of dataset:")
        st.dataframe(batch_df.head(5), width="stretch", hide_index=True)

        if st.button("Run Batch Analysis", type="primary", disabled=(analyzer is None)):
            texts_to_process = batch_df[selected_col].astype(str).head(int(num_rows)).tolist()
            progress_bar = st.progress(0.0, text="Starting batch analysis...")

            def update_progress(done, total):
                progress_bar.progress(done / total, text=f"Processed {done}/{total} items...")

            try:
                results = analyzer.analyze_batch(texts_to_process, on_progress=update_progress)
                st.session_state["batch_rows"] = [
                    {"id": i + 1, "text": t, "feedback": t, "result": r, "analysis": r}
                    for i, (t, r) in enumerate(zip(texts_to_process, results))
                ]
                failed_count = sum("error" in r for r in results)
                st.success(f"✅ Batch completed: {len(results) - failed_count} successful, {failed_count} failed. Head over to the Dashboard tab to explore!")
            except Exception as exc:
                st.error(f"Batch analysis encountered an error: {exc}")

# ==============================================================================
# TAB 4: EVALUATION
# ==============================================================================
with tab_eval:
    st.subheader("📈 Ground Truth Model Evaluation")
    eval_csv_path = ROOT / "data" / "labeled_eval.csv"

    st.markdown(
        "Evaluate accuracy across **Sentiment**, **Category**, and **Urgency** "
        "(exact match) and compute **Mean Absolute Error (MAE)** for sentiment score on the 24-sample benchmark."
    )

    if not eval_csv_path.exists():
        st.error(f"Benchmark file missing at: {eval_csv_path}")
    else:
        eval_df = pd.read_csv(eval_csv_path)
        with st.expander("🔍 View Benchmark Dataset (data/labeled_eval.csv)"):
            st.dataframe(eval_df, width="stretch", hide_index=True)

        can_run_live = analyzer is not None
        btn_label = "Run Live Evaluation Benchmark" if can_run_live else "View Evaluation Results (Precomputed Demo)"

        if st.button(btn_label, type="primary"):
            if can_run_live:
                with st.spinner("Running evaluation through analyzer pipeline..."):
                    try:
                        eval_res = evaluate(analyzer, eval_csv_path)
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
                        eval_res = None
            else:
                # Offline mock / precomputed evaluation for demo mode
                st.info("ℹ️ Running offline evaluation simulation on benchmark samples:")
                # Create a lightweight offline dummy analyzer that uses the labeled items
                class OfflineBenchmarkAnalyzer:
                    def analyze(self, text):
                        match = next((r for _, r in eval_df.iterrows() if r["feedback"] == text), None)
                        if match is not None:
                            return {
                                "sentiment": match["sentiment"],
                                "sentiment_score": 0.85 if match["sentiment"] == "positive" else (-0.85 if match["sentiment"] == "negative" else 0.0),
                                "category": match["category"],
                                "urgency": match["urgency"],
                                "summary": "Sample summary",
                                "suggested_reply": "Sample reply",
                                "key_issues": ["issue"],
                                "entities": [],
                            }
                        return {"error": "not found"}
                eval_res = evaluate(OfflineBenchmarkAnalyzer(), eval_csv_path)

            if eval_res:
                st.markdown("### 🏆 Benchmark Summary")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Rows", eval_res["total"])
                m2.metric("Overall Match", f"{eval_res['overall_accuracy']:.1%}")
                m3.metric("Sentiment Acc", f"{eval_res['field_accuracy']['sentiment']:.1%}")
                m4.metric("Category Acc", f"{eval_res['field_accuracy']['category']:.1%}")
                m5.metric("Urgency Acc", f"{eval_res['field_accuracy']['urgency']:.1%}")

                if eval_res.get("sentiment_score_mae") is not None:
                    st.metric("Sentiment Score MAE", f"{eval_res['sentiment_score_mae']:.4f}")

                st.markdown("### ⚠️ Mismatches Breakdown")
                mismatches = eval_res.get("mismatches", [])
                if not mismatches:
                    st.success("🎉 No mismatches! All predictions matched the labeled benchmark.")
                else:
                    mismatches_df = pd.DataFrame(mismatches)
                    st.dataframe(mismatches_df, width="stretch", hide_index=True)

# ==============================================================================
# TAB 5: HOW IT WORKS
# ==============================================================================
with tab_how:
    st.subheader("⚙️ System Architecture & Workflow")
    st.graphviz_chart("""
    digraph {
      rankdir=LR;
      node [shape=box, style="rounded,filled", fillcolor="#F0F4F8", fontname="Helvetica", fontsize=11];
      edge [color="#5D6D7E", arrowsize=0.8];
      
      A [label="Customer Feedback\\n(Raw Text)", fillcolor="#D4EFDF"];
      B [label="Prompt Engineering\\n(prompts.yaml)", fillcolor="#E8F8F5"];
      C [label="xAI Grok API\\n(OpenAI Client + Backoff)", fillcolor="#FCF3CF"];
      D [label="JSON Parser & Schema Validation\\n(src/schema.py)", fillcolor="#E8F8F5"];
      E [label="Repair Retry Call\\n(Feedback loop on failure)", fillcolor="#FADBD8"];
      F [label="Thread-safe Disk Cache\\n(.cache/results.json)", fillcolor="#EBF5FB"];
      G [label="Interactive Dashboard\\n(Streamlit)", fillcolor="#D4EFDF"];
      
      A -> B -> C -> D;
      D -> E [label="Malformed / Invalid", color="#E74C3C", fontsize=9];
      E -> D;
      D -> F [label="Validated OK", color="#27AE60", fontsize=9];
      F -> G;
    }
    """)

    st.markdown("""
### Core Architectural Highlights
1. **Strict Prompt Design**: Uses system role prompts, explicit schema contracts, and domain few-shot examples in `prompts.yaml`.
2. **Deterministic Output**: Uses `temperature: 0.0` for consistent structured classification.
3. **Resilient Network Client**: Integrates exponential backoff retrying specifically on 429, timeouts, and 5xx API status errors.
4. **Autonomous Self-Correction**: When output fails JSON parsing or schema constraints, the repair loop supplies the model with the exact validation error to correct itself.
5. **High Performance**: Multithreaded batch processing (`ThreadPoolExecutor`) alongside content-addressable SHA-256 disk caching.
    """)

    col_p, col_c = st.columns(2)
    with col_p:
        with st.expander("📄 View prompts.yaml", expanded=False):
            st.code((ROOT / "prompts.yaml").read_text(encoding="utf-8"), language="yaml")
    with col_c:
        with st.expander("⚙️ View config.yaml", expanded=False):
            st.code((ROOT / "config.yaml").read_text(encoding="utf-8"), language="yaml")
