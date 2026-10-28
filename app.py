import re
import streamlit as st
from agents import build_search_agent, build_read_agent, writer_chain, critic_chain

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1a2e 50%, #0d1117 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}

/* ── Pipeline step cards ── */
.pipeline-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin: 1.5rem 0 2rem;
    flex-wrap: wrap;
}
.pipeline-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s ease;
    min-width: 130px;
    justify-content: center;
}
.pipeline-card.active {
    border-color: #38bdf8;
    color: #38bdf8;
    background: rgba(56,189,248,0.08);
    box-shadow: 0 0 20px rgba(56,189,248,0.15);
}
.pipeline-card.done {
    border-color: #34d399;
    color: #34d399;
    background: rgba(52,211,153,0.08);
}
.pipeline-card.error {
    border-color: #f87171;
    color: #f87171;
    background: rgba(248,113,113,0.08);
}

/* ── Search bar styling ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 0.85rem 1.2rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* ── Result card ── */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.metric-card .score {
    font-size: 1.8rem;
    font-weight: 700;
}
.score-high  { color: #34d399; }
.score-mid   { color: #fbbf24; }
.score-low   { color: #f87171; }

/* ── Markdown text color ── */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #cbd5e1 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #e2e8f0 !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(52,211,153,0.1) !important;
    color: #34d399 !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* ── Error / success alerts ── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_scores(critic_text: str) -> dict:
    """Pull X/10 scores from critic markdown output."""
    patterns = {
        "Accuracy":       r"accuracy[^\d]*(\d+)/10",
        "Depth":          r"depth[^\d]*(\d+)/10",
        "Clarity":        r"clarity[^\d]*(\d+)/10",
        "Structure":      r"structure[^\d]*(\d+)/10",
        "Source Quality": r"source[^\d]*(\d+)/10",
        "Overall":        r"overall[^\d]*(\d+)/10",
    }
    scores = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, critic_text, re.IGNORECASE)
        if match:
            scores[name] = int(match.group(1))
    return scores


def score_color(score: int) -> str:
    if score >= 8:
        return "score-high"
    elif score >= 6:
        return "score-mid"
    return "score-low"


def render_pipeline_steps(active: int = -1, error: int = -1):
    steps = ["🔍 Search", "📖 Scrape", "✍️ Write", "🧐 Critique"]
    cards = ""
    for i, label in enumerate(steps):
        if i < active:
            cls = "done"
        elif i == active:
            cls = "active"
        elif i == error:
            cls = "error"
        else:
            cls = ""
        cards += f'<div class="pipeline-card {cls}">{label}</div>'
    st.markdown(f'<div class="pipeline-row">{cards}</div>', unsafe_allow_html=True)


def render_score_cards(scores: dict):
    if not scores:
        return
    cards_html = '<div class="metric-grid">'
    for name, score in scores.items():
        css = score_color(score)
        cards_html += (
            f'<div class="metric-card">'
            f'<div class="label">{name}</div>'
            f'<div class="score {css}">{score}<span style="font-size:1rem;color:#64748b">/10</span></div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔎 Multi-Agent Research Assistant</h1>
</div>
""", unsafe_allow_html=True)

# ── Search input ──────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    topic = st.text_input(
        label="topic",
        label_visibility="collapsed",
        placeholder="🔎  Enter a research topic  (e.g. Latest developments in solid-state batteries)",
        key="topic_input",
    )
with col_btn:
    run_button = st.button(
        "Run Research",
        type="primary",
        disabled=not topic.strip(),
        use_container_width=True,
    )


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_button:
    state = {}
    error_step = -1

    try:
        # Step 0 — Search
        with st.spinner("🔍 Search agent is finding information..."):
            search_agent = build_search_agent()
            search_result = search_agent.invoke({
                "messages": [("user", f"find the recent information on {topic}")]
            })
            state["search_result"] = search_result["messages"][-1].content

        # Step 1 — Scrape
        with st.spinner("📖 Read agent is scraping sources..."):
            read_agent = build_read_agent()
            read_result = read_agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about `{topic}`, "
                    f"pick the most recent URLs and scrape deeper content.\n\n"
                    f"Search result:\n{state['search_result'][:1500]}"
                )]
            })
            state["scraped_content"] = read_result["messages"][-1].content

        # Step 2 — Write
        with st.spinner("✍️ Writer is drafting the report..."):
            combined = (
                f"SEARCH RESULT:\n{state['search_result']}"
                f"\n\nSCRAPED URL DATA:\n{state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": combined,
            })

        # Step 3 — Critique
        with st.spinner("🧐 Critic is reviewing the report..."):
            state["critic_result"] = critic_chain.invoke({
                "topic": topic,
                "report": state["report"],
            })

        st.session_state["last_result"] = state
        st.session_state["last_topic"] = topic
        st.success("✅ Research pipeline complete!", icon="🎉")

    except Exception as e:
        st.error(f"**Pipeline failed:** {e}", icon="🚨")

# ── Display results ───────────────────────────────────────────────────────────
if "last_result" in st.session_state:
    state = st.session_state["last_result"]
    saved_topic = st.session_state.get("last_topic", "Research")

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    tab_report, tab_critic = st.tabs([
        "📄 Final Report",
        "🧐 Critic Feedback"
    ])

    # ── Tab 1: Report ──
    with tab_report:
        st.markdown(state["report"])
        st.divider()
        report_bytes = state["report"].encode("utf-8")
        st.download_button(
            label="⬇️ Download Report as .md",
            data=report_bytes,
            file_name=f"{saved_topic[:40].replace(' ', '_')}_report.md",
            mime="text/markdown",
        )

    # ── Tab 2: Critic ──
    with tab_critic:
        scores = extract_scores(state["critic_result"])
        if scores:
            st.markdown("#### 📊 Scores at a Glance")
            render_score_cards(scores)
            st.divider()
        st.markdown(state["critic_result"])

    # ── Tab 3: Raw ──
 

    # ── Reset button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start New Research", type="secondary"):
        for key in ["last_result", "last_topic"]:
            st.session_state.pop(key, None)
        st.rerun()