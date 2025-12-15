"""
Streamlit App - IUE CourseCompass GUI.
======================================

Web interface for:
- Asking questions about IUE Engineering courses
- Viewing retrieved sources with citations
- Comparing departments
- Running evaluation harness

Run with: streamlit run src/iue_coursecompass/app/streamlit_app.py
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="IUE CourseCompass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports to speed up initial load
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def get_retriever():
    """Get cached retriever instance."""
    from iue_coursecompass.rag.retriever import Retriever
    return Retriever()


@st.cache_resource
def get_generator():
    """Get cached generator instance."""
    from iue_coursecompass.rag.generator import Generator
    return Generator()


@st.cache_resource
def get_grounding_checker():
    """Get cached grounding checker."""
    from iue_coursecompass.rag.grounding import GroundingChecker
    return GroundingChecker()


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_department" not in st.session_state:
        st.session_state.selected_department = "all"

    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    if "top_k" not in st.session_state:
        st.session_state.top_k = 5


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────


def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.title("🧭 CourseCompass")
        st.caption("IUE Engineering Course Assistant")

        st.divider()

        # Department filter
        st.subheader("🎯 Department Filter")
        department = st.selectbox(
            "Focus on department:",
            options=["all", "se", "ce", "eee", "ie"],
            format_func=lambda x: {
                "all": "All Departments",
                "se": "Software Engineering",
                "ce": "Computer Engineering",
                "eee": "Electrical & Electronics",
                "ie": "Industrial Engineering",
            }.get(x, x),
            key="dept_select",
        )
        st.session_state.selected_department = department

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")

        st.session_state.top_k = st.slider(
            "Number of sources to retrieve:",
            min_value=1,
            max_value=15,
            value=5,
            help="More sources = more context but slower",
        )

        st.session_state.show_sources = st.checkbox(
            "Show source citations",
            value=True,
            help="Display retrieved chunks used for the answer",
        )

        st.divider()

        # Quick actions
        st.subheader("🚀 Quick Actions")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        **IUE CourseCompass** helps you find information about
        Izmir University of Economics Faculty of Engineering courses.

        - Ask about course content, prerequisites, ECTS
        - Compare departments
        - Find curriculum requirements

        *Powered by RAG with Gemini*
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────────────────────────────────────


def render_chat():
    """Render main chat interface."""
    st.title("💬 Ask about IUE Engineering Courses")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if st.session_state.show_sources and message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])} citations)"):
                        for source in message["sources"]:
                            render_source_card(source)

    # Chat input
    if prompt := st.chat_input("Ask a question about IUE courses..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()

            with st.spinner("Searching course information..."):
                answer, sources = generate_response(prompt)

            response_placeholder.markdown(answer)

            # Show sources
            if st.session_state.show_sources and sources:
                with sources_placeholder.expander(f"📚 Sources ({len(sources)} citations)"):
                    for source in sources:
                        render_source_card(source)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


def render_source_card(source: dict):
    """Render a single source citation card."""
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**{source.get('course_code', 'Unknown')}** - {source.get('course_title', '')}")
            st.caption(f"Department: {source.get('department', '').upper()}")

        with col2:
            score = source.get("score", 0)
            st.metric("Relevance", f"{score:.2f}")

        # Show text snippet
        text = source.get("text", "")
        if len(text) > 300:
            text = text[:300] + "..."
        st.text(text)

        st.divider()


def generate_response(query: str) -> tuple[str, list[dict]]:
    """
    Generate response for user query.

    Args:
        query: User question

    Returns:
        Tuple of (answer_text, sources_list)
    """
    try:
        retriever = get_retriever()
        generator = get_generator()

        # Get department filter
        dept = st.session_state.selected_department
        if dept == "all":
            dept = None

        # Retrieve relevant chunks
        hits = retriever.retrieve(
            query=query,
            top_k=st.session_state.top_k,
            department=dept,
        )

        if not hits:
            return (
                "I couldn't find any relevant information about that in the indexed course data. "
                "Try rephrasing your question or selecting a different department.",
                [],
            )

        # Generate answer
        response = generator.generate(query=query, hits=hits)

        # Convert sources to dict for serialization
        sources = [
            {
                "chunk_id": h.chunk_id,
                "course_code": h.course_code,
                "course_title": h.course_title,
                "department": h.department,
                "text": h.text,
                "score": h.score,
            }
            for h in hits
        ]

        return response.answer, sources

    except Exception as e:
        return f"An error occurred: {str(e)}", []


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Mode
# ─────────────────────────────────────────────────────────────────────────────


def render_comparison_tab():
    """Render department comparison interface."""
    st.header("🔄 Compare Departments")

    st.markdown("""
    Compare course offerings, curricula, and requirements across departments.
    Select departments and enter a comparison query.
    """)

    col1, col2 = st.columns(2)

    with col1:
        dept1 = st.selectbox(
            "First Department",
            options=["se", "ce", "eee", "ie"],
            format_func=lambda x: {
                "se": "Software Engineering",
                "ce": "Computer Engineering",
                "eee": "Electrical & Electronics",
                "ie": "Industrial Engineering",
            }.get(x, x),
            key="compare_dept1",
        )

    with col2:
        dept2 = st.selectbox(
            "Second Department",
            options=["ce", "se", "eee", "ie"],
            format_func=lambda x: {
                "se": "Software Engineering",
                "ce": "Computer Engineering",
                "eee": "Electrical & Electronics",
                "ie": "Industrial Engineering",
            }.get(x, x),
            key="compare_dept2",
        )

    query = st.text_input(
        "Comparison query:",
        placeholder="e.g., Compare programming courses between departments",
        key="compare_query",
    )

    if st.button("🔍 Compare", type="primary", use_container_width=True):
        if query:
            with st.spinner("Analyzing departments..."):
                comparison = generate_comparison(query, dept1, dept2)
                st.markdown(comparison)
        else:
            st.warning("Please enter a comparison query.")


def generate_comparison(query: str, dept1: str, dept2: str) -> str:
    """Generate department comparison."""
    try:
        retriever = get_retriever()
        generator = get_generator()

        # Retrieve from both departments
        dept_hits = {}
        for dept in [dept1, dept2]:
            hits = retriever.retrieve(query=query, top_k=5, department=dept)
            dept_hits[dept] = hits

        # Generate comparison
        response = generator.generate_comparison(query=query, dept_hits=dept_hits)
        return response.answer

    except Exception as e:
        return f"Error generating comparison: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Tab
# ─────────────────────────────────────────────────────────────────────────────


def render_evaluation_tab():
    """Render evaluation harness interface."""
    st.header("📊 Evaluation Harness")

    st.markdown("""
    Run evaluation on a question bank to measure RAG system quality.
    Upload a JSON file with test questions or use sample questions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Question Bank")

        uploaded_file = st.file_uploader(
            "Upload questions (JSON)",
            type=["json"],
            help="JSON file with 'questions' array",
        )

        use_sample = st.checkbox("Use sample questions", value=True)

    with col2:
        st.subheader("⚙️ Evaluation Settings")

        eval_top_k = st.slider("Retrieval top-k", 1, 20, 10)
        skip_generation = st.checkbox(
            "Retrieval only (faster)",
            value=False,
            help="Skip answer generation, only evaluate retrieval",
        )

    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        run_evaluation_ui(
            uploaded_file=uploaded_file,
            use_sample=use_sample,
            top_k=eval_top_k,
            skip_generation=skip_generation,
        )


def run_evaluation_ui(uploaded_file, use_sample: bool, top_k: int, skip_generation: bool):
    """Run evaluation and display results."""
    try:
        from iue_coursecompass.evaluation import (
            QuestionBank,
            EvaluationRunner,
        )
        from iue_coursecompass.evaluation.questions import create_sample_questions

        # Load questions
        if uploaded_file:
            import json
            data = json.load(uploaded_file)
            questions = QuestionBank(
                [Question(**q) for q in data.get("questions", data)]
            )
        elif use_sample:
            questions = create_sample_questions()
        else:
            st.warning("Please upload a question file or use sample questions.")
            return

        st.info(f"Running evaluation on {len(questions)} questions...")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, question_id):
            progress_bar.progress(current / total)
            status_text.text(f"Evaluating: {question_id} ({current}/{total})")

        # Run evaluation
        runner = EvaluationRunner(
            top_k=top_k,
            skip_generation=skip_generation,
        )

        result = runner.evaluate(questions, progress_callback=progress_callback)

        progress_bar.progress(1.0)
        status_text.text("Evaluation complete!")

        # Display results
        st.success("✅ Evaluation Complete")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MRR", f"{result.retrieval_metrics.mrr:.3f}")

        with col2:
            st.metric("Recall@5", f"{result.retrieval_metrics.recall_at_5:.3f}")

        with col3:
            if not skip_generation:
                st.metric("Grounding Rate", f"{result.answer_metrics.grounding_rate:.3f}")
            else:
                st.metric("Hit Rate", f"{result.retrieval_metrics.hit_rate:.3f}")

        # Full report
        with st.expander("📄 Full Report"):
            st.text(result.summary())

        # Download results
        st.download_button(
            "📥 Download Results (JSON)",
            data=str(result.to_dict()),
            file_name="evaluation_results.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔄 Compare", "📊 Evaluate"])

    with tab1:
        render_chat()

    with tab2:
        render_comparison_tab()

    with tab3:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
