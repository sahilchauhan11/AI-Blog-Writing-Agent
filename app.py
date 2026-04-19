import streamlit as st
import os
import base64
from datetime import date
from pathlib import Path
import re
import json
import uuid

# Import the graph directly
from graph import graph

# History is completely memory-only and resets on refresh per user request.

st.set_page_config(

    page_title="Professional AI Blog Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .hero-section {
        padding: 2.5rem 0;
        text-align: center;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .post-container {
        background-color: white;
        padding: 3rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        color: #1f2937;
        line-height: 1.7;
    }
    .post-container h1, .post-container h2, .post-container h3 {
        color: #111827;
        margin-top: 1.5em;
        margin-bottom: 0.75em;
    }
    .post-container img {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: block;
        margin: 2rem auto;
        max-width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Add title and description in the UI
st.markdown("""
<div class="hero-section">
    <div class="hero-title">AI Blog Post Generator</div>
    <div class="hero-subtitle">High-quality, well-researched technical blog posts generated in minutes using LangGraph and Gemini.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("Configure the parameters for your blog generation.")
    
    as_of_date = st.date_input("As of Date", value=date.today(), help="The reference date for temporal web search queries.")
    st.markdown("---")
    
    if st.button("✨ Reset Session", use_container_width=True):
        st.session_state.current_post = None
        st.session_state.current_topic = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Tech Stack:**
    - 🧠 `gemini-2.5-flash-lite`
    - 🔍 `TavilySearch`
    - 🕸️ `LangGraph`
    """)

st.markdown("### 📝 Generation Details")
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Blog Post Topic", placeholder="Enter your topic here, e.g., 'The state of AI Agents in 2024'", label_visibility="collapsed")
with col2:
    generate = st.button("Generate Blog Post", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

if "current_post" not in st.session_state:
    st.session_state.current_post = None
    st.session_state.current_topic = None

def extract_final(obj):
    """Recursively search for the 'final' key in the streamed state chunks."""
    if isinstance(obj, dict):
        if "final" in obj:
            return obj["final"]
        for k, v in obj.items():
            res = extract_final(v)
            if res: return res
    return None

if generate:
    if not topic:
        st.error("Please enter a topic.")
    else:
        active_thread_id = str(uuid.uuid4())
        # Initialize graph arguments
        inputs = {
            "topic": topic,
            "as_of": as_of_date.isoformat()
        }
        config = {"configurable": {"thread_id": active_thread_id}}

        # Human-readable status mapping
        status_steps = {
            "router": "Routing request and deciding research needs...",
            "research": "Gathering evidence and resources via Web Search...",
            "orchestrator": "Orchestrating Section plans & setting goals...",
            "worker": "Writing content sections based on goals & evidence...",
            "reducer": "Refining and generating images for the blog post...",
            "merge_content": "Merging sections into a cohesive blog post...",
            "decide_images": "Deciding image placements...",
            "generate_and_place_images": "Generating and placing imagery within post..."
        }
        
        final_md = ""
        
        with st.status("Initializing process...", expanded=True) as status_box:
            try:
                # Stream the state updates from LangGraph
                for chunk in graph.stream(inputs, config=config):
                    # Update UI for each completed node
                    for node_name, state_update in chunk.items():
                        desc = status_steps.get(node_name, f"Completed process: {node_name.replace('_', ' ').title()}")
                        st.write(f"✅ {desc}")
                        
                        # Extract final markdown if available in this chunk
                        md_extracted = extract_final(chunk)
                        if md_extracted:
                            final_md = md_extracted

                status_box.update(label="Blog generation complete!", state="complete", expanded=False)
                
                if final_md:
                    st.session_state.current_post = final_md
                    st.session_state.current_topic = topic
                    
            except Exception as e:
                status_box.update(label="An error occurred", state="error", expanded=True)
                st.error(f"Error details: {str(e)}")

        if final_md == "":
            st.warning("No final markdown output was found. Please check logs.")

# Render memory
if st.session_state.current_post:
    st.markdown("---")
    st.markdown(f"### ✨ Your Blog Post: {st.session_state.current_topic}")
    
    rendered_md = st.session_state.current_post
    
    # Render Markdown nicely
    st.markdown(f'<div class="post-container">{rendered_md}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.download_button(
            label="📥 Download Markdown",
            data=rendered_md,
            file_name=f"blog_post_{str(st.session_state.current_topic)[:20].replace(' ', '_').lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )
