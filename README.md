# AI Blog Writing Agent ✍️

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Available-blue?style=for-the-badge&logo=streamlit)](https://ai-blog-writing-agent.streamlit.app/)

A highly professional, multi-agent AI blog post generator that conducts intelligent web research and autonomously drafts technical articles. Engineered dynamically without local database caching (stateless), allowing the application to be completely lightweight and instantly deployable to the cloud.

## 🚀 Live Application
**Try it yourself:** [https://ai-blog-writing-agent.streamlit.app/](https://ai-blog-writing-agent.streamlit.app/)

## ✨ Key Features

- **Multi-Agent Orchestration (LangGraph):** Employs LangGraph to orchestrate complex writing workflows processing seamlessly through routers, researchers, orchestrators, and writer agent nodes.
- **Dynamic Research Router:** AI decides automatically whether a topic needs real-time fetching (Open-Book logic using Tavily API), contextual examples (Hybrid), or foundational logic (Closed-Book/Evergreen). 
- **Automated AI Imagery:** Leverages `gemini-2.5-flash-image` to generate compelling blog illustrations natively converted to rich `base64` Data URIs and embedded directly inside the final Markdown output.
- **Precision Evidence Extraction:** Deduplicates, synthesizes, and cleanly links real-world references within generated documents as inline Markdown citations.
- **Customizable Reference Anchors:** Temporal Search parameters via "As of Date" capabilities handled through the Streamlit interface for strict-time sensitive reporting and weekly-roundups.
- **One-Click Markdown Download:** Delivers structured, clean Markdown directly to the user's browser for publishing.

## 🛠 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Agent Pipeline Framework:** [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://python.langchain.com/)
- **Language Models:** Multi-modal Gemini API (`gemini-2.5-flash-lite` for planning/writing, `gemini-2.5-flash-image` for visuals).
- **Research Engine:** [Tavily Search API](https://tavily.com/)

## 🔮 Future Scope

While the current architecture is explicitly designed to be stateless and reset upon refresh to maximize performance and portability, the following upgrades are planned:

- **Persistent Memory & SQLite Interfacing:** Integrating LangGraph's native Checkpointer schemas to remember active blog generation iterations across refreshes.
- **Session Threading / Multi-Chat:** Implementing dynamic UUID-based threaded routing to provide sidebars where users can infinitely scroll through historical chat patterns on custom profiles.

## 💻 Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd "bloggenerator website"
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure Environment Variables:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

4. Launch the application!
   ```bash
   streamlit run app.py
   ```
