# 🔎 Multi-Agent Research Assistant

A fully autonomous, multi-agent research pipeline built with **LangChain**, **LangGraph**, and **Google Gemini**. Enter any topic and watch four specialized AI agents collaborate to search the web, scrape live content, write a structured report, and critically review it — all in real time via a **Streamlit** UI.

---

## 🧠 How It Works

The pipeline runs four agents sequentially, each with a dedicated role:

```
User Input (Topic)
       │
       ▼
┌─────────────────┐
│  1. Search Agent │  ← Uses Tavily to find recent news & URLs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Read Agent  │  ← Scrapes the most relevant URLs for deep content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Writer Chain│  ← Synthesizes findings into a structured report
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Critic Chain│  ← Reviews the report and scores it across 5 criteria
└─────────────────┘
```

### Agent Breakdown

| Agent | Model | Tools | Role |
|---|---|---|---|
| **Search Agent** | Gemini 2.5 Flash | `get_current_news` (Tavily) | Finds recent news articles and URLs on the topic |
| **Read Agent** | Gemini 2.5 Flash | `scrape_url` (BeautifulSoup) | Deep-scrapes selected URLs for full article content |
| **Writer Chain** | Gemini 2.5 Flash | — | Writes an Introduction → Key Findings → Conclusion → Sources report |
| **Critic Chain** | Gemini 2.5 Flash | — | Scores the report on Accuracy, Depth, Clarity, Structure, Source Quality |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/dnyanesh3005/DeepMind.git
cd DeepMind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root (this file is **gitignored** — never commit it):

```env
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key     # optional, for tracing
LANGCHAIN_TRACING_V2=true                    # optional
GROQ_API_KEY=your_groq_api_key               # optional
```

> Get your keys from:
> - **Google Gemini**: [aistudio.google.com](https://aistudio.google.com/app/apikey)
> - **Tavily**: [tavily.com](https://tavily.com/)
> - **LangSmith**: [smith.langchain.com](https://smith.langchain.com/settings)

### 4. Run the App

#### Option A — Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

#### Option B — Terminal / CLI

```bash
python pipeline.py
```

You'll be prompted to enter a topic and the full pipeline output will print to the console.

---

## 📁 Project Structure

```
Agent_multi/
├── app.py            # Streamlit web UI — the main entry point
├── agents.py         # Agent & chain definitions (Search, Read, Writer, Critic)
├── create_tool.py    # Custom LangChain tools (Tavily search, URL scraper)
├── pipeline.py       # CLI version of the research pipeline
├── requirements.txt  # Python dependencies
├── .env              # ⚠️ Your API keys — DO NOT commit this file
├── .gitignore        # Ensures .env is never pushed to GitHub
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [LangChain](https://www.langchain.com/) | Agent orchestration & chains |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | ReAct agent framework (`create_react_agent`) |
| [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) | LLM powering all agents |
| [Tavily](https://tavily.com/) | Real-time web search API |
| [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) | Web scraping & HTML parsing |
| [Streamlit](https://streamlit.io/) | Interactive web UI |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Secure API key management |

---

## 📸 Demo

Enter any topic (e.g., *"Latest developments in solid-state batteries"*) and the app will:

1. 🔍 **Search** the web for the latest news
2. 📖 **Scrape** the most relevant articles for deep content
3. ✍️ **Write** a professional, structured research report
4. 🧐 **Critique** the report with scores and actionable feedback

All steps are shown in real time via Streamlit's status panel, and intermediate agent outputs are available in an expandable section.

---

## ⚠️ Security Notice

This project uses API keys stored in a `.env` file. **Never commit your `.env` file to version control.**

- The `.gitignore` already excludes `.env`
- If you accidentally push secrets, rotate your keys immediately:
  - [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
  - [Tavily Dashboard](https://tavily.com/)
  - [LangSmith Settings](https://smith.langchain.com/settings)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> Built with ❤️ using LangChain + LangGraph + Google Gemini
