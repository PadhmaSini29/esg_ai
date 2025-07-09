# 🌱 ESG Bot – AI-Powered ESG Insights with RAG

**ESG Bot** is a conversational AI assistant that answers environmental, social, and governance (ESG)-related queries using real company data. It combines the power of Retrieval-Augmented Generation (RAG), structured ESG datasets, and Groq’s LLaMA3 model to provide accurate, contextual responses. It can even generate visualizations on demand.

## 🔍 Features

- 📄 **Contextual Question Answering:** Uses vector-based retrieval from ESG reports and scores.
- 📊 **Data Visualization:** Automatically generates charts from ESG score data using Python and matplotlib.
- 🤖 **Multi-modal Reasoning:** Combines structured CSV/JSON data and document search.
- 🧠 **LLM Powered by Groq:** Fast, efficient inference using `llama3-8b-8192`.
- 📁 **PDF-Ready (Extensible):** Easily extend to support PDF company reports.
- 🧰 **Agent-based Reasoning:** Uses LangChain agents with embedded tools like Python REPL.

## 📁 Project Structure

```
esg_bot/
├── app.py                     # Flask app entry point
├── rag/
│   ├── retriever.py           # Core RAG + agent logic
│   └── vector_store/          # Persisted Chroma DB
├── data/
│   ├── esg_scores.csv         # Tabular ESG data (used by Python REPL)
│   └── esg_scores.json        # Structured scores (for quick lookup)
├── templates/
│   └── index.html             # Web UI for query input
├── requirements.txt           # All required dependencies
└── .env                       # API key for Groq (GROQ_API_KEY)
```

## ⚙️ Installation

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/esg_bot.git
cd esg_bot
```

2. **Create virtual environment**
```bash
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set your Groq API key in `.env`**
```
GROQ_API_KEY=your-groq-api-key-here
```

## 🚀 Running the App

```bash
python app.py
```

Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000) to start chatting with the bot.

## 🧠 How It Works

- ✅ **Retriever:** Uses `langchain_community.vectorstores.Chroma` to retrieve ESG-relevant chunks.
- 💬 **LLM:** `ChatGroq` with LLaMA3 8B responds with synthesized answers.
- 🔎 **Tool Use:** If a query needs computation or plotting, the agent invokes `PythonREPLTool` with access to the ESG DataFrame.
- 📈 **Agent Executor:** Handles code generation and parsing using LangChain agent logic.

## 📊 Sample Questions

- *What is the ESG score for Infosys?*
- *Compare the environmental scores of TCS and Wipro.*
- *Plot the ESG scores for all companies.*
- *How does HDFC rank on governance?*

## 🧪 Testing

To test the core pipeline:

```python
from rag.retriever import get_esg_answer
print(get_esg_answer("What is the ESG score for Infosys?"))
```

## 🛠 Dependencies

- `langchain`, `langchain-community`, `langchain-groq`
- `chromadb`, `sentence-transformers`
- `matplotlib`, `seaborn`, `pandas`
- `flask`, `python-dotenv`

Install them all with:

```bash
pip install -r requirements.txt
```

## 🚧 Known Issues

- 🔁 Some LLM output may include invalid Python syntax (backticks) — ensure correct parsing logic or prompt tuning.
- ⚠️ Chroma deprecated in LangChain >= 0.2.9 — consider switching to `langchain_chroma`.

## 🙌 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Groq](https://groq.com/) for blazing fast inference
- [Sentence Transformers](https://www.sbert.net/) for efficient embedding

## 📜 License

This project is licensed under the MIT License.
