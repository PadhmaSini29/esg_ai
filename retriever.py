import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # <-- ADD THIS LINE


from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found")

# Load ESG data
json_path = "C:/Users/lgspa/Downloads/esg_bot_starter_kit/esg_bot/data/esg_scores.json"
csv_path = "C:/Users/lgspa/Downloads/esg_bot_starter_kit/esg_bot/data/esg_scores.csv"

if not os.path.exists(json_path):
    raise FileNotFoundError(f"Missing file: {json_path}")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing file: {csv_path}")

with open(json_path, "r", encoding="utf-8") as f:
    esg_scores = json.load(f)

df = pd.read_csv(csv_path)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")


# Chroma vector DB
vectordb = Chroma(
    persist_directory="esg_bot/rag/vector_store",
    collection_name="esg_collection",
    embedding_function=embedding_model,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# QA prompt
question_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an ESG expert. Use the context below to answer the question.
If the answer is not present, say "I don't know."

Context:
---------
{context}

Question:
---------
{question}

Answer:
"""
)

combine_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
You are an ESG expert combining partial answers below into a single answer.

Question:
---------
{question}

Partial Answers:
---------
{summaries}

Answer:
"""
)

# LLM
llm_rag = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_rag,
    retriever=retriever,
    chain_type="map_reduce",
    chain_type_kwargs={
        "question_prompt": question_prompt,
        "combine_prompt": combine_prompt,
    }
)

# Python REPL Tool with chart handling
def safe_python_exec(code: str) -> str:
    code = code.strip().strip("`")
    if code.startswith("python"):
        code = code[len("python"):].strip()

    # Auto-replace common wrong column names
    replacements = {
        "ESG_Score": "score",
        "ESG score": "score",
        "Environment score": "environment",
        "Social score": "social",
        "Governance score": "governance"
    }
    for wrong, correct in replacements.items():
        code = code.replace(wrong, correct)

    try:
        exec_globals = {"df": df, "plt": plt}
        exec(code, exec_globals)

        if "plt" in code:
            plt.savefig("static/chart.png")
            plt.close()
            return "Chart generated successfully."
        else:
            return "Code executed successfully."
    except Exception as e:
        return f"Error during execution: {str(e)}"

python_tool = Tool(
    name="PythonREPL",
    func=safe_python_exec,
    description=(
        "Executes Python code on the ESG DataFrame 'df'. "
        "Use for analysis or matplotlib plots. "
        "Available columns: 'Company', 'environment', 'social', 'governance', 'score'."
    )
)

# ReAct Prompt for Plot Agent
react_prompt = PromptTemplate.from_template("""
You are a helpful ESG data assistant. The ESG DataFrame 'df' has these columns:
- Company
- environment
- social
- governance
- score (this is the overall ESG score)

You have access to the following tools:

{tools}

Use the following format:

Question: the user question
Thought: your reasoning
Action: the tool to use (one of [{tool_names}])
Action Input: the input to the tool
Observation: the result
... repeat if needed
Thought: I now know the final answer
Final Answer: the final answer

Begin!

Question: {input}
{agent_scratchpad}
""")


# Create Plotting Agent
plot_agent = create_react_agent(
    llm=llm_rag,
    tools=[python_tool],
    prompt=react_prompt
)

plot_executor = AgentExecutor(
    agent=plot_agent,
    tools=[python_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=100  # Limit retries
)

# Try JSON-based lookup
def try_get_score_from_json(query: str):
    query_lower = query.lower()
    if "score" in query_lower or "esg" in query_lower:
        for company in esg_scores:
            if company.lower() in query_lower:
                data = esg_scores[company]
                return (
                    f"## ESG Scores for **{company}**\n\n"
                    f"- **Environment:** {data['environment']}\n"
                    f"- **Social:** {data['social']}\n"
                    f"- **Governance:** {data['governance']}\n"
                    f"- **Overall Score:** {data['score']}"
                )
    return None

# Final answer function
def get_esg_answer(query: str) -> dict:
    """
    Handles ESG queries.
    Returns:
        {
            "text": final answer,
            "chart": True if chart was generated, else False
        }
    """
    if any(word in query.lower() for word in ["plot", "visualize", "chart", "graph"]):
        result = plot_executor.invoke({"input": query})
        return {
            "text": result.get("output", "Chart generated."),
            "chart": True
        }

    json_answer = try_get_score_from_json(query)
    if json_answer:
        return {"text": json_answer, "chart": False}

    answer = qa_chain.run(query)
    return {"text": answer, "chart": False}

# Expose core components
__all__ = ["df", "esg_scores", "qa_chain", "retriever", "get_esg_answer"]


