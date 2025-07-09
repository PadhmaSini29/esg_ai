from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Same embedding model you used in embedder.py
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load your vector DB
vectordb = Chroma(
    persist_directory="esg_bot/rag/vector_store.db",
    collection_name="esg_collection",
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# Try retrieving for 'HCLTech'
results = retriever.get_relevant_documents("HCLTech governance")

for i, doc in enumerate(results):
    print(f"\nResult {i + 1}\n{'-'*20}\n{doc.page_content}\n")
