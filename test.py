from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vectordb = Chroma(
    collection_name="esg_collection",
    persist_directory="esg_bot/rag/vector_store.db",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Try retrieving something
results = vectordb.similarity_search("What are Wipro's ESG initiatives in 2023?", k=3)
for doc in results:
    print(doc.page_content)
