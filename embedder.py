import os
import glob
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 500
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PDF_DIR = "esg_bot/data/reports"
VECTOR_DB_DIR = "esg_bot/rag/vector_store.db"
COLLECTION_NAME = "esg_collection"

# Load and split PDFs into text chunks
def load_and_split_pdfs(directory):
    documents = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    print(f"📁 Found {len(pdf_files)} PDF files:")

    for file in pdf_files:
        print(f"   → {os.path.basename(file)}")
        loader = PyPDFLoader(file)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(pages)
        documents.extend(chunks)

    return documents

# Build and persist vector store
def build_vectorstore():
    documents = load_and_split_pdfs(PDF_DIR)
    print(f"🧠 Total document chunks: {len(documents)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    print("📥 Inserting in batches...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        ids = [str(uuid.uuid4()) for _ in batch]
        vectordb.add_documents(batch, ids=ids)
        print(f"   → Added batch {i // BATCH_SIZE + 1} with {len(batch)} chunks")

    vectordb.persist()
    print(f"✅ Vectorstore saved to: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    build_vectorstore()
