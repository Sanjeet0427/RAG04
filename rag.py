import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = os.path.dirname(__file__)
PDF_PATH = os.path.join(BASE_DIR, "document_loader", "STATS.pdf")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_data")
COLLECTION = "stats_book"

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor in Statistical Techniques. "
        "Answer the question accurately based ONLY on the provided context. "
        "If the answer cannot be found in the context, say that you don't know.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def build_rag(k: int = 4):
    """Returns (retriever, chain). Builds the index on the first run only."""
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not found in .env")

    embeddings = FastEmbedEmbeddings()
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")

    client = QdrantClient(path=QDRANT_PATH)
    exists = client.collection_exists(COLLECTION)
    client.close()

    if exists:
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            path=QDRANT_PATH,
            collection_name=COLLECTION,
        )
    else:
        docs = PyMuPDFLoader(PDF_PATH).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=80
        ).split_documents(docs)
        vectorstore = QdrantVectorStore.from_documents(
            chunks,
            embedding=embeddings,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            path=QDRANT_PATH,
            collection_name=COLLECTION,
        )

    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",  # exact naam build.nvidia.com par check karo
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = PROMPT | llm | StrOutputParser()
    return retriever, chain


def format_context(docs) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs
    )


def source_info(docs) -> list[dict]:
    return [
        {
            "page": (d.metadata.get("page", 0) + 1),
            "preview": d.page_content.strip().replace("\n", " ")[:150],
        }
        for d in docs
    ]