
import os
import warnings
from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)
from qdrant_client import QdrantClient
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_path = os.path.join(
    BASE_DIR,
    "document_loader",
    "STATS.pdf"
)

QDRANT_PATH = os.path.join(
    BASE_DIR,
    "qdrant_db"
)

COLLECTION = "rag_collection"

# True: rebuild old index once
# False: load existing index
REBUILD_INDEX = False


# --------------------------------------------------
# 2. Load NVIDIA API key
# --------------------------------------------------

load_dotenv(
    os.path.join(BASE_DIR, ".env"),
    override=True
)

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA API key not found in .env")

api_key = api_key.strip()

print("API key loaded:", bool(api_key))


# --------------------------------------------------
# 3. Embeddings
# --------------------------------------------------

print("Loading embedding models...")

embeddings = FastEmbedEmbeddings()

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)


# --------------------------------------------------
# 4. Qdrant Vector Database
# --------------------------------------------------

os.makedirs(QDRANT_PATH, exist_ok=True)

client = QdrantClient(
    path=QDRANT_PATH
)

index_exists = client.collection_exists(
    COLLECTION
)

# Delete old index if rebuilding
if REBUILD_INDEX and index_exists:

    print("Deleting old incorrect index...")

    client.delete_collection(
        collection_name=COLLECTION
    )

    index_exists = False

client.close()


# --------------------------------------------------
# 5. Load or Build Index
# --------------------------------------------------

if index_exists:

    print("Loading existing index...")

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        path=QDRANT_PATH,
        collection_name=COLLECTION,
    )

else:

    print("Building new index...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print("Loading PDF:", pdf_path)

    docs = PyMuPDFLoader(pdf_path).load()

    print("Loaded pages:", len(docs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )

    chunks = text_splitter.split_documents(docs)

    print("Created chunks:", len(chunks))

    print("Creating hybrid vector index...")

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        path=QDRANT_PATH,
        collection_name=COLLECTION,
    )

    print("Indexing complete!")


# --------------------------------------------------
# 6. NVIDIA LLM
# --------------------------------------------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)


# --------------------------------------------------
# 7. Retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 8. RAG Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor in Statistical Techniques. "
        "Answer accurately using ONLY the provided context. "
        "If the answer is not present in the context, "
        "say that you don't know.\n\n"
        "Context:\n{context}"
    ),
    (
        "human",
        "{question}"
    ),
])

rag_chain = prompt | llm | StrOutputParser()

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI()


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask_question(data: AskRequest):

    question = data.question

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Prepare context
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}] "
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    # Generate answer
    response = rag_chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "question": question,
        "answer": response
    }

# --------------------------------------------------
# 9. RAG Query Function
# --------------------------------------------------

def query_rag(question: str):

    print("\n" + "=" * 60)
    print("Question:", question)
    print("=" * 60)

    retrieved_docs = retriever.invoke(question)

    print(
        f"\nRetrieved {len(retrieved_docs)} chunks:"
    )

    for i, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        page = doc.metadata.get("page", 0)

        page_number = (
            page + 1
            if isinstance(page, int)
            else "?"
        )

        preview = (
            doc.page_content
            .strip()
            .replace("\n", " ")[:150]
        )

        print(
            f"[{i}] Page {page_number}: "
            f"{preview}..."
        )

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}] "
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    print("\nGenerating answer...")

    response = rag_chain.invoke({
        "context": context,
        "question": question,
    })

    print("\nAnswer:")
    print(response)

    print("=" * 60)

    return response


# --------------------------------------------------
# 10. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    query_rag("What is correlation?")

    while True:

        question = input(
            "\nAsk a question (or type exit): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            ""
        }:
            print("Chat ended.")
            break

        query_rag(question)
