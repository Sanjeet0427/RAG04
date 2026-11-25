# Statistics RAG Project

A Retrieval-Augmented Generation (RAG) application that allows users to
ask questions about a statistics PDF. The project retrieves relevant
document passages from a local Qdrant vector database and uses an NVIDIA
Nemotron language model to generate answers based on the retrieved
context.

## Features

-   Load and extract text from PDF documents using PyMuPDF.
-   Split document text into overlapping chunks.
-   Store document chunks in a local Qdrant vector database.
-   Perform hybrid retrieval using dense embeddings and BM25 sparse
    embeddings.
-   Generate context-based answers with NVIDIA Nemotron.
-   Provide a FastAPI endpoint for submitting questions.
-   Support API testing through Swagger UI and a Python client.

## Project Structure

``` text
RAG Project/
├── document_loader/
│   └── STATS.pdf
├── qdrant_db/              # Local vector database
├── main.py                 # FastAPI application and RAG pipeline
├── rag.py                  # RAG-related code
├── api.py                  # API-related code
├── test_client.py           # Client for asking questions
├── test_nvidia.py           # NVIDIA model API test
├── .env                    # NVIDIA API key
├── requirements.txt
└── README.md
```

## Technologies Used

-   Python
-   FastAPI
-   Uvicorn
-   LangChain
-   PyMuPDF
-   FastEmbed
-   Qdrant
-   NVIDIA Nemotron

## Requirements

-   Python 3.11
-   NVIDIA Build API key with access to the selected hosted inference
    model
-   Statistics PDF file named `STATS.pdf`

## Installation

Open the project folder in VS Code and create a virtual environment:

``` powershell
python -m venv .venv
```

Activate the virtual environment:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

``` powershell
python -m pip install -r requirements.txt
```

## API Key Configuration

Create a `.env` file in the project root and add your NVIDIA API key:

``` env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Keep your API key private. Do not upload the `.env` file to GitHub.

## PDF Document

Place the statistics PDF in the following location:

``` text
document_loader/STATS.pdf
```

The application loads the PDF, extracts its text, and divides it into
smaller chunks for retrieval.

## Vector Database

The project uses Qdrant as a local vector database. The document chunks
are indexed using hybrid retrieval, which combines dense embeddings with
BM25 sparse embeddings.

The index is stored locally in the `qdrant_db` directory and can be
reused when the application is restarted.

## Running the Application

Make sure the virtual environment is activated and run the FastAPI
server from the project root:

``` powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The API will be available at:

``` text
http://127.0.0.1:8000
```

## API Documentation

Open the interactive FastAPI Swagger documentation in your browser:

``` text
http://127.0.0.1:8000/docs
```

## Asking Questions

### Using Swagger UI

1.  Open `http://127.0.0.1:8000/docs`.
2.  Expand the `POST /ask` endpoint.
3.  Click **Try it out**.
4.  Enter your question in the request body:

``` json
{
  "question": "What is correlation?"
}
```

5.  Click **Execute** to submit the question and view the response.

### Using the Python Client

Open a second terminal in the project folder, activate the virtual
environment, and run:

``` powershell
python test_client.py "What is correlation?"
```

You can replace the question inside the quotation marks with any
question related to the PDF.

## How It Works

1.  The PDF document is loaded and its text is extracted.
2.  The extracted text is split into smaller overlapping chunks.
3.  The chunks are converted into embeddings and stored in Qdrant.
4.  When a user asks a question, the retriever finds relevant document
    chunks.
5.  The retrieved context and question are sent to the NVIDIA Nemotron
    model.
6.  The model generates an answer using the retrieved context.
7.  The answer is returned through the FastAPI endpoint.

## Security

-   Store API keys in the `.env` file.
-   Keep `.env` out of public repositories.
-   Do not include API keys in source code or screenshots.
