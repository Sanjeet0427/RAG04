import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: loading RAG pipeline...")
    app.state.retriever, app.state.chain = rag.build_rag()
    print("Ready.")
    yield


app = FastAPI(title="Stats RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    retriever = app.state.retriever
    chain = app.state.chain

    async def event_stream():
        try:
            docs = await asyncio.to_thread(retriever.invoke, req.question)
            yield {"event": "sources", "data": json.dumps(rag.source_info(docs))}

            payload = {"context": rag.format_context(docs), "question": req.question}
            async for token in chain.astream(payload):
                yield {"event": "token", "data": json.dumps({"t": token})}

            yield {"event": "done", "data": "[DONE]"}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_stream())