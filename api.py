from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from rag_core import initialize_rag_chain

# Global variables
rag_chain = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_chain
    try:
        rag_chain = initialize_rag_chain()
        print("RAG chain initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG chain: {e}")
    
    yield
    
    # Shutdown (cleanup if needed)
    print("Shutting down...")

app = FastAPI(title="Contoh RAG API", lifespan=lifespan)

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system is not initialized")
    
    try:
        response = rag_chain.invoke({"query": request.query})
        
        answer = response.get("result", "")
        sources = []

        # Check if answer is NOT_FOUND, fallback to base model
        if "NOT_FOUND" in answer:
            print("Answer not found in context, using base model...")
            from langchain_google_genai import GoogleGenerativeAI
            import os
            
            base_llm = GoogleGenerativeAI(
                model="models/gemini-2.5-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            answer = base_llm.invoke(request.query)
            # No sources for base model response
        else:
            # Extract sources if available
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    sources.append(source)
                sources = list(set(sources))
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Edubia RAG API is running. Send POST requests to /query."}
