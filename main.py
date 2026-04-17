"""
COMPLETE PDF CHAT API - LOCAL OLLAMA (No API Keys Needed!)
Upload PDFs → Chat → Summarize
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List
import ollama
import io
import pdfplumber
import re
from typing import Dict, Tuple

# ========================================
# FASTAPI APP SETUP
# ========================================
app = FastAPI(
    title="🆓 Local PDF Chat API",
    description="Upload PDFs and chat/summarize with LOCAL AI (Ollama llama3.2)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global PDF storage
_pdf_store: Dict[str, str] = {}

# ========================================
# DATA MODELS
# ========================================
class UploadResponse(BaseModel):
    uploaded: List[str]
    total_words: int
    status: str

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    audience: Literal["children", "students", "adults"] = "adults"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    word_count: int

class SummarizeRequest(BaseModel):
    audience: Literal["children", "students", "adults"] = "adults"

class SummarizeResponse(BaseModel):
    summary: str
    sources: List[str]

# ========================================
# PDF PROCESSING
# ========================================
def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n\n".join(page for page in pages if page.strip())
    except Exception as e:
        print(f"PDF extract error: {e}")
        return ""

# ========================================
# CONTEXT RETRIEVAL
# ========================================
def get_context() -> Tuple[str, List[str]]:
    """Get all PDF context or raise error."""
    if not _pdf_store:
        raise HTTPException(400, "No PDFs uploaded! Use /upload first.")
    
    sources = list(_pdf_store.keys())
    context = "\n\n==========\n\n".join(
        f"📄 FILE: {name}\n\n{text[:2000]}" 
        for name, text in _pdf_store.items()
    )
    return context[:8000], sources  # Limit context size

def find_relevant_context(full_context: str, question: str) -> str:
    """Simple keyword-based RAG."""
    keywords = re.findall(r'\b\w{4,}\b', question.lower())
    sentences = full_context.split('.')
    scored = []
    
    for sent in sentences:
        score = sum(1 for kw in keywords if kw in sent.lower())
        if score > 0:
            scored.append((score, sent))
    
    scored.sort(reverse=True)
    relevant = '. '.join(sent[1] for sent in scored[:5])
    return relevant or full_context[:3000]

# ========================================
# LOCAL AI CALLS (OLLAMA)
# ========================================
async def call_ollama(prompt: str, max_tokens: int = 800) -> str:
    """Call local Ollama model."""
    try:
        model = "llama3.2:1b"  # Your downloaded model
        
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.3,
                "num_predict": max_tokens,
                "top_p": 0.9
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        raise HTTPException(503, f"AI error: {str(e)}. Check 'ollama serve' is running.")

# ========================================
# API ENDPOINTS
# ========================================
@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Upload and process PDF files."""
    if not files:
        raise HTTPException(400, "No files provided")
    
    uploaded, total_words = [], 0
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
            
        content = await file.read()
        text = extract_pdf_text(content)
        
        if text.strip():
            _pdf_store[file.filename] = text
            uploaded.append(file.filename)
            total_words += len(text.split())
    
    return UploadResponse(
        uploaded=uploaded,
        total_words=total_words,
        status=f"✅ {len(uploaded)} PDFs loaded ({total_words:,} words)"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdfs(request: ChatRequest):
    """Chat with uploaded PDFs using local AI."""
    context, sources = get_context()
    relevant = find_relevant_context(context, request.question)
    
    personas = {
        "children": "Explain simply for kids 6-12. Use easy words and fun examples.",
        "students": "Be clear and structured for students. Bold **key terms**.",
        "adults": "Professional, detailed answer with clear reasoning."
    }
    
    persona = personas.get(request.audience, personas["adults"])
    
    prompt = f"""{persona}

IMPORTANT: Answer ONLY using the PDF context below. If not in PDFs, say "Not found in documents."

CONTEXT (from your PDFs):
{relevant}

QUESTION: {request.question}

ANSWER:"""
    
    answer = await call_ollama(prompt)
    return ChatResponse(
        answer=answer,
        sources=sources,
        word_count=len(answer.split())
    )

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_pdfs(request: SummarizeRequest):
    """Summarize all uploaded PDFs."""
    context, sources = get_context()
    
    summary_styles = {
        "children": "Fun summary for kids. Use emojis! 4-6 bullet points.",
        "students": "Study notes format. Bold **key terms**. 6-8 bullets.",
        "adults": "Executive summary. Concise bullets. Key findings only."
    }
    
    style = summary_styles.get(request.audience, summary_styles["adults"])
    
    prompt = f"""{style}

SUMMARIZE ONLY the PDF content below:

{context}

SUMMARY:"""
    
    summary = await call_ollama(prompt, max_tokens=600)
    return SummarizeResponse(summary=summary, sources=sources)

@app.get("/health")
async def health_check():
    """Check if everything is working."""
    try:
        ollama.list()
        return {
            "status": "🚀 LIVE",
            "ollama": "✅ Connected",
            "model": "llama3.2:1b",
            "pdfs_loaded": len(_pdf_store),
            "endpoints": ["/upload", "/chat", "/summarize", "/clear"]
        }
    except:
        return {"status": "❌ Ollama not running", "fix": "Run 'ollama serve'"}

@app.delete("/clear")
async def clear_documents():
    """Clear all uploaded PDFs."""
    _pdf_store.clear()
    return {"message": "🗑️ All PDFs cleared"}

@app.get("/")
async def root():
    return {"message": "🆓 Local PDF Chat API running!", "docs": "/docs"}

# ========================================
# RUN SERVER
# ========================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Local PDF Chat API...")
    print("📱 Open: http://localhost:8000/docs")
    print("💡 Make sure 'ollama serve' is running!")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)