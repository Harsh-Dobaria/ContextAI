from fastapi import FastAPI

app = FastAPI(
    title="KnowledgeHub AI",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "KnowledgeHub AI Backend Running 🚀"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
    