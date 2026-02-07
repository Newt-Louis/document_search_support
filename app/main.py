# app/main.py
from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.api.v1.routes_chat import router as chat_router
from app.api.v1.routes_ingest import router as ingest_router

def create_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="Syezain AI Agent",
        version="1.0.0",
    )

    @app.get("/")
    def health_check():
        return {"status": "running", "engine": "Ollama + Qdrant"}

    app.include_router(chat_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
