"""
FastAPI entrypoint. Wires together the API Service router, sets up logging,
and exposes a health check.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging_config import setup_logging
from app.routers import extract

setup_logging()

app = FastAPI(
    title="Form Extractor API",
    description="Upload a PDF/Image of a form -> get structured JSON back.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
