"""ITMS (Indigenous Contactless Integrated Track Monitoring System) - Backend Entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ITMS Backend API",
    description="Indigenous Contactless Integrated Track Monitoring System - Backend API",
    version="0.1.0",
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint confirming ITMS backend is running."""
    return {
        "status": "healthy",
        "service": "ITMS Backend",
        "version": "0.1.0",
    }
