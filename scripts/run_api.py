"""
Start the FastAPI server for the Fraud Detection API.
Usage: python scripts/run_api.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.config import API_HOST, API_PORT


def main():
    print("Starting FraudShield AI API Server...")
    print(f"  Docs: http://localhost:{API_PORT}/docs")
    print(f"  Health: http://localhost:{API_PORT}/health")
    
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        workers=1
    )


if __name__ == "__main__":
    main()
