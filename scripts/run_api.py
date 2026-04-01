import os
import sys
from pathlib import Path
import uvicorn

# 🔥 ADD THIS (CRITICAL FIX)
sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
