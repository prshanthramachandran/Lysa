"""
Lysa — entry point.

Run with:
    python server.py
    # or
    uvicorn lysa.app:app --host 0.0.0.0 --port 8050 --reload
"""

from lysa.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8050, reload=True)
