"""
FastAPI application factory — assembles routes and middleware.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .routes.images import router as images_router
from .routes.analysis import router as analysis_router
from .routes.merge import router as merge_router
from .routes.folders import router as folders_router, viewer_router
from .routes.lif import router as lif_router
from .routes.segmentation import router as segmentation_router
from .routes.filters import router as filters_router
from .routes.root_growth import router as root_growth_router
from .routes.tiles import router as tiles_router
from .routes.sessions import router as sessions_router


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(title="Lysa", version="1.0.0")

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ---
    app.include_router(images_router)
    app.include_router(analysis_router)
    app.include_router(merge_router)
    app.include_router(folders_router)
    app.include_router(lif_router)
    app.include_router(segmentation_router)
    app.include_router(filters_router)
    app.include_router(root_growth_router)
    app.include_router(tiles_router)
    app.include_router(sessions_router)
    app.include_router(viewer_router)       # /view/{id} — no /api prefix

    # --- Frontend ---
    static_dir = Path(__file__).resolve().parent.parent / "static"

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index_path = static_dir / "index.html"
        return HTMLResponse(content=index_path.read_text(), status_code=200)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
