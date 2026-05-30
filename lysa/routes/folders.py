"""
Folder and file-loading routes — load from directory, serve standalone viewer.
"""

import platform
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..models import FolderLoadParams
from ..image_processing import ingest_image
from .. import store

router = APIRouter(prefix="/api", tags=["folders"])


@router.get("/pick-directory")
async def pick_directory(prompt: str = "Select folder"):
    """
    Open a native OS folder chooser and return the selected path.
    Uses AppleScript on macOS, tkinter elsewhere as a fallback.
    Returns {"path": "..."} or {"path": null} if the user cancelled.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            # AppleScript gives a reliable POSIX path and a proper macOS dialog
            script = (
                f'POSIX path of (choose folder with prompt "{prompt}")'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                # User cancel returns non-zero; detect cancel specifically
                if "User canceled" in (result.stderr or "") or "-128" in (result.stderr or ""):
                    return {"path": None, "cancelled": True}
                raise HTTPException(500, f"osascript failed: {result.stderr.strip()}")
            path = result.stdout.strip().rstrip("/")
            return {"path": path or None}
        else:
            # tkinter fallback for Linux/Windows
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title=prompt)
            root.destroy()
            return {"path": path or None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Folder picker failed: {e}")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}


@router.get("/pick-files")
async def pick_files(prompt: str = "Select images"):
    """
    Open a native OS file chooser (multiple selection) and return the chosen
    paths. macOS uses AppleScript; tkinter is the fallback elsewhere. Returns
    {"paths": [...]} (empty list if cancelled). LIF files are intentionally
    excluded — use the dedicated "Open LIF" flow for those.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            # Build a newline-joined list of POSIX paths from the selection.
            script = (
                'set theFiles to choose file with prompt "' + prompt + '" '
                'with multiple selections allowed\n'
                'set AppleScript\'s text item delimiters to linefeed\n'
                'set out to ""\n'
                'repeat with f in theFiles\n'
                '    set out to out & POSIX path of f & linefeed\n'
                'end repeat\n'
                'return out'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                if "-128" in (result.stderr or "") or "User canceled" in (result.stderr or ""):
                    return {"paths": [], "cancelled": True}
                raise HTTPException(500, f"osascript failed: {result.stderr.strip()}")
            paths = [p for p in result.stdout.splitlines() if p.strip()]
            return {"paths": paths}
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            paths = filedialog.askopenfilenames(title=prompt)
            root.destroy()
            return {"paths": list(paths)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"File picker failed: {e}")


@router.post("/load-files")
async def load_files(payload: dict):
    """Ingest an explicit list of local image file paths.

    Body: {"paths": ["/abs/file1.tif", ...]}. Skips LIF and unsupported
    extensions (mirrors load-folder); returns the same shape so the frontend
    can reuse its handling.
    """
    paths = payload.get("paths") or []
    results = []
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            results.append({"error": "not a file", "filename": fp.name})
            continue
        if fp.suffix.lower() not in ALLOWED_EXTENSIONS:
            results.append({"error": f"unsupported type {fp.suffix}", "filename": fp.name})
            continue
        try:
            result = ingest_image(str(fp), fp.name)
            store.put(result["image_id"], result["entry"])
            results.append({"image_id": result["image_id"], "metadata": result["metadata"]})
        except Exception as e:
            results.append({"error": str(e), "filename": fp.name})

    return {
        "loaded": sum(1 for r in results if "image_id" in r),
        "errors": sum(1 for r in results if "error" in r),
        "images": results,
    }


@router.post("/load-folder")
async def load_folder(params: FolderLoadParams):
    """Load all supported images from a local directory."""
    folder = Path(params.path)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a valid directory: {params.path}")

    files = sorted(folder.glob(params.pattern)) if params.pattern else sorted(folder.iterdir())
    results = []

    for fp in files:
        if fp.is_file() and fp.suffix.lower() in ALLOWED_EXTENSIONS:
            try:
                result = ingest_image(str(fp), fp.name)
                store.put(result["image_id"], result["entry"])
                results.append({"image_id": result["image_id"], "metadata": result["metadata"]})
            except Exception as e:
                results.append({"error": str(e), "filename": fp.name})

    return {
        "loaded": sum(1 for r in results if "image_id" in r),
        "errors": sum(1 for r in results if "error" in r),
        "images": results,
    }


# --- Standalone viewer (kept for right-click → open in new tab) ---

_viewer_router = APIRouter(tags=["viewer"])


@_viewer_router.get("/view/{image_id}")
async def view_image(image_id: str):
    """Serve a standalone zoom/pan viewer for one image."""
    entry = store.get(image_id)
    meta = entry["metadata"]
    name = entry["name"]

    return HTMLResponse(f"""<!DOCTYPE html>
<html><head>
<title>{name} — Lysa</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0e1a; display:flex; align-items:center; justify-content:center;
         height:100vh; overflow:hidden; font-family:'Jura',system-ui,sans-serif; }}
  img {{ max-width:100vw; max-height:100vh; object-fit:contain; cursor:grab; transition:transform .1s; }}
  img.dragging {{ cursor:grabbing; transition:none; }}
  .info {{ position:fixed; top:10px; left:10px; color:#8296aa; font-size:13px;
           background:rgba(10,14,26,.85); padding:8px 14px; border-radius:8px; border:1px solid #1e2a40; }}
  .info h3 {{ font-size:14px; color:#e4e8ee; margin-bottom:4px; }}
  .controls {{ position:fixed; bottom:14px; left:50%; transform:translateX(-50%); display:flex; gap:6px; }}
  .controls button {{ background:#131a2e; color:#ccc; border:1px solid #1e2a40; padding:6px 14px;
           border-radius:6px; cursor:pointer; font-size:13px; }}
  .controls button:hover {{ background:#1e2a40; color:#fff; }}
</style></head><body>
<img id="img" src="/api/images/{image_id}/raw" alt="{name}">
<div class="info">
  <h3>{name}</h3>
  {meta.get('width','')} x {meta.get('height','')} &middot; {meta.get('mode','')}
  {(' &middot; ' + str(meta.get('bit_depth','')) + '-bit') if meta.get('bit_depth',8) > 8 else ''}
</div>
<div class="controls">
  <button onclick="z(1.3)">Zoom +</button>
  <button onclick="z(1/1.3)">Zoom -</button>
  <button onclick="scale=1;tx=0;ty=0;apply()">Fit</button>
</div>
<script>
let scale=1,tx=0,ty=0,drag=false,sx,sy;
const img=document.getElementById('img');
function apply(){{ img.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{scale}})` }}
function z(f){{ scale=Math.min(Math.max(scale*f,.1),30); apply() }}
img.addEventListener('wheel',e=>{{ e.preventDefault(); z(e.deltaY<0?1.15:1/1.15) }});
img.addEventListener('mousedown',e=>{{ drag=true; sx=e.clientX-tx; sy=e.clientY-ty; img.classList.add('dragging') }});
window.addEventListener('mousemove',e=>{{ if(drag){{ tx=e.clientX-sx; ty=e.clientY-sy; apply() }} }});
window.addEventListener('mouseup',()=>{{ drag=false; img.classList.remove('dragging') }});
</script></body></html>""")


# Attach the viewer router (no /api prefix)
viewer_router = _viewer_router
