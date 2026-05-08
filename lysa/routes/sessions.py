"""
Session save / load routes.

A "session" is a JSON snapshot of the workspace: opaque client `state` blob
plus a list of image records pointing at on-disk source files. On reload we
re-ingest those files into the in-memory store under their original IDs so
references inside the client state (open tabs, ROIs, polylines, etc.) stay
valid.

Sessions are stored as `{SESSIONS_DIR}/{name}.json`. LIF sub-images are
re-registered through the LIF lazy loader so we don't pay a full decode at
session load time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import store
from .. import lif_handles

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SessionImageRecord(BaseModel):
    image_id: str


class SessionSaveParams(BaseModel):
    name: str
    state: dict                            # opaque client state blob
    images: List[SessionImageRecord]       # which IDs to capture from the store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_session_name(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not cleaned:
        raise HTTPException(400, "Session name must contain alphanumerics")
    return cleaned


def _capture_image_record(iid: str) -> Optional[dict]:
    """Build a session-side record for image `iid` if it lives in the store.

    For LIF entries we store enough info to call lif_handles.read_full_plane
    again at load time; for regular files we store the on-disk path.
    """
    if not store.contains(iid):
        return None
    e = store.get(iid)
    rec = {
        "image_id": iid,
        "name": e["name"],
        "metadata": e["metadata"],
        "source": e.get("source"),  # "lif" or None for regular files
    }
    if e.get("source") == "lif":
        rec["lif_path"] = e.get("lif_path")
        rec["lif_index"] = e.get("lif_index")
        rec["lif_z"] = e.get("lif_z")
        rec["lif_t"] = e.get("lif_t", 0)
        rec["lif_m"] = e.get("lif_m", 0)
    else:
        rec["file_path"] = e.get("path")
    return rec


def _restore_image(rec: dict) -> Optional[str]:
    """Re-register one image record under its original id. Returns error
    string on failure, None on success."""
    iid = rec.get("image_id")
    if not iid:
        return "missing image_id"
    if store.contains(iid):
        return None  # already restored

    if rec.get("source") == "lif":
        path = rec.get("lif_path")
        idx = rec.get("lif_index")
        if not path or idx is None:
            return "incomplete LIF record"
        if not Path(path).is_file():
            return f"LIF file missing: {path}"
        try:
            imgs = lif_handles.list_images(path)
            if idx < 0 or idx >= len(imgs):
                return f"LIF index {idx} out of range"
            lif_img = imgs[idx]
        except Exception as exc:  # noqa: BLE001
            return f"LIF open failed: {exc}"
        entry = {
            "path": f"lif://{path}#{idx}",
            "name": rec.get("name") or (lif_img.name or f"LIF_image_{idx}"),
            "metadata": rec.get("metadata") or {},
            "array": None,
            "display_array": None,
            "source": "lif",
            "lif_path": path,
            "lif_index": idx,
            "lif_z": rec.get("lif_z"),
            "lif_t": rec.get("lif_t", 0),
            "lif_m": rec.get("lif_m", 0),
        }
        store.put(iid, entry)
        return None

    # Regular file path
    file_path = rec.get("file_path")
    if not file_path:
        return "missing file_path"
    if not Path(file_path).is_file():
        return f"file missing: {file_path}"
    try:
        from ..image_processing import ingest_image
        result = ingest_image(file_path, rec.get("name") or Path(file_path).name)
    except Exception as exc:  # noqa: BLE001
        return f"ingest failed: {exc}"
    # Override the new uuid with the original id so client refs stay valid
    store.put(iid, result["entry"])
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/save")
async def save_session(params: SessionSaveParams):
    """Persist the client's session state under a sanitised name."""
    name = _safe_session_name(params.name)
    image_records = []
    for entry in params.images:
        rec = _capture_image_record(entry.image_id)
        if rec is not None:
            image_records.append(rec)

    payload = {
        "version": 1,
        "name": name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "state": params.state,
        "images": image_records,
    }
    out_path = SESSIONS_DIR / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return {
        "name": name,
        "saved_at": payload["saved_at"],
        "image_count": len(image_records),
    }


@router.get("/list")
async def list_sessions():
    """List saved sessions (newest first)."""
    out = []
    for f in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            out.append({
                "name": data.get("name", f.stem),
                "saved_at": data.get("saved_at", ""),
                "image_count": len(data.get("images", [])),
            })
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda s: s.get("saved_at", ""), reverse=True)
    return out


@router.get("/load/{name}")
async def load_session(name: str):
    """Re-ingest a saved session into the store and return the state blob."""
    name = _safe_session_name(name)
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Session '{name}' not found")
    data = json.loads(path.read_text())

    missing = []
    for img in data.get("images", []):
        err = _restore_image(img)
        if err is not None:
            missing.append({
                "image_id": img.get("image_id"),
                "name": img.get("name"),
                "error": err,
            })

    return {
        "name": data.get("name", name),
        "saved_at": data.get("saved_at", ""),
        "state": data.get("state", {}),
        "images": data.get("images", []),
        "missing": missing,
    }


@router.delete("/{name}")
async def delete_session(name: str):
    name = _safe_session_name(name)
    path = SESSIONS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
    return {"deleted": name}
