"""
Version / update-check route.

Exposes GET /api/version which reports the running version and, when
possible, whether a newer GitHub Release exists. Designed to be safe at
startup:

  - Fail-silent: any network/parse error leaves update_available = False and
    never raises. The app must start normally offline.
  - Cached: the GitHub API is queried at most once per CACHE_TTL (24h),
    in-process. The frontend calls this asynchronously after load, so even
    the first (uncached) call never blocks startup.
  - Opt-out: set LYSA_DISABLE_UPDATE_CHECK=1 to skip the network entirely.

The only network egress is a single GET to the public GitHub API, which
necessarily exposes the client IP to GitHub — documented for transparency.
"""

import os
import time
import json
import urllib.request

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .. import __version__

router = APIRouter(prefix="/api", tags=["version"])

GITHUB_REPO = "prshanthramachandran/Lysa"
RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CACHE_TTL = 24 * 3600          # seconds
FETCH_TIMEOUT = 3.0            # seconds — keep startup snappy

# In-process cache: {"ts": <epoch>, "data": {tag,url,name} or None}
_cache = {"ts": 0.0, "data": None}


def _parse_version(v: str):
    """Parse 'v3.1.2' / '3.1.2-beta' into a comparable tuple of ints.

    Leading 'v' is stripped; each dotted component contributes its leading
    integer (non-numeric suffixes like '-beta' are ignored for ordering).
    Unparseable parts become 0 so comparison never raises.
    """
    if not v:
        return ()
    v = v.strip().lstrip("vV")
    parts = []
    for comp in v.split("."):
        num = ""
        for ch in comp:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _is_newer(latest: str, current: str) -> bool:
    """True if version string `latest` orders strictly after `current`."""
    a = _parse_version(latest)
    b = _parse_version(current)
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return a > b


def _fetch_latest_release() -> dict:
    """Fetch the latest GitHub Release. Raises on any failure (caller guards)."""
    req = urllib.request.Request(
        RELEASES_URL,
        headers={
            "User-Agent": "Lysa-update-check",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return {
        "tag": payload.get("tag_name"),
        "url": payload.get("html_url"),
        "name": payload.get("name"),
    }


@router.get("/version")
def get_version():
    """Report current version and whether a newer GitHub Release exists.

    Response shape:
        {
          "current": "3.0.0",
          "latest": "3.1.0" | null,
          "update_available": bool,
          "release_url": str | null,
          "release_name": str | null,
          "checked": bool        # did we obtain remote release info?
        }
    """
    result = {
        "current": __version__,
        "latest": None,
        "update_available": False,
        "release_url": None,
        "release_name": None,
        "checked": False,
    }

    if os.environ.get("LYSA_DISABLE_UPDATE_CHECK"):
        return JSONResponse(result)

    now = time.time()
    info = _cache["data"]
    if info is None or (now - _cache["ts"]) >= CACHE_TTL:
        try:
            info = _fetch_latest_release()
            _cache["data"] = info
            _cache["ts"] = now
        except Exception:
            pass  # offline / 404 (no releases yet) / parse error → stay silent

    if info and info.get("tag"):
        result["checked"] = True
        result["latest"] = info["tag"]
        result["release_url"] = info.get("url")
        result["release_name"] = info.get("name")
        result["update_available"] = _is_newer(info["tag"], __version__)

    return JSONResponse(result)
