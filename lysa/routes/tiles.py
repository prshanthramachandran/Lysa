"""
Tile server — serves DeepZoom tiles for OpenSeadragon.

URL layout:
    GET /api/tiles/{image_id}/info              → JSON info (dims, levels)
    GET /api/tiles/{image_id}/info.dzi          → DZI XML descriptor
    GET /api/tiles/{image_id}/{level}/{col}_{row}.png  → single tile
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
import io

from .. import store
from .. import pyramid

router = APIRouter(prefix="/api/tiles", tags=["tiles"])


@router.get("/{image_id}/info")
def tile_info(image_id: str):
    if not store.contains(image_id):
        raise HTTPException(404, "Image not found")
    return JSONResponse(pyramid.dzi_info(image_id))


@router.get("/{image_id}/info.dzi")
def tile_info_dzi(image_id: str):
    """Return the DeepZoom XML descriptor (what OpenSeadragon fetches)."""
    if not store.contains(image_id):
        raise HTTPException(404, "Image not found")
    info = pyramid.dzi_info(image_id)
    w = info["width"]
    h = info["height"]
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Image TileSize="{ts}" Overlap="{ov}" Format="png" '
        'xmlns="http://schemas.microsoft.com/deepzoom/2008">'
        '<Size Width="{w}" Height="{h}"/>'
        '</Image>'
    ).format(ts=pyramid.TILE_SIZE, ov=pyramid.TILE_OVERLAP, w=w, h=h)
    return Response(content=xml, media_type="application/xml")


@router.get("/{image_id}/{level}/{col}_{row}.png")
def tile(image_id: str, level: int, col: int, row: int):
    if not store.contains(image_id):
        raise HTTPException(404, "Image not found")
    try:
        data = pyramid.get_tile_png(image_id, level, col, row)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


# DeepZoom's default tile URL format uses `_files/{level}/{col}_{row}.png`.
# We register an alias so `info.dzi` + default OSD settings "just work".
@router.get("/{image_id}/info_files/{level}/{col}_{row}.png")
def tile_alias(image_id: str, level: int, col: int, row: int):
    return tile(image_id, level, col, row)


@router.post("/{image_id}/clear")
def clear_pyramid(image_id: str):
    """Drop cached pyramid levels for this image (called on tab close)."""
    pyramid.clear_image(image_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# v3: per-channel tile endpoints. One independent grayscale pyramid per source
# channel, plus a per-channel max value the shader uses to convert between
# native-range slider values and the 0..255 linearly-scaled pyramid storage.
# ---------------------------------------------------------------------------

@router.get("/{image_id}/ch/{channel}/info")
def channel_tile_info(image_id: str, channel: int):
    """DZI-shaped info for a single channel's pyramid.

    The dimensions match the full image (channel pyramids share image extent);
    the extra `data_max` field is the native-dtype maximum used to scale
    pixel values into the 0..255 PNG storage. The frontend converts its
    slider values via `clMin_normalized = clMin / data_max` for shader use.
    """
    if not store.contains(image_id):
        raise HTTPException(404, "Image not found")
    info = pyramid.dzi_info(image_id)
    info["channel"] = channel
    info["data_max"] = pyramid.get_channel_max(image_id, channel)
    p1, p99 = pyramid.get_channel_percentiles(image_id, channel)
    info["percentile_1"] = p1
    info["percentile_99"] = p99
    return JSONResponse(info)


@router.get("/{image_id}/ch/{channel}/info.dzi")
def channel_tile_info_dzi(image_id: str, channel: int):
    if not store.contains(image_id):
        raise HTTPException(404, "Image not found")
    info = pyramid.dzi_info(image_id)
    w = info["width"]; h = info["height"]
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Image TileSize="{ts}" Overlap="{ov}" Format="png" '
        'xmlns="http://schemas.microsoft.com/deepzoom/2008">'
        '<Size Width="{w}" Height="{h}"/>'
        '</Image>'
    ).format(ts=pyramid.TILE_SIZE, ov=pyramid.TILE_OVERLAP, w=w, h=h)
    return Response(content=xml, media_type="application/xml")


@router.get("/{image_id}/ch/{channel}/{level}/{col}_{row}.png")
def channel_tile(image_id: str, channel: int, level: int, col: int, row: int):
    """Render one grayscale tile for a specific source channel of an image."""
    if not store.contains(image_id):
        raise HTTPException(404, "Image not found")
    try:
        data = pyramid.get_channel_tile_png(image_id, channel, level, col, row)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


# DeepZoom default URL alias (mirrors the existing /info_files alias above).
@router.get("/{image_id}/ch/{channel}/info_files/{level}/{col}_{row}.png")
def channel_tile_alias(image_id: str, channel: int, level: int, col: int, row: int):
    return channel_tile(image_id, channel, level, col, row)
