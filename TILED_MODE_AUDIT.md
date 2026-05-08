# Lysa Tiled Mode — Tool Audit

Status of every tool/feature when `state.useTiledViewer = true`. Based on a full read of `static/index.html` and the backend routes.

Legend:
- **WORKS** — functions unchanged in tiled mode
- **WORKS (ported)** — I already ported the rendering/interaction path
- **DEGRADED** — runs but with a visual approximation or missing polish
- **BROKEN** — the backing operation runs but the user will not see a correct result
- **NEEDS SHADER** — only a true WebGL/LUT layer can fix it cleanly

---

## 1. Drawing & measurement tools (overlay canvas)

| Tool | State | Notes |
|---|---|---|
| Pan | WORKS | OSD native |
| Rectangle ROI (`roi`) | WORKS (ported) | `installTiledMouseTracker` dispatches; image-coord draw in `_tiledDrawROI` + `_tiledDrawInProgress` |
| Ellipse ROI (`ellipse`) | WORKS (ported) | same path |
| Polygon ROI (`polygon`) | DEGRADED | Click-to-add and image-coord preview work. The double-click / Enter finisher from the legacy canvas was not ported to the tiled tracker — needs 10 lines. |
| Point (`point`) | WORKS (ported) | |
| Line (`line`) | WORKS (ported) | |
| Measure distance (`measure`) | WORKS (ported) | |
| Angle (`angle`) | WORKS (ported) | three-click angle now dispatches through overlay |
| Freehand (`freehand`) | WORKS (ported) | mousemove accumulates image-coord path |
| Calibrate (`calibrate`) | WORKS (ported) | stores scale, unaffected by viewer |
| Text annotation (`anntext`) | WORKS (ported) | |
| Arrow (`annarrow`) | WORKS (ported) | |
| Rect annotation (`annrect`) | WORKS (ported) | |
| Ellipse annotation (`annellipse`) | WORKS (ported) | |
| Scribble (`annscribble`) | WORKS (ported) | |
| 4-point crop (`crop4`) | WORKS (ported) | draws quad; bake uses backend so it's fine |
| Edit mode drag-to-move | WORKS (ported) | hit-test done in image coords |

Saved ROI / polyline / measurement / annotation overlays all re-render on pan/zoom via `redrawTiledOverlay` — verified. Correct.

---

## 2. Contrast / colour / LUT panels

| Feature | State | Why |
|---|---|---|
| Brightness | DEGRADED | CSS `brightness()` is a simple scalar multiply; matches the backend only when `clMin=0`. Off by a bias otherwise. |
| Contrast / contrast limits (`clMin`/`clMax`) | DEGRADED | Approximated as `brightness + contrast` filters. Visually close on mid-gray images, wrong on images with clipped black/whites. |
| Gamma | **BROKEN** | CSS filter has no gamma. Current code `Math.pow(1.0, 1/gamma)` always returns 1 — literally a no-op. NEEDS SHADER. |
| Invert | WORKS | CSS `invert(1)`. |
| Opacity | WORKS | CSS `opacity()`. |
| Per-channel R/G/B toggles (channel isolation) | **BROKEN** | Legacy path rewrites ImageData. Tiled path has no pixel access. NEEDS SHADER. |
| LUT / colormap (viridis, magma, fire, etc.) | **BROKEN** | Legacy path runs client-side LUT against ImageData. No code path in tiled mode. NEEDS SHADER — or fall back to a server-side colormapped `/render` variant and invalidate the pyramid when the LUT changes. |
| "Apply current adjustments" (bake to image) | WORKS but **inconsistent with what you see** | Server uses real math; tiled preview uses CSS filter. After bake, the preview and the actual pixels finally agree — but during the dialog the WYSIWYG promise is broken. |

---

## 3. Threshold panel

| Operation | State | Notes |
|---|---|---|
| Run threshold (otsu / manual / adaptive) | WORKS | Backend `/api/images/{id}/threshold` unchanged. `thresholdPreview[id]` is populated. |
| Run on all images | WORKS | backend loop |
| **Show threshold overlay** | **BROKEN** | Legacy renderer draws `thrPrev.overlayImage` onto the base canvas at image coords (line 6991). In tiled mode the overlay canvas is screen-space and `redrawTiledOverlay` doesn't consult `state.thresholdPreview`. **Fix:** one code block in `redrawTiledOverlay` that walks the overlay-image pixels through `imgToOverlay` — cheap since the overlay is already a PNG the same size as the image and we can `drawImage` with a scale transform derived from the viewport. |
| Clear threshold | WORKS | state-only |

---

## 4. Segmentation (Classical + Cellpose)

| Operation | State | Notes |
|---|---|---|
| Run segmentation | WORKS | `/api/images/{id}/segment` is backend |
| Run on all images | WORKS | backend loop |
| Show segmentation contours | WORKS (ported) | `_tiledDrawSegObject` walks contour in image coords |
| Segmentation *mask* overlay (filled mask, not contours) | n/a | Only contours are drawn in legacy mode too — parity preserved |
| Object numbers at centroids | DEGRADED | Centroid labels aren't drawn in `_tiledDrawSegObject`. Legacy has them (line ~7029). **Fix:** 6 lines. |
| Hover highlight | DEGRADED | Legacy tracks `segHoverObject`; tiled path doesn't hit-test yet. **Fix:** extend `onOverlayMouseMove`. |
| Promote to ROIs | WORKS | state + panel |
| Export segmentation | WORKS | backend |
| Clear segmentation | WORKS | state + backend |

---

## 5. Filter & preprocessing panel

| Operation | State | Notes |
|---|---|---|
| Filter **preview** (Gaussian / median / unsharp / rolling-ball) | **BROKEN** | Path: `refreshFilterPreview` POSTs to `/api/images/{id}/filter/preview`, gets back a PNG, then stuffs it into `img.originalImageData` for the legacy canvas renderer to draw. Tiled mode reads tiles from the pyramid, which still holds the unfiltered pixels → the user sees no change. **Fix option A:** invalidate `pyramid._levels[image_id]` and refresh tiles on preview. Cheap but re-warms the pyramid every slider tick; debouncing already at 180 ms helps. **Fix option B:** preview endpoint builds a throwaway pyramid variant keyed by param hash, client swaps tile source. Better but more plumbing. |
| Apply to image (bake) | **BROKEN for display** | Backend bakes the filter successfully; pyramid cache is **never invalidated**, so tiles continue to serve pre-filter pixels. **Fix:** call `pyramid.clear_image(image_id)` inside the filter-apply route and force OSD to re-request tiles (`viewer.world.getItemAt(0).source`; easier path is `destroyTiledViewer` + `createTiledViewer` for that image). |
| Apply to all (batch) | **BROKEN for display** | Same pyramid-invalidation bug, all affected images. Same fix. |
| Undo filter bake | **BROKEN for display** | Reverts server pixels but not the pyramid. |
| Reset filters | WORKS | state-only |
| "Live preview non-destructive" toggle | WORKS | state-only; once the display bug is fixed this works automatically |

---

## 6. Transform panel (rotate / flip / crop / resize)

| Operation | State | Notes |
|---|---|---|
| Rotate 90 / 180 / 270 | **BROKEN for display** | Backend rotates; pyramid invalidation missing. Same one-line fix as filter-bake. |
| Flip H/V | **BROKEN for display** | Same. |
| Crop (from rect ROI) | **BROKEN for display** | Same, and image metadata (width/height) changes — tiled viewer must be rebuilt, not just re-opened, because DZI info is cached client-side. |
| Resize | **BROKEN for display** | Same as crop (dimensions change). |
| Undo transform | **BROKEN for display** | Same pattern. |

**The single common fix** for §5 and §6 is: any backend route that mutates `store`'s pixel data must call `pyramid.clear_image(image_id)` before returning, and the client must call `destroyTiledViewer(image_id); createTiledViewer(image_id, cell)` after every such operation (or swap the tile source URL so OSD re-sniffs /info).

---

## 7. Scale bar / calibration

| Feature | State | Notes |
|---|---|---|
| Set pixel size / unit | WORKS | state-only |
| Scale bar overlay | DEGRADED | Legacy draws scale bar directly onto the image canvas in image coords. Tiled mode has no scale-bar drawing in `redrawTiledOverlay`. **Fix:** draw in screen coords at the viewer corner from `redrawTiledOverlay`, using the OSD `viewport.getZoom()` to pick a "nice" length. ~30 lines. |

---

## 8. Batch processing panel

| Operation | State | Notes |
|---|---|---|
| Batch filter apply | **BROKEN for display** (see §5) |
| Batch threshold | **BROKEN for overlay display** (see §3) |
| Batch segmentation | WORKS (contours only) |
| Batch export (ROIs / measurements / segmentation / annotations to xlsx) | WORKS | backend `/api/export-*` unchanged |
| "Apply adjustments to all" | DEGRADED | Same CSS-filter caveat as §2 |
| Select subset of images | WORKS | state-only |

---

## 9. Root Growth workflow (agar plate seedlings)

| Step | State | Notes |
|---|---|---|
| Step 0 Setup (num plates / genotypes / names) | WORKS | |
| Step 1 Plates (draw bounding rects on the plate) | WORKS (ported) | `_tiledDrawRootGrowth` handles plate rects in image coords; drawing dispatched in tiled mouse tracker |
| Step 2 Genotype boxes | WORKS (ported) | same draw function |
| Step 3 Tracing (click-to-add root polyline per seedling) | WORKS (ported) | polyline click-to-add dispatched; image-coord preview and segments draw correctly |
| Save root / next genotype / finish | WORKS | state-only |
| Export Root Growth to xlsx | WORKS | backend |
| "Visible full-resolution zoom to see individual root hairs" | **WORKS as intended** — this is precisely what drove the tiled refactor |

**This is the critical workflow you flagged as mandatory, and it is functional.** I'd still want to do one end-to-end pass on the 20260319_01 plate to confirm nothing I missed.

---

## 10. Channel tools

| Operation | State | Notes |
|---|---|---|
| Split channels (R/G/B → three new images) | WORKS | backend `/split-channels`, creates new image IDs, opens a new tab |
| Merge channels | WORKS | backend |
| Channel histograms | WORKS | backend serves histogram PNGs |
| Per-channel visibility toggle | **BROKEN** | see §2 |

---

## 11. Metadata / info bar

| Feature | State | Notes |
|---|---|---|
| File name, dimensions, bit depth, channels, source format | WORKS | from `/api/images/{id}/metadata` |
| Cursor X/Y readout | WORKS (ported) | overlay `mousemove` fires `onOverlayMouseMove` which updates the info bar |
| Cursor RGB / intensity readout | **BROKEN** | Legacy reads from `img.originalImageData` directly. Tiled mode has no pixel buffer. **Fix:** add `/api/images/{id}/pixel?x=&y=` endpoint that returns the intensity tuple; debounce client-side. ~20 lines backend + client. |
| Histogram of visible region | WORKS | backend `/histogram` is full-image |

---

## 12. Annotations list / ROI list panels

| Feature | State | Notes |
|---|---|---|
| Panel lists (render, toggle visibility, delete, rename) | WORKS | state-only + `redrawTiledOverlay` |
| "Zoom to item" | **BROKEN** | Legacy path sets canvas pan/zoom directly. Tiled needs OSD `viewport.fitBounds(imageRectToViewport(...))`. **Fix:** ~15 lines. |
| Show/hide all | WORKS | |
| Reorder | WORKS | |

---

## 13. File I/O

| Feature | State | Notes |
|---|---|---|
| Open single file | WORKS | |
| Open folder | WORKS | |
| Close tab | WORKS — but `pyramid.clear_image(id)` is **not** called on close. Memory leak: each closed 100 MB plate keeps its full pyramid in RAM. **Fix:** one line. |
| Drag-and-drop | WORKS | |
| Reload / refresh tab | WORKS | |

---

## Summary by category

**Broken and blocking for real use:**
1. Filter preview display (§5)
2. Filter bake / transform bake display (§5, §6) — user presses Apply, nothing visibly changes
3. Threshold overlay (§3) — the actual thresholding runs, user just can't see it
4. Gamma (§2) — slider does nothing
5. LUTs / colormaps (§2) — slider does nothing
6. Per-channel isolation (§2, §10) — toggles do nothing

**Degraded but usable:**
7. Brightness / contrast (CSS approximation; close enough for preview, bake gives the real thing)
8. Scale bar (missing until I add it)
9. Polyline finish (missing double-click)
10. Segmentation centroid labels / hover
11. Zoom-to-item
12. Cursor RGB readout
13. Pyramid leak on tab close

**Works unchanged:**
Everything backend-driven: file I/O, segmentation compute, filter compute, threshold compute, transforms compute, histograms, split/merge channels, all exports, calibration, Root Growth data/compute, batch operations (compute side).

**Works because I ported them:**
All drawing tools (rect/ellipse/polygon/point/line/measure/angle/freehand/calibrate/anntext/annarrow/annrect/annellipse/annscribble/crop4/edit-mode), saved overlay rendering (ROIs, polylines, measurements, segmentation contours), Root Growth workflow, cursor X/Y, CSS-filter adjustments (brightness/contrast/invert/opacity).

---

## Recommended fix order (cheapest → highest leverage)

**Tier 0 — one-line fixes, do now (1 hour total):**
- Pyramid invalidation on filter-apply / transform / crop / resize / undo routes
- Client: `destroyTiledViewer` + `createTiledViewer` after any bake
- Pyramid clear on tab close
- Polyline double-click finish
- Segmentation centroid labels in `_tiledDrawSegObject`

**Tier 1 — small but distinct fixes (half a day each):**
- Threshold overlay draw in `redrawTiledOverlay` (leverages existing PNG overlay)
- Scale bar in screen coords
- Zoom-to-item via `viewport.fitBounds`
- `/api/images/{id}/pixel` endpoint + cursor RGB readout

**Tier 2 — one big fix that collapses six bugs (1–2 days):**
- WebGL shader layer between OSD tiles and the overlay canvas. A ~200-line fragment shader handles contrast limits, real gamma, per-channel RGB gates, and LUT sampling in one pass. This eliminates **all** of (gamma, LUT, channel isolation, brightness/contrast approximation mismatch, filter preview round-trip for display-only adjustments). OSD supports this via `drawer` hooks in 4.x or a custom canvas overlay composited over the tile grid.

**Tier 3 — filter preview re-architecture:**
- Either debounced pyramid-clear on preview, or a per-param-hash pyramid side-cache. Can wait until after Tier 2.

---

## What I'd do before the next user test

Tier 0, then end-to-end on your 20260319_01.tif plate running the Root Growth workflow to completion — that's your mandatory path and it's the one I'm least nervous about but have verified least. That test will also catch anything I missed in `_tiledDrawRootGrowth`.
