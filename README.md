# Lysa

**Microscopy image viewer and analysis tool.** Browser-based, runs locally, handles 16-bit fluorescence stacks and Leica LIF files without converting them to 8-bit.

Lysa is built for biologists who need to look at multi-channel microscopy data and pull measurements out of it — line profiles, ROI statistics, angle and distance measurements, root growth analysis — without spending the day in ImageJ macros.

## Features

- **Native bit-depth rendering.** 16-bit data stays 16-bit. Per-channel auto-contrast on load means images don't render blown-out the way they do when other tools force you to an 8-bit preview.
- **Leica LIF support.** Open multi-image LIF files directly. Filter sub-images by name pattern when you only want, say, the merged channels.
- **Per-channel display.** Each source channel gets its own LUT, contrast window, gamma, opacity, and blend mode — independent of the others. Compose 4–5 channel images (fluorescence + brightfield) without competing for R/G/B slots.
- **Tiled viewer for large images.** OpenSeadragon-backed pyramid rendering kicks in automatically above 15 MP, so 1.3 GB LIF files pan and zoom smoothly.
- **Tools.** Rectangle / ellipse / polygon ROIs, line profiles, multi-point sets, distance and angle measurements, freehand curves, calibration, crop (drag-rect or 4-corner), and an annotation layer (text, arrow, scribble, shapes).
- **Analysis.** Threshold (Otsu, manual), segmentation, root-growth workflow, batch apply across image sets.
- **Sessions.** Save the full UI state — image list, ROIs, annotations, channel settings — to JSON; reload later.

## Quick start

```bash
# 1. Clone
git clone https://github.com/prshanthramachandran/Lysa.git
cd Lysa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python server.py
```

Then open http://localhost:8050 in Chrome, Firefox, or Safari. The first launch creates an `uploads/` folder for cached image files (gitignored).

For installation, the interface walkthrough, and a tools cheatsheet, see [docs/Lysa-Manual.pdf](docs/Lysa-Manual.pdf).

## Requirements

- Python 3.9+
- A modern browser (Chrome / Firefox / Safari / Edge — anything with WebGL2)
- ~2× your largest LIF file's size in free disk space (uploads are content-addressed and dedup, but you still need room for the originals)

## License

MIT — see [LICENSE](LICENSE).
