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
```

Then either:

- **macOS — double-click `Lysa.app`** in Finder. The first launch shows a Privacy dialog ("Lysa would like to access files in your Documents folder") — click **Allow** once and the server starts and Chrome opens automatically. Drag `Lysa.app` to your Dock for one-click access afterward.
- **Any platform — run from terminal**:
  ```bash
  python server.py
  ```
  Then open http://localhost:8050 in **Chrome** (recommended).

The first launch creates an `uploads/` folder for cached image files (gitignored).

For installation, the interface walkthrough, and a tools cheatsheet, see [docs/Lysa-Manual.pdf](docs/Lysa-Manual.pdf).

## Requirements

- Python 3.9+
- **Chrome** (recommended — the only browser Lysa is actively tested in). Firefox, Safari, and Edge should work since Lysa uses standard WebGL2, but they aren't extensively validated.
- ~2× your largest LIF file's size in free disk space (uploads are content-addressed and dedup, but you still need room for the originals)

## Troubleshooting

**Double-clicking `Lysa.app` does nothing on macOS.** Some clone or extraction methods (`git clone` over an unusual protocol, unzipping a downloaded archive, copying the folder via certain file managers) can drop the executable bit on the launcher script. Fix:

```bash
chmod +x Lysa.app/Contents/MacOS/Lysa
```

Then double-click again. If the bundle still won't launch, run the script directly to see the error:

```bash
./Lysa.app/Contents/MacOS/Lysa
```

**`numpy.dtype size changed` error on startup.** Some `scikit-image` versions were built against numpy 1.x and break under numpy 2.x. Upgrade in place:

```bash
pip install --upgrade scikit-image
```

## License

MIT — see [LICENSE](LICENSE).
