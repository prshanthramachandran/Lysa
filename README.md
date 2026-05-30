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

Then launch — pick what fits your setup:

| Option | OS | Terminal? | Notes |
|---|---|---|---|
| Double-click **`Lysa.app`** | macOS | hidden | Looks like a regular app. Drag to your Dock for one-click access. |
| Double-click **`Start Lysa.command`** | macOS | visible | Opens Terminal so you can see server logs (handy for debugging). |
| Double-click **`Start Lysa.bat`** | Windows | visible | Opens Command Prompt, starts the server, opens default browser. |
| Double-click **`Start Lysa Desktop.command`** | macOS | visible | Runs Lysa in a **native desktop window** instead of a browser tab (see below). |
| `python server.py` | any | visible | Manual terminal launch (browser). |
| `python desktop.py` | any | visible | Native desktop window (browser not required). |

The browser opens to `http://localhost:8050`. The first launch creates an `uploads/` folder for cached image files (gitignored).

**macOS heads-up:** Apple's Privacy framework blocks unsigned `.app` bundles from reading files in `~/Documents`, `~/Downloads`, and `~/Desktop`. If you put Lysa in one of those folders, `Lysa.app` will show an alert telling you to either move the project elsewhere (e.g. `~/Lysa`) or grant access in **System Settings → Privacy & Security → Files and Folders**. The `.command` file and `python server.py` aren't affected — they inherit your terminal's permissions.

For installation, the interface walkthrough, and a tools cheatsheet, see [docs/Lysa-Manual.pdf](docs/Lysa-Manual.pdf).

## Desktop app

Lysa can run in a **native desktop window** instead of a browser tab, while
keeping the full feature set (tile viewer, per-channel pyramids, LIF,
analysis). It starts the same local server and shows it in the OS's built-in
webview via [pywebview](https://pywebview.flowdev.org/) — no bundled browser.

```bash
pip install -r requirements.txt -r requirements-desktop.txt
python desktop.py        # or double-click "Start Lysa Desktop.command" on macOS
```

macOS uses WebKit, Windows uses WebView2, Linux uses GTK/WebKit2. To ship a
double-clickable `.app` / `.exe`, bundle `desktop.py` with PyInstaller or
cx_Freeze (include the `static/` and `lysa/` trees).

## Update notifications

On launch, Lysa checks GitHub for a newer version and, if one exists, shows a
dismissable banner linking to it. It compares the running `lysa.__version__`
against the repository's highest **git tag**.

The check is deliberately unobtrusive:

- **Non-blocking & fail-silent** — runs after the UI loads, on a 3-second
  timeout; offline or API errors are ignored and never delay startup.
- **Cached** — queried at most once per 24 h (server-side); a dismissed
  version is remembered client-side so it won't re-nag.
- **Opt-out** — set the environment variable `LYSA_DISABLE_UPDATE_CHECK=1`
  to disable the network check entirely.
- **Privacy** — it's a single request to `api.github.com`, which necessarily
  exposes the client's IP to GitHub. Nothing else is sent.

### Releasing a new version

1. Bump `__version__` in [`lysa/__init__.py`](lysa/__init__.py).
2. Commit, then tag and push:
   ```bash
   git tag v3.1.0
   git push && git push --tags
   ```
   Use `vMAJOR.MINOR.PATCH`. Pushing the tag is enough for users' update
   banners to fire — no separate GitHub *Release* is required.

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
