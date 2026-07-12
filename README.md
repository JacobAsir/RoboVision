# RoboVision

**AI computer vision for enterprise security & logistics** — built for [RoboFounders.ai](https://robofounders.ai).

RoboVision is a bilingual (**English / 日本語**) Streamlit demo that shows real-time object detection and event logging for two operational use cases:

1. **Secure Product Monitoring** — detect when monitored products (e.g. bottles) are removed from view and log the event  
2. **Loading & Packing Verification** — count packages on a conveyor against a manifest quantity and flag mismatches  

---

## Features

| Feature | Description |
|--------|-------------|
| **Secure CCTV demo** | YOLOv8 tracking + spatial slots so multi-item removals log correctly |
| **Loading verification** | YOLO-World open-vocab package detection + belt ROI |
| **Bilingual UI** | English / 日本語 toggle (brand name **RoboVision** stays English) |
| **Live metrics & logs** | Removals, counts, match/mismatch status, CSV export |
| **Demo videos** | Built-in bottle shelf + conveyor footage; optional upload |
| **Fresh sessions** | Tab switch / refresh clears prior logs so each run starts clean |

---

## Project structure

```
RoboVision/
├── main.py                 # Streamlit app (UI + vision pipeline)
├── requirements.txt        # Python dependencies
├── pyproject.toml
├── README.md
├── logo-icon.png           # Header / tab icon
├── rofi-3d.png             # Mascot
├── bottle-detection.mp4    # Secure monitoring demo
├── 5903898-hd_1920_1080_30fps.mp4   # Loading demo
├── yolov8n.pt              # General detection / tracking
├── yolov8s.pt              # Optional larger COCO model
├── yolov8s-world.pt        # Open-vocab packages (loading tab)
└── robovision.db           # SQLite event logs (created at runtime)
```

---

## Requirements

- **Python** 3.10+ (3.12 recommended)
- **Windows / macOS / Linux**
- CPU is fine for the demo; GPU optional (CUDA) via PyTorch if installed separately  
- Disk: models + demo videos ≈ **tens of MB–hundreds of MB**

---

## Local setup

### 1. Clone / open the project

```bash
cd RoboVision
```

### 2. Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

First run may download extra Ultralytics/Torch assets if weights are missing.  
This repo already includes `yolov8n.pt` and `yolov8s-world.pt` for offline demos.

### 4. Run the app

```bash
streamlit run main.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## How to use

### Language

- Toggle **English / 日本語** under the robot mascot (top-right).  
- Switching language does **not** reset analysis.  
- **RoboVision** and **RoboFounders.ai** stay in English.

### Secure Product Monitoring

1. Select the use case tab  
2. Keep **Use Demo Video** (or upload your own)  
3. Target product defaults to **bottle**  
4. Leave **Activate CCTV Feed** on  
5. Watch for **Product Removed** logs when items leave the frame  

Demo video starts at **~3 seconds** (skips idle intro).

### Loading & Packing Verification

1. Select the loading use case  
2. Item type **package** uses YOLO-World + belt ROI  
3. Set **Expected Quantity (Manifest)** (e.g. `3`–`4` for the demo belt)  
4. Compare **Detected** vs **Expected** (match / mismatch)

---

## Configuration (optional)

Key constants live at the top of [`main.py`](main.py):

| Constant | Role |
|----------|------|
| `SECURE_CONF_THRESHOLD` | Detection confidence for secure tab |
| `SECURE_PLAYBACK_SPEED` | Demo playback speed multiplier |
| `SECURE_DEMO_START_SEC` | Skip first N seconds of bottle demo |
| `PACKAGE_CONF_THRESHOLD` | Package (YOLO-World) confidence |
| `OCCLUSION_LIMIT` / `MIN_PRODUCT_SEEN` | Removal confirmation windows |

---

## Tech stack

- [Streamlit](https://streamlit.io/) — UI  
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — detection / tracking  
- [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) — open-vocab packages  
- OpenCV — video I/O & drawing  
- Pandas + SQLite — logs & CSV export  
- Pillow — Japanese text on video frames (Windows fonts: Meiryo / Yu Gothic)

---

## Deployment

### Important: Vercel is **not** a good host for this app

RoboVision is a **long-running Streamlit process** with:

- Continuous video + YOLO inference  
- Large weight files (`*.pt`) and demo MP4s  
- WebSocket-style streaming UI  

[Vercel](https://vercel.com/) is built for **serverless / short HTTP** workloads (Next.js, static sites, edge functions). It is a poor fit because:

| Constraint | Why it breaks this app |
|------------|-------------------------|
| Serverless timeout | Vision loops run for minutes |
| No persistent process | Streamlit needs a live server |
| Package size limits | Torch + OpenCV + YOLO exceed typical limits |
| Binary / native deps | OpenCV & Torch are heavy on Linux serverless |

**Do not expect `vercel deploy` to run Streamlit + YOLO out of the box.**

---

### Recommended hosts (pick one)

#### Option A — Streamlit Community Cloud (simplest demo)

1. Push this repo to **GitHub**  
2. Go to [share.streamlit.io](https://share.streamlit.io)  
3. Deploy → select repo → main file `main.py`  
4. Ensure `requirements.txt` is at the repo root  

**Note:** Free tier has resource limits; large videos/models may need Git LFS or download-on-first-run.

#### Option B — Render (recommended Docker deploy)

This repo includes production files for [Render](https://render.com):

| File | Purpose |
|------|---------|
| [`Dockerfile`](Dockerfile) | Image with OpenCV system libs + Streamlit |
| [`render.yaml`](render.yaml) | Render Blueprint (one-click style deploy) |
| [`.dockerignore`](.dockerignore) | Smaller, faster builds |

**Deploy with Blueprint**

1. Push the project to GitHub  
2. Render Dashboard → **New** → **Blueprint**  
3. Connect the repo (reads `render.yaml`)  
4. Apply → wait for build  

**Deploy manually (Web Service)**

1. **New** → **Web Service** → connect repo  
2. Runtime: **Docker**  
3. Dockerfile path: `./Dockerfile`  
4. Instance: **Starter** or higher (YOLO needs RAM; free tier may OOM)  
5. Deploy  

App URL will look like: `https://robovision.onrender.com`

**Notes for Render**

- First build can take several minutes (PyTorch / Ultralytics)  
- Cold starts on free/sleeping instances are slow  
- Keep `*.pt` and demo `*.mp4` in the repo (or download in Docker build)  
- Logs: Render dashboard → service → Logs  

#### Option C — Docker on Railway / Fly.io / a VPS

Same [`Dockerfile`](Dockerfile) works on Railway, Fly.io, Azure Container Apps, AWS ECS, etc.

#### Option D — Vercel only for a **marketing site**

Use Vercel for a landing page that **links** to the Streamlit demo hosted elsewhere (Cloud / Railway).  
Keep vision compute off Vercel.

---

### If you still want a Vercel project in the monorepo

You can put a small static or Next.js landing page under e.g. `web/` and deploy **that** folder to Vercel. Keep `main.py` on Streamlit Cloud or Docker.

Example Vercel project root for a static landing (optional, not included):

```text
web/
  index.html   → “Open RoboVision demo” button → Streamlit URL
```

---

## Git tips before deploy

Add a `.gitignore` if missing (do **not** commit secrets; consider not committing huge binaries if the host can download models):

```gitignore
.venv/
__pycache__/
*.pyc
temp_assets/
.streamlit/secrets.toml
# optional: exclude large files if using LFS or runtime download
# *.pt
# *.mp4
robovision.db
```

For GitHub + Streamlit Cloud with large files, prefer [Git LFS](https://git-lfs.com/) for `*.pt` / demo `*.mp4`.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Blank video / no demo | Confirm MP4s sit next to `main.py`; use **Use Demo Video** |
| Japanese UI but empty radios | Hard refresh (Ctrl+F5); language uses stable option ids |
| No removal logs | Let bottles stay visible a few frames, then leave the frame |
| Package count jumps | Adjust Expected quantity; belt ROI counts only conveyor area |
| Slow on CPU | Expected; secure tab already resizes frames for speed |
| `ModuleNotFoundError` | Activate `.venv` and `pip install -r requirements.txt` |

---

## License & brand

- Demo for **RoboFounders** / **RoboVision** presentations  
- Third-party models (Ultralytics YOLO) are subject to their own licenses  
- Demo videos and assets are for demonstration use  

---

## Quick start (copy-paste)

```bash
cd RoboVision
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run main.py
```

For production demos to Japanese clients: host on **Streamlit Cloud** or **Docker + Railway/Render**, not Vercel.
