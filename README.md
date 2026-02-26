# BG-cleaner

Portrait background removal powered by [MODNet](https://github.com/ZHKKKe/MODNet) and [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0).

Upload a photo, get a clean alpha matte, fine-tune it with brightness / contrast / blur controls, and export as transparent PNG or standalone alpha mask.

---

## Features

- **Multi-model support** — switch between MODNet (fast, lightweight) and RMBG-2.0 (high quality) directly in the UI.
- **Auto-discovery** — drop an ONNX file in `assets/models/` and it appears in the model selector automatically.
- **Interactive alpha editing** — adjust brightness, contrast, sharpness, blur, and threshold on the matte before exporting.
- **Before / after comparison** — interactive slider to compare the original and the result side by side.
- **Dual export** — download the full RGBA image (transparent PNG) or the alpha matte alone.
- **GPU acceleration** — optional CUDA support for faster inference on NVIDIA GPUs.
- **Modular architecture** — provider-based design makes adding a new model a single-file operation.

---

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/bg-cleaner.git
cd bg-cleaner
```

### 2. Install dependencies

```bash
uv sync
uv pip install -e .
```

<details>
<summary>Without uv (pip fallback)</summary>

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

</details>

### 3. Download model weights

Place ONNX files in `assets/models/` using the **exact filenames** listed below. If the downloaded file has a different name, **rename it** before placing it in the directory.

#### Available models

| Model | Filename to use | Download size | Source |
|---|---|---|---|
| MODNet | `modnet.onnx` | ~25 MB | [HuggingFace](https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.onnx) |
| RMBG-2.0 (FP32) | `RMBG-2.0.onnx` | ~1 GB | [HuggingFace](https://huggingface.co/briaai/RMBG-2.0/tree/main/onnx) → `model.onnx` |
| RMBG-2.0 (INT8) | `RMBG-2.0_int8.onnx` | ~366 MB | [HuggingFace](https://huggingface.co/briaai/RMBG-2.0/tree/main/onnx) → `model_int8.onnx` |
| RMBG-2.0 (FP16) | `RMBG-2.0_fp16.onnx` | ~514 MB | [HuggingFace](https://huggingface.co/briaai/RMBG-2.0/tree/main/onnx) → `model_fp16.onnx` |

> **Important:** Downloaded files must be renamed to match the filenames above.
> For example, `model_int8.onnx` from HuggingFace must be renamed to `RMBG-2.0_int8.onnx`.

**Quick download examples:**

```bash
# MODNet
curl -L -o assets/models/modnet.onnx \
  "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.onnx"

# RMBG-2.0 INT8 (recommended for CPU)
curl -L -o assets/models/RMBG-2.0_int8.onnx \
  "https://huggingface.co/briaai/RMBG-2.0/resolve/main/onnx/model_int8.onnx"
```

### 4. Verify the setup

```bash
uv run python -c "
from bgcleaner.config import Settings
from bgcleaner.core.providers import discover_available
cards = discover_available(Settings())
print(f'Found {len(cards)} model(s):')
for c in cards:
    print(f'  - {c.name} ({c.filename})')
"
```

---

## Usage

### Run the Streamlit app

```bash
uv run streamlit run src/bgcleaner/ui/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Windows shortcut

A `launch.bat` script is provided that activates the venv, sets up NVIDIA DLL paths (if present), and starts the app:

```cmd
launch.bat
```

### User workflow

1. **Select** a model from the sidebar dropdown (only models found in `assets/models/` are listed).
2. **Upload** a portrait image (JPG, PNG, or WebP, up to 10 MB).
3. **Compare** the original and result using the interactive before/after slider.
4. **Adjust** the matte using the sidebar controls (brightness, contrast, sharpness, threshold, blur).
5. **Download** the transparent PNG or the alpha mask.

---

## GPU acceleration (optional)

BG-cleaner supports NVIDIA GPU inference via ONNX Runtime CUDA. This is optional — the app works fine on CPU.

### Prerequisites

- **CUDA 12.x** (any 12.x version works — tested with 12.6)
- An NVIDIA GPU with up-to-date drivers

### Setup

**Step 1 — Install GPU dependencies:**

```bash
uv sync --extra gpu
```

This installs `onnxruntime-gpu` (replaces the CPU-only `onnxruntime`) and `nvidia-cudnn-cu12` (cuDNN 9.x).

<details>
<summary>Without uv (pip fallback)</summary>

```bash
pip install -e ".[gpu]"
```

</details>

> On Windows, the app automatically registers the NVIDIA pip-installed DLL directories
> in `PATH` at startup, so no manual PATH configuration is needed.

**Step 2 — Enable CUDA** in your `.env` file:

```env
BGC_ONNX_PROVIDERS=["CUDAExecutionProvider","CPUExecutionProvider"]
```

The fallback to CPU is automatic — if CUDA initialisation fails, ONNX Runtime silently falls back to `CPUExecutionProvider`.

### Verify GPU is active

```bash
python -c "
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
"
```

> **Note:** The first inference after loading a model takes a few extra seconds due to
> CUDA/cuDNN initialisation. The app runs a warmup pass during model loading to absorb
> this cost — subsequent images are processed at full GPU speed.

---

## Docker (CPU)

A lightweight Docker image for CPU deployment. Model weights are mounted at runtime so the image stays small.

### Build

```bash
docker build -t bg-cleaner .
```

### Run

Mount your local `assets/models/` directory containing the ONNX files:

```bash
docker run --rm -p 8501:8501 \
  -v "$(pwd)/assets/models:/app/assets/models:ro" \
  bg-cleaner
```

Then open [http://localhost:8501](http://localhost:8501).

### Configuration

Pass environment variables to override settings:

```bash
docker run --rm -p 8501:8501 \
  -v "$(pwd)/assets/models:/app/assets/models:ro" \
  -e BGC_MAX_UPLOAD_MB=20 \
  bg-cleaner
```

---

## Project structure

```
bg-cleaner/
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .env.example
├── launch.bat               # Windows launcher with GPU support
├── assets/
│   └── models/              # Drop ONNX files here
│       ├── modnet.onnx
│       └── RMBG-2.0_int8.onnx
├── src/
│   └── bgcleaner/
│       ├── __init__.py
│       ├── config.py        # Pydantic BaseSettings
│       ├── schemas.py       # Data contracts (MattingInput/Output, etc.)
│       ├── errors.py        # Custom exception hierarchy
│       ├── core/
│       │   ├── __init__.py
│       │   ├── image_processing.py
│       │   └── providers/   # ← One file per model family
│       │       ├── __init__.py  # Registry + discovery
│       │       ├── _base.py     # Protocol + shared ONNX helpers
│       │       ├── modnet.py    # MODNet engine
│       │       └── rmbg.py     # RMBG-2.0 engine (all variants)
│       └── ui/
│           ├── __init__.py
│           ├── app.py
│           ├── state.py
│           └── widgets.py
└── tests/
```

### Adding a new model

1. Create `src/bgcleaner/core/providers/my_model.py`
2. Define `MODEL_CARDS` (list of `ModelCard`) and an engine class with a `predict(MattingInput) -> MattingOutput` method
3. Export a `create(model_path, settings) -> engine` factory function
4. Register it in `providers/__init__.py` (one line)

That's it — the UI picks it up automatically.

---

## Configuration

All settings can be overridden via environment variables (prefixed with `BGC_`) or a `.env` file:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `BGC_MODELS_DIR` | `assets/models` | Directory to scan for ONNX files |
| `BGC_ONNX_PROVIDERS` | `["CPUExecutionProvider"]` | ONNX Runtime execution providers |
| `BGC_MAX_UPLOAD_MB` | `10.0` | Maximum upload file size (MB) |

---

## Testing

```bash
uv run pytest
```

The test suite (90 tests) covers schemas validation, image processing, both provider engines (with mocked ONNX sessions), the registry discovery logic, and the exception hierarchy. **No model files are needed to run tests.**

---

## Model details

| Property | MODNet | RMBG-2.0 |
|---|---|---|
| Architecture | MODNet (MobileNetV2) | BiRefNet |
| Input normalisation | `[-1, 1]` | ImageNet mean/std |
| Input size | Dynamic (ref 512px) | 1024×1024 |
| Output activation | Sigmoid (built-in) | Logits (sigmoid applied in post) |
| Best for | Portraits, fast inference | General images, high quality |
| License | Apache 2.0 | CC BY-NC 4.0 |

---

## License

This project is licensed under the MIT License. Model weights have their own licenses (see table above).