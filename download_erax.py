"""Download and export the EraX-NSFW-V1.0 model to ONNX (CPU).

Usage:
    python download_erax.py            # interactive: asks which size
    python download_erax.py nano       # nano (fastest, smallest)
    python download_erax.py small      # small
    python download_erax.py medium     # medium (most accurate, slower)

The exported model is written to models/erax_nsfw.onnx (the same path for any
size), so nsfw_guard always loads that one file.

Requires: huggingface_hub, ultralytics (pulls in onnx / onnxslim automatically).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
CANONICAL_ONNX = os.path.join(MODEL_DIR, "erax_nsfw.onnx")

REPO_ID = "erax-ai/EraX-NSFW-V1.0"
SIZE_TO_FILE = {
    "nano": "erax_nsfw_yolo11n.pt",
    "small": "erax_nsfw_yolo11s.pt",
    "medium": "erax_nsfw_yolo11m.pt",
}


def model_exists() -> bool:
    """True if the canonical ONNX model is already on disk."""
    return os.path.exists(CANONICAL_ONNX)


def download(size: str = "nano") -> str:
    """Download the chosen size and export it to the canonical ONNX path.

    Returns the path to the exported ONNX file.
    """
    size = size.lower().strip()
    if size not in SIZE_TO_FILE:
        raise ValueError("Unknown size %r. Choose: nano, small, medium" % size)

    os.makedirs(MODEL_DIR, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    pt_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=SIZE_TO_FILE[size],
        local_dir=MODEL_DIR,
    )
    print("Downloading/exporting EraX-NSFW-V1.0 [%s]..." % size)

    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=640, opset=12, simplify=True)

    # Ultralytics writes <stem>.onnx next to the .pt. Move it to the
    # canonical path so nsfw_guard always finds the same file.
    generated = os.path.splitext(pt_path)[0] + ".onnx"
    if os.path.abspath(generated) != os.path.abspath(CANONICAL_ONNX):
        os.replace(generated, CANONICAL_ONNX)

    print("NSFW model ready: %s (%.2f MB)" % (CANONICAL_ONNX, os.path.getsize(CANONICAL_ONNX) / 1e6))
    return CANONICAL_ONNX


def prompt_size() -> str:
    """Ask the user which model size to use. Falls back to 'nano' when there
    is no interactive terminal (e.g. piped stdin)."""
    if not sys.stdin.isatty():
        print("No interactive terminal; defaulting to 'nano'.")
        return "nano"

    print("No NSFW model found on disk. Which one do you want to download?")
    print("  nano   - fastest, smallest (recommended)")
    print("  small  - balanced")
    print("  medium - most accurate, slower")
    while True:
        choice = input("Choice [nano]: ").strip().lower()
        if choice in ("", "nano"):
            return "nano"
        if choice in SIZE_TO_FILE:
            return choice
        print("Invalid choice. Pick nano, small, or medium.")


if __name__ == "__main__":
    args = [a.lower().strip() for a in sys.argv[1:] if a.strip()]
    size = args[0] if args else prompt_size()
    download(size)
