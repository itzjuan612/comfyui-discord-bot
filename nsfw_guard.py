"""NSFW guardrail for the image generation bot.

Detects NSFW text (prompts) and NSFW channels. The bot refuses NSFW prompts
unless the command is run in a channel that Discord has marked NSFW.

Channel detection prefers the documented ``is_nsfw()`` method and falls back
to raw attribute forms so it works across discord.py versions.

Image-level checking uses EraX-NSFW-V1.0 (YOLO11 nano, Apache-2.0), exported
to ONNX and run locally on CPU via ONNX Runtime. It is lazy-loaded so importing
nsfw_guard has no cost until an image is actually checked, and it runs in a
worker thread so the asyncio event loop is never blocked.
"""
import asyncio
import io
import logging
import os
import re

log = logging.getLogger("nsfw_guard")

# Broad keyword list. Matched as whole words (case-insensitive), with an
# optional trailing "s" for plurals. Includes common synonyms and slang so
# creative phrasings are still caught.
DEFAULT_NSFW_TERMS = (
    # Explicit / pornographic
    "nsfw", "porn", "pornographic", "pornography", "explicit", "xxx", "erotic", "eroticism", "hentai",
    # Nudity
    "nude", "naked", "nudity", "topless", "strip", "striptease",
    "undress", "undressed",
    # Sexual acts
    "sex", "sexual", "intercourse", "orgy", "threesome", "gangbang",
    "fetish", "kinky", "lust", "arousal", "aroused", "orgasm",
    "cum", "cumshot", "creampie",
    # Body parts / sexual references
    "boob", "boobs", "breast", "tits", "titty", "butt", "ass",
    "asshole", "vagina", "vulva", "pussy", "penis", "cock", "dick",
    "anal", "anus", "booty", "thong", "lingerie",
    # Other
    "stripper", "sex worker", "prostitute", "escort", "slut", "whore",
    "sexy", "seductive", "seduction", "18+",
)

_pattern = None


def configure(extra_terms=()):
    """(Re)build the compiled keyword pattern, optionally extending the
    built-in term list with user-supplied extra keywords from config."""
    global _pattern
    terms = tuple(t.lower() for t in DEFAULT_NSFW_TERMS) + tuple(
        t.lower() for t in extra_terms
    )
    body = "|".join(re.escape(t) for t in terms)
    # Match whole words only, allowing a trailing "s" for plurals.
    _pattern = re.compile(r"(?<![A-Za-z])(?:%s)s?(?![A-Za-z])" % body)


if _pattern is None:
    configure()


def is_nsfw(text: str) -> bool:
    """Return True if the text contains an NSFW keyword."""
    if not text:
        return False
    return _pattern.search(text.lower()) is not None


def is_nsfw_channel(channel) -> bool:
    """Return True if the channel is marked NSFW by Discord.

    Prefers the documented ``is_nsfw()`` method, then falls back to raw
    attribute forms (``nsfw`` / ``_nsfw``) so it works across discord.py
    versions. Returns False when NSFW status cannot be determined, which
    keeps non-NSFW channels from being treated as NSFW.
    """
    # Documented method (preferred).
    try:
        return bool(channel.is_nsfw())
    except (AttributeError, TypeError):
        pass
    # Fallback to raw attribute forms.
    for attr in ("nsfw", "_nsfw"):
        value = getattr(channel, attr, None)
        if isinstance(value, bool):
            return value
    return False


# ---------------------------------------------------------------------------
# Image-level NSFW check (EraX-NSFW-V1.0, CPU-only ONNX Runtime).
#
# EraX-NSFW-V1.0 is a YOLO11 nano object detector (classes: anus,
# make_love, nipple, penis, vagina). Exported to ONNX, it runs entirely on
# the CPU, so it does not fight ComfyUI for GPU/VRAM. For moderation we only
# need to know whether *any* NSFW object is present, so we take the maximum
# class score across all anchor positions (no NMS needed).
# ---------------------------------------------------------------------------

# Canonical ONNX path: download_erax.py always writes the chosen size here,
# so the guard loads the same file regardless of nano/small/medium.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "erax_nsfw.onnx")
_INPUT_SIZE = 640
_NUM_CLASSES = 5  # anus, make_love, nipple, penis, vagina

_image_session = None
_image_threshold = 0.3
_image_check_enabled = True


def configure_image_check(enabled: bool = True, threshold: float = 0.3) -> None:
    """Enable/disable the image-level NSFW check and set the decision threshold.

    ``threshold`` is a confidence in [0, 1]; the image is flagged NSFW when
    the strongest detected NSFW class score meets or exceeds it.
    """
    global _image_threshold, _image_check_enabled
    _image_check_enabled = bool(enabled)
    _image_threshold = float(threshold)


def _get_image_session():
    """Lazily build the ONNX Runtime session (CPU-only)."""
    global _image_session
    if _image_session is None:
        import onnxruntime as ort

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "EraX-NSFW-V1.0 ONNX model not found at %s. "
                "Run `python download_erax.py` (pick nano/small/medium) to export it." % MODEL_PATH
            )
        sess_options = ort.SessionOptions()
        # Keep the thread count low so the check stays lightweight and leaves
        # CPU cores free for ComfyUI generation.
        sess_options.intra_op_num_threads = 2
        _image_session = ort.InferenceSession(MODEL_PATH, sess_options)
        log.info("NSFW image classifier loaded (EraX-NSFW-V1.0, CPU ONNX)")
    return _image_session


def _preprocess(image_bytes: bytes):
    """Letterbox the image to 640x640 and normalize for YOLO.

    Returns a float32 array shaped (1, 3, 640, 640), padded with gray (114)
    and normalized with ``(pixel - 114) / 255`` as YOLO expects.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    w, h = img.size
    scale = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (_INPUT_SIZE, _INPUT_SIZE), (114, 114, 114))
    canvas.paste(resized, ((_INPUT_SIZE - new_w) // 2, (_INPUT_SIZE - new_h) // 2))
    arr = np.asarray(canvas, dtype=np.float32)
    # YOLO normalization: divide by 255 (letterbox padding value 114 becomes ~0.447).
    arr = arr / 255.0
    return arr.transpose(2, 0, 1)[None]


def _max_nsfw_score(image_bytes: bytes) -> float:
    """Run the ONNX detector and return the highest NSFW class score in [0, 1]."""
    import numpy as np

    session = _get_image_session()
    data = _preprocess(image_bytes)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: data})[0]
    # Output shape: (1, 4 + num_classes, num_anchors); boxes are dims 0..3.
    scores = output[0, 4 : 4 + _NUM_CLASSES]
    return float(np.max(scores))


def is_nsfw_image(image_bytes: bytes) -> bool:
    """Return True if the image is classified as NSFW."""
    score = _max_nsfw_score(image_bytes)
    log.info(
        "nsfw_image_check: max_score=%.3f threshold=%.2f -> %s",
        score, _image_threshold, "NSFW" if score >= _image_threshold else "safe",
    )
    return score >= _image_threshold


async def check_image_nsfw(image_bytes: bytes, interaction) -> bool:
    """Run the image-level NSFW check and decide whether the image may be sent.

    Returns True if the image is NSFW *and* the channel is not NSFW-marked
    (i.e. it must be refused). Safe images, or NSFW images in an NSFW channel,
    return False (allowed).
    """
    if not _image_check_enabled:
        return False
    try:
        is_nsfw = await asyncio.to_thread(is_nsfw_image, image_bytes)
    except Exception as exc:  # Fail open: never block generation on a classifier error.
        log.warning("NSFW image check failed (%s); allowing image", exc)
        return False
    nsfw_channel = is_nsfw_channel(interaction.channel)
    return is_nsfw and not nsfw_channel
