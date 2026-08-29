"""NSFW guardrail for the image generation bot.

Detects NSFW text (prompts) and NSFW channels. The bot refuses NSFW prompts
unless the command is run in a channel that Discord has marked NSFW.

Channel detection prefers the documented ``is_nsfw()`` method and falls back
to raw attribute forms so it works across discord.py versions.
"""
import asyncio
import logging
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
# Image-level NSFW check (lightweight, CPU-only ONNX classifier).
#
# Uses the opennsfw-onnx package (Yahoo open_nsfw ResNet-50 via ONNX
# Runtime). It is lazy-loaded so importing nsfw_guard has no cost until an
# image is actually checked. Runs in a worker thread to avoid blocking the
# asyncio event loop.
# ---------------------------------------------------------------------------

_image_classifier = None
_image_threshold = 0.5
_image_check_enabled = True


def configure_image_check(enabled: bool = True, threshold: float = 0.5) -> None:
    """Enable/disable the image-level NSFW check and set the decision threshold."""
    global _image_threshold, _image_check_enabled
    _image_check_enabled = bool(enabled)
    _image_threshold = float(threshold)


def _get_image_classifier():
    """Lazily build the ONNX NSFW classifier (CPU-only)."""
    global _image_classifier
    if _image_classifier is None:
        from opennsfw_onnx import NSFWClassifier
        # CPU-only keeps it lightweight and avoids VRAM contention with ComfyUI.
        _image_classifier = NSFWClassifier(providers=["CPUExecutionProvider"])
        _image_classifier.warmup()
        log.info("NSFW image classifier loaded (CPU ONNX)")
    return _image_classifier


def is_nsfw_image(image_bytes: bytes) -> bool:
    """Return True if the image is classified as NSFW."""
    pred = _get_image_classifier().classify(image_bytes)
    log.info("nsfw_image_check: nsfw_score=%.3f threshold=%.2f -> %s",
             pred.nsfw, _image_threshold, "NSFW" if pred.nsfw >= _image_threshold else "safe")
    return pred.nsfw >= _image_threshold


async def check_image_nsfw(image_bytes: bytes, interaction) -> bool:
    """Run the image-level NSFW check and decide whether the image may be sent.

    Returns True if the image is NSFW *and* the channel is not NSFW-marked
    (i.e. it must be refused). Safe images, or NSFW images in an NSFW channel,
    return False (allowed).
    """
    if not _image_check_enabled:
        return False
    is_nsfw = await asyncio.to_thread(is_nsfw_image, image_bytes)
    nsfw_channel = is_nsfw_channel(interaction.channel)
    return is_nsfw and not nsfw_channel
