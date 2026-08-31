"""Shared aiohttp client session for the bot.

Creating a fresh ``aiohttp.ClientSession`` for every request prevents TCP
connection reuse (Keep-Alive), adds overhead from repeated TCP handshakes,
and can exhaust file descriptors when several generations run at once.

This module provides a lazily-created session that is shared by all callers
(ComfyUI API, LLM endpoint, Discord image downloads) and is closed once
when the bot shuts down.

The session is recreated whenever a different event loop is active, e.g. if
a session is created outside the bot's main loop (a transient or test loop)
and then accessed from the main loop. aiohttp versions that bind a session
to its creation loop make the old session unusable afterwards.
"""

import asyncio
import logging

import aiohttp

log = logging.getLogger("http_session")

_session: aiohttp.ClientSession | None = None
_session_loop: asyncio.AbstractEventLoop | None = None


def get_session() -> aiohttp.ClientSession:
    """Return the shared ClientSession for the current event loop.

    The session is created lazily on first use (it must be created inside a
    running event loop). If the current running loop differs from the one
    the session was created in, the old session is discarded and a new one
    is created.
    """
    global _session, _session_loop
    loop = asyncio.get_running_loop()
    if _session is None or _session.closed or _session_loop is not loop:
        _session = aiohttp.ClientSession()
        _session_loop = loop
    return _session


async def close_session() -> None:
    """Close the shared session. Called once on bot shutdown."""
    global _session, _session_loop
    if _session is None:
        return
    try:
        await _session.close()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # If the loop is being torn down (e.g. Ctrl+C), closing sockets can
        # raise harmless connection-reset errors. We must still mark the
        # session as closed so Python's interpreter-exit hooks don't print
        # "Unclosed client session" warnings.
        log.debug("Error while closing aiohttp session (ignored): %s", exc)
    _session = None
    _session_loop = None
