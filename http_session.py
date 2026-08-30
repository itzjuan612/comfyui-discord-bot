"""Shared aiohttp client session for the bot.

Creating a fresh ``aiohttp.ClientSession`` for every request prevents TCP
connection reuse (Keep-Alive), adds overhead from repeated TCP handshakes,
and can exhaust file descriptors when several generations run at once.

This module provides a lazily-created session that is shared by all callers
(ComfyUI API, LLM endpoint, Discord image downloads) and is closed once
when the bot shuts down.

The session is recreated whenever a different event loop is active. This
matters because ``llm_client`` runs ``asyncio.run()`` at import time (to
build the LLM model autocomplete list); a session created inside that
temporary loop is unusable in the bot's main loop on aiohttp versions that
bind the session to its creation loop.
"""

import asyncio

import aiohttp

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
    if _session is not None:
        await _session.close()
        _session = None
        _session_loop = None
