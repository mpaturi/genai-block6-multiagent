"""classify_exception - the node wrapper's second line of defense (see
docs/plan.md §5).

Maps a caught exception to one of four kinds so degradation logic (and
later, Block 7's forensics) can distinguish an infrastructure failure
from something worth flagging as suspicious, instead of treating every
error identically.

neo4j.exceptions.ServiceUnavailable can mean either a genuine timeout or
a plain connection refusal, and the exception itself carries no separate
flag for which - the only signal available is its own message text, so
that's what this checks for a timeout-indicating substring.
"""
import asyncio

import httpx
from neo4j.exceptions import ServiceUnavailable
from pydantic import ValidationError

# Case-insensitive substrings that mean "this ServiceUnavailable was
# actually a timeout", not a plain connection refusal.
_TIMEOUT_MESSAGE_KEYWORDS = ["timed out", "timeout"]


def classify_exception(exc: Exception) -> str:
    """Return one of "timeout"/"connection_error"/"validation_error"/"unknown"."""
    # A real supervisory timeout (docs/plan.md §7's 150s ceiling, or a
    # tool's own internal timeout) always lands here first.
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"

    # ServiceUnavailable is ambiguous by itself - check its message text
    # for a timeout-indicating substring before falling back to a plain
    # connection_error classification.
    if isinstance(exc, ServiceUnavailable):
        message = str(exc).lower()
        if any(keyword in message for keyword in _TIMEOUT_MESSAGE_KEYWORDS):
            return "timeout"
        return "connection_error"

    # Any other flavor of "couldn't connect" - a plain refused connection,
    # or httpx's own connect-failure exception.
    if isinstance(exc, (ConnectionError, httpx.ConnectError)):
        return "connection_error"

    # A schema/shape problem, not an infrastructure problem.
    if isinstance(exc, ValidationError):
        return "validation_error"

    return "unknown"
