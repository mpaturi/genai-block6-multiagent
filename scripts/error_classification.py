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
from neo4j.exceptions import ClientError, ServiceUnavailable, SessionExpired, TransientError
from pydantic import ValidationError

# Case-insensitive substrings that mean "this ServiceUnavailable was
# actually a timeout", not a plain connection refusal.
_TIMEOUT_MESSAGE_KEYWORDS = ["timed out", "timeout"]

# The ClientError codes this classifier treats as a timeout - not a
# ServiceUnavailable (the driver only raises ServiceUnavailable for
# transport-level failures) - a PR #8 review's regression finding.
# Verified directly against a live Neo4j 5.18-community server (not
# assumed): a real Query(timeout=...) expiring server-side surfaces as
# the "ClientConfiguration" variant specifically - that suffix denotes a
# client-requested transaction timeout, distinct from the bare code,
# which is what a server-configured transaction timeout
# (dbms.transaction.timeout) surfaces as instead. Both recognized here
# since both are real, valid Neo4j timeout codes, even though this
# codebase's own Query(timeout=...) calls will only ever produce the
# ClientConfiguration variant (a later follow-up finding - PR #8's
# original fix checked only the wrong one of the two real codes).
_CLIENT_ERROR_TIMEOUT_CODES = {
    "Neo.ClientError.Transaction.TransactionTimedOut",
    "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
}


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

    # A real Query(timeout=...) expiring raises one of these specific
    # ClientError codes - not every ClientError is a timeout (a syntax
    # error is a real bug, not worth retrying), so only these count.
    if isinstance(exc, ClientError) and exc.code in _CLIENT_ERROR_TIMEOUT_CODES:
        return "timeout"

    # A transient database condition (leader switch, deadlock, momentarily
    # unavailable) - retrying later plausibly helps, same bucket as timeout.
    if isinstance(exc, TransientError):
        return "timeout"

    # The session itself is no longer usable (e.g. after a dropped
    # connection) - a fresh session/connection is what's needed, not a
    # retry of the same one, so this is a connection problem.
    if isinstance(exc, SessionExpired):
        return "connection_error"

    # Any other flavor of "couldn't connect" - a plain refused connection,
    # or httpx's own connect-failure exception.
    if isinstance(exc, (ConnectionError, httpx.ConnectError)):
        return "connection_error"

    # A schema/shape problem, not an infrastructure problem.
    if isinstance(exc, ValidationError):
        return "validation_error"

    return "unknown"
