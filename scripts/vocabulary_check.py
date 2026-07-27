"""get_known_vocabulary - Block 3's real condition/lab vocabulary (see
docs/plan.md §12).

Two layers use this: the CI-time entry point (Phase 4) that checks the 8
fixed eval questions against it once, and reconcile_node (scripts/
orchestrator.py), which calls this at runtime for any question whenever
it hits a nothing_found/answered split - so a genuinely new, unfamiliar
condition or lab isn't mistaken for a graph-schema mismatch just because
it hasn't been added to a fixed list.

Cached with a TTL (docs/plan.md §12) rather than indefinitely, so a
condition/lab added to Block 3's graph after process startup - a real
concern once this becomes a long-running deployed service - doesn't get
silently misreported as unknown forever.
"""
import time

from scripts.cohort_tool import KNOWN_LAB_NAMES, NEO4J_DATABASE, get_driver

# Re-query Block 3's graph if the cached vocabulary is older than this,
# rather than caching indefinitely (docs/plan.md §12).
_CACHE_TTL_SECONDS = 15 * 60

_cached_vocabulary = None
_cached_at = 0.0

_DISTINCT_CONDITION_NAMES_QUERY = """
MATCH (c:Condition)
RETURN DISTINCT c.condition_name AS condition_name
"""


def _fetch_known_vocabulary(*, driver=None) -> dict:
    """Query Block 3's graph for its real, current condition names.

    Lab names are this repo's own fixed whitelist (scripts/cohort_tool.py's
    KNOWN_LAB_NAMES) rather than a graph query - labs are stored as Patient
    node properties (latest_sbp, latest_bmi, ...), not as their own nodes
    with a queryable "lab name", so the known-lab set is exactly the set
    of lab names this repo's own Cypher already knows how to handle.
    """
    driver = driver if driver is not None else get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        rows = session.run(_DISTINCT_CONDITION_NAMES_QUERY)
        conditions = {row["condition_name"] for row in rows}
    return {"conditions": conditions, "labs": set(KNOWN_LAB_NAMES)}


def get_known_vocabulary(*, driver=None) -> dict:
    """Return {"conditions": set[str], "labs": set[str]} - Block 3's real,
    current vocabulary, cached for up to _CACHE_TTL_SECONDS.
    """
    global _cached_vocabulary, _cached_at
    now = time.monotonic()
    is_cache_fresh = _cached_vocabulary is not None and (now - _cached_at) < _CACHE_TTL_SECONDS
    if is_cache_fresh:
        return _cached_vocabulary

    _cached_vocabulary = _fetch_known_vocabulary(driver=driver)
    _cached_at = now
    return _cached_vocabulary
