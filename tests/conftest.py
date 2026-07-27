"""Session-wide pytest fixtures (see docs/plan.md §6).

Every test in this repo uses fakes - no test ever calls
scripts.cohort_tool.get_driver() directly. This fixture exists for the
Neo4j driver teardown plan.md §6 requires once Phase 3 introduces that
factory: a disposable test-container driver, if one was ever actually
constructed during the session (e.g. by a future integration test), gets
a clean .close() rather than lingering past teardown and risking flaky
CI from an unclosed connection.

Guarded for ImportError so Phase 2's fakes-only suite - where
scripts.cohort_tool doesn't exist yet - still collects and runs; the
guard stays useful after Phase 3 too, since most test files here will
still never construct a real driver.
"""
import pytest


@pytest.fixture(scope="session", autouse=True)
def _close_neo4j_driver_after_session():
    yield
    try:
        from scripts.cohort_tool import get_driver
    except ImportError:
        return
    # get_driver() is functools.lru_cache(maxsize=1) (plan.md §6) - only
    # close it if a driver was actually constructed during the session,
    # never construct one for the first time just to close it.
    if get_driver.cache_info().currsize > 0:
        get_driver().close()
