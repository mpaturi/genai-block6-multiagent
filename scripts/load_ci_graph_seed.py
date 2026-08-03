"""Loads data/seed/ci_graph_seed.cypher into a running Neo4j instance -
used by CI to seed the ephemeral service container before tests/eval run.
Not meant for the real Block 3 graph, and not something a local dev run
needs. Mirrors Block 5's block5_agent/load_ci_graph_seed.py exactly,
pointed at this repo's own seed path (data/seed/, not data/eval/).
"""
import os
import time
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

SEED_PATH = Path("data/seed/ci_graph_seed.cypher")
_MAX_WAIT_SECONDS = 60


def _wait_for_neo4j(driver) -> None:
    """CI starts this step as soon as the service container is created,
    which can be before Neo4j is actually ready to accept connections -
    retry instead of requiring a separate Docker healthcheck."""
    deadline = time.monotonic() + _MAX_WAIT_SECONDS
    while True:
        try:
            driver.verify_connectivity()
            return
        except ServiceUnavailable:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def main() -> None:
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    statements = [
        line.strip().rstrip(";")
        for line in SEED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        _wait_for_neo4j(driver)
        with driver.session(database=database) as session:
            for statement in statements:
                session.run(statement)

    print(f"Loaded {len(statements)} statements from {SEED_PATH}")


if __name__ == "__main__":
    main()
