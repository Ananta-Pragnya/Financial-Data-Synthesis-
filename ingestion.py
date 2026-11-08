from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

DEFAULT_SOURCE = Path(r"C:\Users\user\Downloads\archive\dax40_news_sentiment_labeled.csv")
DATASET_ID = "dataset:dax40_news_sentiment_2017_2026"
EXPECTED_COLUMNS = [
    "news_date", "ticker_id", "title", "sentiment_label", "impact_score",
    "time_horizon_months", "confidence", "sentiment_reason",
]
SENTIMENTS = {"negative", "neutral", "positive"}
HORIZONS = {"0-1": (0, 1), "1-3": (1, 3), "3-6": (3, 6)}

CONSTRAINTS = [
    "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (n:Dataset) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT sentiment_id IF NOT EXISTS FOR (n:Sentiment) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT horizon_id IF NOT EXISTS FOR (n:Horizon) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT month_id IF NOT EXISTS FOR (n:Month) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT news_id IF NOT EXISTS FOR (n:NewsItem) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX news_date IF NOT EXISTS FOR (n:NewsItem) ON (n.news_date)",
    "CREATE INDEX news_impact IF NOT EXISTS FOR (n:NewsItem) ON (n.impact_score)",
    "CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.name)",
]

INGEST_BATCH = """
UNWIND $rows AS row
MERGE (d:Dataset {id: $dataset_id})
SET d.name = 'DAX40 Financial News Sentiment',
    d.description = 'LLM-labeled financial news headlines for DAX40 constituents',
    d.license = 'Non-commercial research only',
    d.row_count = $row_count,
    d.start_date = datetime($start_date),
    d.end_date = datetime($end_date)
MERGE (c:Company {id: row.company_id})
SET c.name = row.company_name
MERGE (s:Sentiment {id: row.sentiment_id})
SET s.name = row.sentiment
MERGE (h:Horizon {id: row.horizon_id})
SET h.name = row.horizon,
    h.min_months = row.horizon_min,
    h.max_months = row.horizon_max
MERGE (m:Month {id: row.month_id})
SET m.label = row.month_label,
    m.year = row.year,
    m.month = row.month
MERGE (n:NewsItem {id: row.news_id})
SET n.news_date = datetime(row.news_date),
    n.title = row.title,
    n.impact_score = row.impact_score,
    n.confidence = row.confidence,
    n.sentiment_reason = row.sentiment_reason,
    n.source_row = row.source_row
MERGE (d)-[:CONTAINS]->(n)
MERGE (n)-[:ABOUT]->(c)
MERGE (n)-[:HAS_SENTIMENT]->(s)
MERGE (n)-[:HAS_HORIZON]->(h)
MERGE (n)-[:PUBLISHED_IN]->(m)
"""


def stable_id(prefix: str, value: str, length: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def utc_iso(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def process_source(source: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate the complete source before returning database-ready records."""
    records: list[dict[str, Any]] = []
    occurrences: Counter[str] = Counter()
    errors: list[str] = []

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        for source_row, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            try:
                if any(not row[column] for column in EXPECTED_COLUMNS):
                    raise ValueError("missing value")
                news_date = utc_iso(row["news_date"])
                impact = float(row["impact_score"])
                confidence_number = float(row["confidence"])
                if not confidence_number.is_integer():
                    raise ValueError(f"confidence is not an integer: {confidence_number}")
                confidence = int(confidence_number)
                sentiment = row["sentiment_label"].lower()
                horizon = row["time_horizon_months"]
                if sentiment not in SENTIMENTS:
                    raise ValueError(f"invalid sentiment {sentiment!r}")
                if horizon not in HORIZONS:
                    raise ValueError(f"invalid horizon {horizon!r}")
                if not -3 <= impact <= 3:
                    raise ValueError(f"impact score out of range: {impact}")
                if not 1 <= confidence <= 3:
                    raise ValueError(f"confidence out of range: {confidence}")
            except (TypeError, ValueError) as exc:
                errors.append(f"row {source_row}: {exc}")
                continue

            identity = "\0".join(row[column] for column in EXPECTED_COLUMNS)
            occurrences[identity] += 1
            month_label = news_date[:7]
            horizon_min, horizon_max = HORIZONS[horizon]
            records.append({
                "news_id": stable_id("news", f"{identity}\0{occurrences[identity]}", 24),
                "news_date": news_date,
                "title": row["title"],
                "impact_score": impact,
                "confidence": confidence,
                "sentiment_reason": row["sentiment_reason"],
                "source_row": source_row,
                "company_id": stable_id("company", row["ticker_id"], 20),
                "company_name": row["ticker_id"],
                "sentiment_id": f"sentiment:{sentiment}",
                "sentiment": sentiment,
                "horizon_id": f"horizon:{horizon}",
                "horizon": horizon,
                "horizon_min": horizon_min,
                "horizon_max": horizon_max,
                "month_id": f"month:{month_label}",
                "month_label": month_label,
                "year": int(month_label[:4]),
                "month": int(month_label[5:]),
            })

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Source validation failed for {len(errors)} rows:\n{preview}")
    if not records:
        raise ValueError("The source contains no data rows.")

    expected = {
        "Dataset": 1,
        "Company": len({row["company_id"] for row in records}),
        "Sentiment": len({row["sentiment_id"] for row in records}),
        "Horizon": len({row["horizon_id"] for row in records}),
        "Month": len({row["month_id"] for row in records}),
        "NewsItem": len(records),
    }
    return records, expected


def verify_database(session, expected: dict[str, int]) -> dict[str, Any]:
    node_counts = {
        label: session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()["count"]
        for label in ("Dataset", "Company", "Sentiment", "Horizon", "Month", "NewsItem")
    }
    relationship_counts = {
        row["type"]: row["count"]
        for row in session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count")
    }
    expected_relationships = {name: expected["NewsItem"] for name in (
        "CONTAINS", "ABOUT", "HAS_SENTIMENT", "HAS_HORIZON", "PUBLISHED_IN"
    )}
    incomplete = session.run(
        "MATCH (n:NewsItem) WHERE NOT EXISTS { MATCH (:Dataset)-[:CONTAINS]->(n) } "
        "OR NOT EXISTS { MATCH (n)-[:ABOUT]->(:Company) } "
        "OR NOT EXISTS { MATCH (n)-[:HAS_SENTIMENT]->(:Sentiment) } "
        "OR NOT EXISTS { MATCH (n)-[:HAS_HORIZON]->(:Horizon) } "
        "OR NOT EXISTS { MATCH (n)-[:PUBLISHED_IN]->(:Month) } "
        "RETURN count(n) AS count"
    ).single()["count"]
    if node_counts != expected:
        raise RuntimeError(f"Node verification failed. Expected {expected}, found {node_counts}")
    if relationship_counts != expected_relationships:
        raise RuntimeError(
            f"Relationship verification failed. Expected {expected_relationships}, found {relationship_counts}"
        )
    if incomplete:
        raise RuntimeError(f"{incomplete} NewsItem nodes do not have all required relationships")
    return {"nodes": node_counts, "relationships": relationship_counts, "incomplete_news": incomplete}


def ingest(source: Path, batch_size: int, replace: bool) -> dict[str, Any]:
    records, expected = process_source(source)
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        driver.verify_connectivity()
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            if replace:
                session.run("MATCH (n) DETACH DELETE n").consume()
            for statement in CONSTRAINTS:
                session.run(statement).consume()
            parameters = {
                "dataset_id": DATASET_ID,
                "row_count": len(records),
                "start_date": min(row["news_date"] for row in records),
                "end_date": max(row["news_date"] for row in records),
            }
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                session.run(INGEST_BATCH, rows=batch, **parameters).consume()
                print(f"Ingested {min(start + batch_size, len(records)):,}/{len(records):,}")
            result = verify_database(session, expected)
            result.update({"source_rows": len(records), "status": "ok"})
            return result
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--replace", action="store_true", help="Delete the existing dedicated graph before loading")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        records, expected = process_source(args.source)
        print({"status": "ok", "source_rows": len(records), "expected_nodes": expected})
        return
    print(ingest(args.source, args.batch_size, args.replace))


if __name__ == "__main__":
    main()
