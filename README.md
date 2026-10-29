# DAX40 Neo4j knowledge graph

This is a lossless Neo4j-ready graph derived from `dax40_news_sentiment_labeled.csv` and adapted from the local Graph-Vector project's Neo4j 5.12 configuration.

## Graph model

`(Dataset)-[:CONTAINS]->(NewsItem)-[:ABOUT]->(Company)`

Each `NewsItem` also has one `HAS_SENTIMENT`, one `HAS_HORIZON`, and one `PUBLISHED_IN` relationship. The complete headline, timestamp, impact score, confidence, rationale, and original CSV row number remain properties of the news node.

## Load into Neo4j

1. Start Docker Desktop.
2. In PowerShell, run `./load.ps1` from this directory.
3. Open <http://localhost:7474> and sign in with `neo4j` / `12345678`.

The loader is idempotent: constraints plus `MERGE` prevent duplicate graph elements if it is run again.

## Explore

```cypher
MATCH (c:Company)<-[:ABOUT]-(n:NewsItem)-[:HAS_SENTIMENT]->(s:Sentiment)
RETURN c, n, s LIMIT 100;
```

```cypher
MATCH (n:NewsItem)-[:ABOUT]->(c:Company)
RETURN c.name, count(*) AS stories, round(avg(n.impact_score), 3) AS mean_impact
ORDER BY stories DESC;
```

```cypher
MATCH (n:NewsItem)-[:ABOUT]->(c:Company), (n)-[:HAS_SENTIMENT]->(s:Sentiment)
WHERE n.news_date >= datetime('2025-01-01T00:00:00Z')
RETURN c.name, s.name, count(*) AS stories
ORDER BY stories DESC LIMIT 25;
```

The generated CSV files use Neo4j bulk-import-style typed headers and can also be reused with `neo4j-admin database import` if desired. `analysis.json` contains the independently computed validation totals.

