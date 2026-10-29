MATCH (n) RETURN labels(n) AS labels, count(*) AS nodes ORDER BY nodes DESC;
MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS relationships ORDER BY relationships DESC;
MATCH (n:NewsItem) RETURN count(n) AS news_items, min(n.news_date) AS earliest, max(n.news_date) AS latest,
  round(avg(n.impact_score), 4) AS mean_impact, round(avg(n.confidence), 4) AS mean_confidence;
MATCH (n:NewsItem)-[:HAS_SENTIMENT]->(s:Sentiment)
RETURN s.name AS sentiment, count(*) AS articles ORDER BY articles DESC;
MATCH (n:NewsItem)-[:ABOUT]->(c:Company)
RETURN c.name AS company, count(*) AS articles, round(avg(n.impact_score), 3) AS mean_impact
ORDER BY articles DESC LIMIT 10;

