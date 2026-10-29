CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (n:Dataset) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT sentiment_id IF NOT EXISTS FOR (n:Sentiment) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT horizon_id IF NOT EXISTS FOR (n:Horizon) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT month_id IF NOT EXISTS FOR (n:Month) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT news_id IF NOT EXISTS FOR (n:NewsItem) REQUIRE n.id IS UNIQUE;
CREATE INDEX news_date IF NOT EXISTS FOR (n:NewsItem) ON (n.news_date);
CREATE INDEX news_impact IF NOT EXISTS FOR (n:NewsItem) ON (n.impact_score);
CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.name);

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///dataset.csv' AS row
  MERGE (n:Dataset {id: row.`id:ID(Dataset-ID)`})
  SET n.name = row.name, n.description = row.description, n.license = row.license,
      n.row_count = toInteger(row.`row_count:int`),
      n.start_date = datetime(row.`start_date:datetime`), n.end_date = datetime(row.`end_date:datetime`)
} IN TRANSACTIONS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///companies.csv' AS row
  MERGE (n:Company {id: row.`id:ID(Company-ID)`}) SET n.name = row.name
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///sentiments.csv' AS row
  MERGE (n:Sentiment {id: row.`id:ID(Sentiment-ID)`}) SET n.name = row.name
} IN TRANSACTIONS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///horizons.csv' AS row
  MERGE (n:Horizon {id: row.`id:ID(Horizon-ID)`})
  SET n.name = row.name, n.min_months = toInteger(row.`min_months:int`), n.max_months = toInteger(row.`max_months:int`)
} IN TRANSACTIONS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///months.csv' AS row
  MERGE (n:Month {id: row.`id:ID(Month-ID)`})
  SET n.label = row.label, n.year = toInteger(row.`year:int`), n.month = toInteger(row.`month:int`)
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///news.csv' AS row
  MERGE (n:NewsItem {id: row.`id:ID(News-ID)`})
  SET n.news_date = datetime(row.`news_date:datetime`), n.title = row.title,
      n.impact_score = toFloat(row.`impact_score:double`), n.confidence = toInteger(row.`confidence:int`),
      n.sentiment_reason = row.sentiment_reason, n.source_row = toInteger(row.`source_row:int`)
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///news_about_company.csv' AS row
  MATCH (a:NewsItem {id: row.`:START_ID(News-ID)`}), (b:Company {id: row.`:END_ID(Company-ID)`})
  MERGE (a)-[:ABOUT]->(b)
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///news_has_sentiment.csv' AS row
  MATCH (a:NewsItem {id: row.`:START_ID(News-ID)`}), (b:Sentiment {id: row.`:END_ID(Sentiment-ID)`})
  MERGE (a)-[:HAS_SENTIMENT]->(b)
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///news_has_horizon.csv' AS row
  MATCH (a:NewsItem {id: row.`:START_ID(News-ID)`}), (b:Horizon {id: row.`:END_ID(Horizon-ID)`})
  MERGE (a)-[:HAS_HORIZON]->(b)
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///news_published_in.csv' AS row
  MATCH (a:NewsItem {id: row.`:START_ID(News-ID)`}), (b:Month {id: row.`:END_ID(Month-ID)`})
  MERGE (a)-[:PUBLISHED_IN]->(b)
} IN TRANSACTIONS OF 1000 ROWS;

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///dataset_contains_news.csv' AS row
  MATCH (a:Dataset {id: row.`:START_ID(Dataset-ID)`}), (b:NewsItem {id: row.`:END_ID(News-ID)`})
  MERGE (a)-[:CONTAINS]->(b)
} IN TRANSACTIONS OF 1000 ROWS;

