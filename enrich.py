import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from ai import DATABASE, VECTOR_INDEX, ask, embed, neo4j_driver

CACHE = Path(__file__).parent / ".cache" / "embeddings.npz"
BATCH = 64

SCHEMA = [
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT quarter_id IF NOT EXISTS FOR (n:Quarter) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT year_id IF NOT EXISTS FOR (n:Year) REQUIRE n.id IS UNIQUE",
    "CREATE FULLTEXT INDEX news_text IF NOT EXISTS FOR (n:NewsItem) ON EACH [n.title, n.sentiment_reason]",
]

TIME_TREE = """
MATCH (m:Month)
MERGE (y:Year {id: 'year:' + toString(m.year)}) SET y.year = m.year
MERGE (q:Quarter {id: 'quarter:' + toString(m.year) + '-Q' + toString((m.month - 1) / 3 + 1)})
  SET q.year = m.year, q.quarter = (m.month - 1) / 3 + 1
MERGE (m)-[:IN_QUARTER]->(q)
MERGE (q)-[:IN_YEAR]->(y)
"""

MONTH_CHAIN = """
MATCH (m:Month) WITH m ORDER BY m.label
WITH collect(m) AS months
UNWIND range(0, size(months) - 2) AS i
WITH months[i] AS a, months[i + 1] AS b
MERGE (a)-[:NEXT]->(b)
"""

NEWS_CHAIN = """
MATCH (n:NewsItem)-[:ABOUT]->(c:Company)
WITH c, n ORDER BY n.news_date, n.id
WITH c, collect(n) AS items
WHERE size(items) > 1
UNWIND range(0, size(items) - 2) AS i
WITH items[i] AS a, items[i + 1] AS b
MERGE (a)-[r:NEXT_FOR_COMPANY]->(b)
SET r.gap_days = duration.inDays(a.news_date, b.news_date).days
"""

MONTHLY_SENTIMENT = """
MATCH (n:NewsItem)-[:ABOUT]->(c:Company), (n)-[:PUBLISHED_IN]->(m:Month)
WITH c, m, avg(n.impact_score) AS avg_impact, count(n) AS news_count
MERGE (c)-[r:MONTHLY_SENTIMENT]->(m)
SET r.avg_impact = avg_impact, r.news_count = news_count
"""

READ_NEWS = """
MATCH (n:NewsItem)-[:ABOUT]->(c:Company)
RETURN n.id AS id, n.title AS title, n.impact_score AS impact, c.id AS company, c.name AS ticker
ORDER BY n.id
"""

READ_MONTHLY = """
MATCH (c:Company)-[r:MONTHLY_SENTIMENT]->(m:Month)
RETURN c.id AS company, c.name AS ticker, m.id AS month_id, m.label AS month,
       r.avg_impact AS avg_impact, r.news_count AS news_count
"""

CLEAR_DERIVED = [
    "MATCH (t:Topic) DETACH DELETE t",
    "MATCH ()-[r:SIMILAR_TO|RELATED_TO|HAD_SHOCK]->() DELETE r",
]

WRITE_VECTORS = """
UNWIND $rows AS row
MATCH (n:NewsItem {id: row.id})
CALL db.create.setNodeVectorProperty(n, 'embedding', row.vector)
"""

WRITE_TOPICS = """
UNWIND $rows AS row
MERGE (t:Topic {id: row.id})
SET t.name = row.name, t.description = row.description, t.size = row.size, t.avg_impact = row.avg_impact
"""

WRITE_MEMBERSHIP = """
UNWIND $rows AS row
MATCH (n:NewsItem {id: row.news_id}), (t:Topic {id: row.topic_id})
MERGE (n)-[r:IN_TOPIC]->(t)
SET r.similarity = row.similarity
"""

WRITE_SIMILAR = """
UNWIND $rows AS row
MATCH (a:NewsItem {id: row.a}), (b:NewsItem {id: row.b})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.score = row.score
"""

WRITE_RELATED = """
UNWIND $rows AS row
MATCH (a:Company {id: row.a}), (b:Company {id: row.b})
MERGE (a)-[r:RELATED_TO]->(b)
SET r.weight = row.weight, r.co_movement = row.co_movement
"""

WRITE_SHOCKS = """
UNWIND $rows AS row
MATCH (c:Company {id: row.company}), (m:Month {id: row.month_id})
MERGE (c)-[r:HAD_SHOCK]->(m)
SET r.z = row.z, r.avg_impact = row.avg_impact, r.news_count = row.news_count, r.direction = row.direction
"""

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
    "required": ["name", "description"],
}


def write_in_batches(session, query, rows, size=2000):
    for start in range(0, len(rows), size):
        session.run(query, rows=rows[start : start + size]).consume()


# ---------- 2. embeddings ----------

def load_cache():
    if not CACHE.exists():
        return {}
    data = np.load(CACHE)
    return dict(zip(data["ids"], data["vectors"]))


def save_cache(store):
    CACHE.parent.mkdir(exist_ok=True)
    ids = list(store)
    np.savez(CACHE, ids=np.array(ids), vectors=np.stack([store[i] for i in ids]))


def prepare(title):
    """Drop the leading 'Company : ' so topics reflect what happened, not which company is named."""
    return title.split(" : ", 1)[-1].strip()


def center(vectors):
    """Remove the direction every headline shares, so similarities spread out and become meaningful."""
    shifted = vectors - vectors.mean(axis=0)
    return shifted / np.linalg.norm(shifted, axis=1, keepdims=True)


def embed_news(news):
    """Embed headlines. The cache is keyed by text, so interrupted runs resume and edits re-embed."""
    texts = [prepare(n["title"]) for n in news]
    keys = [hashlib.sha1(text.encode("utf-8")).hexdigest() for text in texts]
    store = load_cache()
    todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in store]
    print(f"Embedding {len(todo):,} unique headlines ({len(set(keys)) - len(todo):,} already cached)")
    for number, start in enumerate(range(0, len(todo), BATCH), start=1):
        chunk = todo[start : start + BATCH]
        store.update(zip((k for k, _ in chunk), embed([t for _, t in chunk])))
        if number % 50 == 0:
            save_cache(store)
            print(f"  {start + len(chunk):,}/{len(todo):,}")
    if todo:
        save_cache(store)
    return np.stack([store[k] for k in keys])


# ---------- 3. topics ----------

def find_topics(vectors):
    """Try several cluster counts (range scales with dataset size) and keep the best silhouette score."""
    n = len(vectors)
    low, high = max(2, int(n ** 0.25)), max(3, int(n ** 0.5 / 4))
    sample = np.random.default_rng(0).choice(n, min(n, 4000), replace=False)
    best_score, best_model = -1, None
    for k in np.unique(np.linspace(low, high, 6).astype(int)):
        model = MiniBatchKMeans(int(k), n_init=3, random_state=0).fit(vectors)
        score = silhouette_score(vectors[sample], model.labels_[sample])
        print(f"  k={k:<3} silhouette={score:.3f}")
        if score > best_score:
            best_score, best_model = score, model
    print(f"  chose k={best_model.n_clusters}")
    return best_model


def name_topics(news, vectors, model):
    topics, memberships = [], []
    for k, centroid in enumerate(model.cluster_centers_):
        members = np.where(model.labels_ == k)[0]
        centroid = centroid / np.linalg.norm(centroid)
        closeness = vectors[members] @ centroid
        typical = members[np.argsort(-closeness)[:15]]
        headlines = "\n".join(f"- {prepare(news[i]['title'])}" for i in typical)
        taken = ", ".join(t["name"] for t in topics) or "none yet"
        named = ask(
            f"These headlines come from one cluster of financial news:\n{headlines}\n\n"
            f"Topic names already used: {taken}.\n"
            "Give this cluster a short, specific topic name (2-4 words) that differs from those, "
            "and a one-sentence description of the event type.",
            schema=TOPIC_SCHEMA,
        )
        topic_id = f"topic:{k}"
        topics.append({
            "id": topic_id, "name": named["name"], "description": named["description"],
            "size": len(members), "avg_impact": float(np.mean([news[i]["impact"] for i in members])),
        })
        memberships += [
            {"news_id": news[i]["id"], "topic_id": topic_id, "similarity": float(s)}
            for i, s in zip(members, closeness)
        ]
        print(f"  {topic_id:<9} {len(members):>6,} headlines  {named['name']}")
    return topics, memberships


# ---------- 4. similarity and company links ----------

def nearest_neighbors(vectors, k, chunk=1000):
    n = len(vectors)
    neighbor_ids = np.zeros((n, k), dtype=int)
    neighbor_sims = np.zeros((n, k), dtype="float32")
    for start in range(0, n, chunk):
        block = vectors[start : start + chunk] @ vectors.T
        rows = np.arange(len(block))
        block[rows, start + rows] = -1
        top = np.argpartition(-block, k, axis=1)[:, :k]
        neighbor_ids[start : start + chunk] = top
        neighbor_sims[start : start + chunk] = np.take_along_axis(block, top, axis=1)
    return neighbor_ids, neighbor_sims


def similar_news_edges(news, vectors):
    k = max(3, int(np.log(len(news))))
    ids, sims = nearest_neighbors(vectors, k)
    cutoff = np.percentile(sims, 90)
    print(f"  k={k}, keeping neighbours with similarity >= {cutoff:.3f} (top 10% of all)")
    return [
        {"a": news[i]["id"], "b": news[j]["id"], "score": float(s)}
        for i in range(len(news))
        for j, s in zip(ids[i], sims[i])
        if s >= cutoff
    ]


def company_edges(news, vectors, monthly):
    companies = sorted({n["company"] for n in news})
    centroids = []
    for company in companies:
        rows = [i for i, n in enumerate(news) if n["company"] == company]
        centroid = vectors[rows].mean(axis=0)
        centroids.append(centroid / np.linalg.norm(centroid))
    similarity = np.array(centroids) @ np.array(centroids).T

    upper = similarity[np.triu_indices(len(companies), 1)]
    cutoff = upper.mean() + upper.std()
    print(f"  keeping company pairs with similarity >= {cutoff:.3f} (mean + 1 std)")

    pivot = monthly.pivot(index="month", columns="company", values="avg_impact")
    correlation = pivot.corr(min_periods=max(3, int(pivot.count().median() / 4)))
    edges = []
    for i, a in enumerate(companies):
        for j in range(i + 1, len(companies)):
            if similarity[i, j] >= cutoff:
                b = companies[j]
                value = correlation.at[a, b] if a in correlation.index and b in correlation.columns else np.nan
                edges.append({
                    "a": a, "b": b, "weight": float(similarity[i, j]),
                    "co_movement": None if pd.isna(value) else float(value),
                })
    return edges


# ---------- 5. shocks ----------

def find_shocks(monthly):
    """A shock is a well-covered company-month whose sentiment is unusually far from that company's norm."""
    stats = monthly.groupby("company")["avg_impact"].agg(["mean", "std"])
    frame = monthly.join(stats, on="company")
    frame["z"] = (frame["avg_impact"] - frame["mean"]) / frame["std"]
    frame = frame[(frame["news_count"] >= frame["news_count"].median()) & frame["z"].notna()]
    cutoff = frame["z"].abs().quantile(0.95)
    print(f"  flagging |z| >= {cutoff:.2f} (top 5% of well-covered company-months)")
    shocks = frame[frame["z"].abs() >= cutoff].copy()
    shocks["direction"] = np.where(shocks["z"] > 0, "gain", "loss")
    return shocks[["company", "month_id", "z", "avg_impact", "news_count", "direction"]].to_dict("records")


# ---------- run ----------

def main():
    driver = neo4j_driver()
    driver.verify_connectivity()
    with driver.session(database=DATABASE) as session:
        print("1. Time tree, news chains, monthly sentiment")
        for statement in SCHEMA + [TIME_TREE, MONTH_CHAIN, NEWS_CHAIN, MONTHLY_SENTIMENT] + CLEAR_DERIVED:
            session.run(statement).consume()
        news = session.run(READ_NEWS).data()
        monthly = pd.DataFrame(session.run(READ_MONTHLY).data())

        print("2. Embeddings")
        vectors = embed_news(news)
        rows = [{"id": n["id"], "vector": v.tolist()} for n, v in zip(news, vectors)]
        write_in_batches(session, WRITE_VECTORS, rows, size=500)
        session.run(
            f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (n:NewsItem) ON n.embedding "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {vectors.shape[1]}, "
            "`vector.similarity_function`: 'cosine'}}"
        ).consume()

        print("3. Topics")
        centered = center(vectors)
        topics, memberships = name_topics(news, centered, find_topics(centered))
        write_in_batches(session, WRITE_TOPICS, topics)
        write_in_batches(session, WRITE_MEMBERSHIP, memberships)

        print("4. Similar headlines and related companies")
        write_in_batches(session, WRITE_SIMILAR, similar_news_edges(news, centered))
        write_in_batches(session, WRITE_RELATED, company_edges(news, centered, monthly))

        print("5. Shocks")
        write_in_batches(session, WRITE_SHOCKS, find_shocks(monthly))

        counts = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC")
        print("\nRelationships in the graph:")
        for row in counts:
            print(f"  {row['type']:<20} {row['count']:,}")
    driver.close()


if __name__ == "__main__":
    main()
