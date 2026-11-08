"""Catalyst: find the probable reasons behind a sentiment shock, using only the dataset.

A shock is a company-month whose sentiment is a statistical outlier (found by enrich.py).
Catalyst walks the graph around it, scores the news, and asks a reasoning model to explain it.

Usage:
  python catalyst.py shocks --ticker siemens     list the biggest shocks
  python catalyst.py explain siemens 2020-03     explain one month
  python catalyst.py train                       learn which topics move each company
"""

import argparse
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from ai import DATABASE, DEEP_MODEL, FAST_MODEL, ask, neo4j_driver

WINDOW = """n.news_date >= datetime($month + '-01T00:00:00Z') - duration({days: $lookback})
            AND n.news_date < datetime($month + '-01T00:00:00Z') + duration({months: 1})"""

OWN_NEWS = f"""
MATCH (n:NewsItem)-[:ABOUT]->(c:Company {{id: $company}})
WHERE {WINDOW}
MATCH (n)-[:HAS_HORIZON]->(h:Horizon)
OPTIONAL MATCH (n)-[:IN_TOPIC]->(t:Topic)
RETURN n.id AS id, n.news_date AS date, n.title AS title, c.name AS ticker, n.impact_score AS impact,
       n.confidence AS confidence, h.max_months AS horizon, coalesce(t.id, 'topic:none') AS topic_id,
       coalesce(t.name, 'other') AS topic, 1.0 AS link, true AS own
"""

RELATED_NEWS = f"""
MATCH (:Company {{id: $company}})-[r:RELATED_TO]-(o:Company)
MATCH (n:NewsItem)-[:ABOUT]->(o)
WHERE {WINDOW}
MATCH (n)-[:HAS_HORIZON]->(h:Horizon)
OPTIONAL MATCH (n)-[:IN_TOPIC]->(t:Topic)
RETURN n.id AS id, n.news_date AS date, n.title AS title, o.name AS ticker, n.impact_score AS impact,
       n.confidence AS confidence, h.max_months AS horizon, coalesce(t.id, 'topic:none') AS topic_id,
       coalesce(t.name, 'other') AS topic, r.weight AS link, false AS own
"""

MEDIAN_HORIZON = """
MATCH (:Company {id: $company})<-[:ABOUT]-(:NewsItem)-[:HAS_HORIZON]->(h:Horizon)
RETURN percentileCont(h.max_months, 0.5) AS months
"""

LINK_RANGE = "MATCH ()-[r:RELATED_TO]->() RETURN min(r.weight) AS lowest"

MONTH_FACTS = """
MATCH (c:Company {id: $company})-[r:MONTHLY_SENTIMENT]->(m:Month {label: $month})
MATCH (c)-[all:MONTHLY_SENTIMENT]->()
RETURN r.avg_impact AS month_impact, avg(all.avg_impact) AS normal_impact, r.news_count AS news
"""

PRECEDENTS = """
MATCH (source:NewsItem {id: $id})
CALL db.index.vector.queryNodes($index, $k, source.embedding) YIELD node, score
WHERE node <> source
MATCH (node)-[:ABOUT]->(c:Company), (node)-[:PUBLISHED_IN]->(m:Month)
OPTIONAL MATCH (m)-[:NEXT]->(next:Month)<-[ms:MONTHLY_SENTIMENT]-(c)
RETURN node.title AS title, c.name AS company, m.label AS month, score, ms.avg_impact AS next_impact
"""

GLOBAL_NORM = "MATCH ()-[r:MONTHLY_SENTIMENT]->() RETURN avg(r.avg_impact) AS norm"

ALL_SHOCKS = """
MATCH (c:Company)-[s:HAD_SHOCK]->(m:Month)
RETURN c.id AS company, c.name AS ticker, m.label AS month, s.direction AS direction
ORDER BY m.label
"""

LIST_SHOCKS = """
MATCH (c:Company)-[s:HAD_SHOCK]->(m:Month)
WHERE $ticker = '' OR toLower(c.name) CONTAINS toLower($ticker)
RETURN c.name AS ticker, m.label AS month, s.direction AS direction, s.z AS z, s.news_count AS news
ORDER BY abs(s.z) DESC LIMIT $limit
"""

FIND_COMPANY = "MATCH (c:Company) WHERE toLower(c.name) CONTAINS toLower($text) RETURN c.id AS id, c.name AS name"

READ_TRUST = """
MATCH (:Company {id: $company})-[s:SENSITIVE_TO]->(t:Topic)
RETURN t.id AS topic_id, s.alpha AS alpha, s.beta AS beta
"""

WRITE_TRUST = """
UNWIND $rows AS row
MATCH (c:Company {id: row.company}), (t:Topic {id: row.topic_id})
MERGE (c)-[s:SENSITIVE_TO]->(t)
SET s.alpha = row.alpha, s.beta = row.beta, s.trust = row.alpha / (row.alpha + row.beta)
"""

SAVE_EXPLANATION = """
MATCH (c:Company {id: $company})
MERGE (x:Explanation {id: $id})
SET x.month = $month, x.direction = $direction, x.text = $text, x.model = $model
MERGE (x)-[:EXPLAINS]->(c)
WITH x
UNWIND $cited AS cite
MATCH (n:NewsItem {id: cite.id})
MERGE (x)-[r:CITES]->(n)
SET r.score = cite.score
"""

SYSTEM_PROMPT = (
    "You are a financial analyst writing a short post-mortem on why a company's news sentiment shifted. "
    "Use ONLY the evidence given. Refer to news by its tag such as [N3]. "
    "Write four short sections titled Likely drivers, Counter-signals, Historical precedent, Confidence. "
    "State plainly that this is statistical attribution from news timing, not proof of cause."
)


def month_end(month):
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    return (start + timedelta(days=32)).replace(day=1)


def find_company(session, text):
    matches = session.run(FIND_COMPANY, text=text).data()
    if not matches:
        raise SystemExit(f"No company matches '{text}'")
    if len(matches) > 1:
        raise SystemExit("Be more specific: " + ", ".join(m["name"] for m in matches))
    return matches[0]


def fetch_evidence(session, company, month, link_floor):
    horizon = session.run(MEDIAN_HORIZON, company=company).single()["months"]
    args = {"company": company, "month": month, "lookback": int(horizon * 30)}
    rows = session.run(OWN_NEWS, args).data() + session.run(RELATED_NEWS, args).data()
    for row in rows:
        row["date"] = row["date"].to_native()
        if not row["own"]:
            row["link"] = max(0.0, (row["link"] - link_floor) / (1 - link_floor))
    return [row for row in rows if row["link"] > 0]


def score_news(news, end, trust):
    days = (end - news["date"]).total_seconds() / 86400
    freshness = 0.5 ** (max(days, 0) / (news["horizon"] * 15))
    topic_trust = 2 * trust.get(news["topic_id"], 0.5)
    return news["impact"] * news["confidence"] / 3 * freshness * news["link"] * topic_trust


def attribute(evidence, month, direction, trust):
    """Group scored news by topic and rank topics by how much they pushed in the shock's direction."""
    end = month_end(month)
    for news in evidence:
        news["score"] = score_news(news, end, trust)
    topics = defaultdict(list)
    for news in evidence:
        topics[news["topic_id"]].append(news)

    total = sum(abs(n["score"]) for n in evidence if n["score"] * direction > 0) or 1
    drivers = []
    for topic_id, items in topics.items():
        pro = sorted((n for n in items if n["score"] * direction > 0), key=lambda n: -abs(n["score"]))
        con = sorted((n for n in items if n["score"] * direction < 0), key=lambda n: -abs(n["score"]))
        drivers.append({
            "topic_id": topic_id, "topic": items[0]["topic"], "net": sum(n["score"] for n in items),
            "share": sum(abs(n["score"]) for n in pro) / total, "pro": pro[:3], "con": con[:1],
        })
    return sorted(drivers, key=lambda d: -d["share"])


def read_trust(session, company):
    rows = session.run(READ_TRUST, company=company).data()
    return {r["topic_id"]: r["alpha"] / (r["alpha"] + r["beta"]) for r in rows}


def precedent_outlook(session, news_id):
    rows = session.run(PRECEDENTS, id=news_id, index="news_embeddings", k=10).data()
    outcomes = [r["next_impact"] for r in rows if r["next_impact"] is not None]
    norm = session.run(GLOBAL_NORM).single()["norm"]
    return rows, (sum(outcomes) / len(outcomes) if outcomes else None), norm


def build_brief(company, month, direction, facts, drivers, precedents, outlook, norm):
    tags, lines = {}, []
    lines.append(f"Company: {company}. Month: {month}. Shift: {'gain' if direction > 0 else 'loss'}. "
                 f"Month sentiment {facts['month_impact']:+.2f} vs the company's normal {facts['normal_impact']:+.2f} "
                 f"from {facts['news']} news items.")
    for driver in drivers[:4]:
        lines.append(f"\nTopic '{driver['topic']}' explains {driver['share']:.0%} of the push.")
        for kind, items in (("supports", driver["pro"]), ("opposes", driver["con"])):
            for news in items:
                tag = f"N{len(tags) + 1}"
                tags[tag] = news
                origin = "own news" if news["own"] else f"related company {news['ticker']}"
                lines.append(f"  [{tag}] {kind} ({origin}, impact {news['impact']:+.1f}): {news['title']}")
    lines.append("\nHistorical precedent (semantically similar past headlines):")
    for p in precedents[:5]:
        lines.append(f"  - {p['company']} {p['month']}: {p['title']}")
    if outlook is not None:
        lines.append(f"  Month after such headlines the average sentiment was {outlook:+.2f} (dataset-wide norm {norm:+.2f}).")
    return "\n".join(lines), tags


def explain(session, text, month, fast, save):
    company = find_company(session, text)
    facts = session.run(MONTH_FACTS, company=company["id"], month=month).single()
    if facts is None:
        raise SystemExit(f"No data for {company['name']} in {month}")
    direction = 1 if facts["month_impact"] >= facts["normal_impact"] else -1

    link_floor = session.run(LINK_RANGE).single()["lowest"] or 0.0
    evidence = fetch_evidence(session, company["id"], month, link_floor)
    drivers = attribute(evidence, month, direction, read_trust(session, company["id"]))
    if not drivers or not drivers[0]["pro"]:
        raise SystemExit("No supporting evidence found for this month")

    precedents, outlook, norm = precedent_outlook(session, drivers[0]["pro"][0]["id"])
    brief, tags = build_brief(company["name"], month, direction, facts, drivers, precedents, outlook, norm)
    model = FAST_MODEL if fast else DEEP_MODEL
    text = ask(brief, model=model, system=SYSTEM_PROMPT)
    text = re.sub(r"\[N\d+\]", lambda m: m.group(0) if m.group(0)[1:-1] in tags else "[?]", text)

    print(f"\n{company['name']}  {month}  ({'gain' if direction > 0 else 'loss'})\n")
    print(brief, "\n\n--- AI analysis (" + model + ") ---\n")
    print(text)
    if save:
        key = hashlib.sha1(f"{company['id']}{month}".encode()).hexdigest()[:16]
        cited = [{"id": n["id"], "score": n["score"]} for n in tags.values()]
        session.run(SAVE_EXPLANATION, company=company["id"], id=key, month=month, text=text, model=model,
                    direction="gain" if direction > 0 else "loss", cited=cited).consume()
        print("\nSaved as an Explanation node linked to the news it cites.")


def train(session):
    """Replay every historical shock in time order. Each topic's trust only uses shocks that came before it."""
    link_floor = session.run(LINK_RANGE).single()["lowest"] or 0.0
    shocks = session.run(ALL_SHOCKS).data()
    state = defaultdict(lambda: [1.0, 1.0])
    print(f"Replaying {len(shocks)} shocks in time order")
    for number, shock in enumerate(shocks, start=1):
        direction = 1 if shock["direction"] == "gain" else -1
        trust = {topic: a / (a + b) for (company, topic), (a, b) in state.items() if company == shock["company"]}
        evidence = fetch_evidence(session, shock["company"], shock["month"], link_floor)
        for driver in attribute(evidence, shock["month"], direction, trust):
            hit = driver["net"] * direction > 0
            state[(shock["company"], driver["topic_id"])][0 if hit else 1] += 1
        if number % 25 == 0:
            print(f"  {number}/{len(shocks)}")
    rows = [{"company": c, "topic_id": t, "alpha": a, "beta": b} for (c, t), (a, b) in state.items()]
    session.run(WRITE_TRUST, rows=rows).consume()
    print(f"Stored {len(rows)} SENSITIVE_TO edges (company -> topic).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    shocks = commands.add_parser("shocks")
    shocks.add_argument("--ticker", default="")
    shocks.add_argument("--limit", type=int, default=15)
    explain_cmd = commands.add_parser("explain")
    explain_cmd.add_argument("company")
    explain_cmd.add_argument("month", help="YYYY-MM")
    explain_cmd.add_argument("--fast", action="store_true", help="use the faster model instead of deepseek")
    explain_cmd.add_argument("--save", action="store_true")
    commands.add_parser("train")
    args = parser.parse_args()

    driver = neo4j_driver()
    with driver.session(database=DATABASE) as session:
        if args.command == "shocks":
            for r in session.run(LIST_SHOCKS, ticker=args.ticker, limit=args.limit):
                print(f"{r['month']}  {r['ticker']:<28} {r['direction']:<5} z={r['z']:+.2f}  ({r['news']} news)")
        elif args.command == "explain":
            explain(session, args.company, args.month, args.fast, args.save)
        else:
            train(session)
    driver.close()


if __name__ == "__main__":
    main()
