import argparse
from pathlib import Path
import pandas as pd
from ai import DATABASE, DEEP_MODEL, FAST_MODEL, ask, neo4j_driver

OVERVIEW = """
MATCH (n:NewsItem)-[:ABOUT]->(c:Company)
RETURN count(n) AS news, count(DISTINCT c) AS companies, min(n.news_date).year AS first_year,
       max(n.news_date).year AS last_year, avg(n.impact_score) AS avg_impact
"""

TOPICS = "MATCH (t:Topic) RETURN t.name AS topic, t.size AS news, t.avg_impact AS avg_impact ORDER BY t.size DESC"

TOPIC_YEARS = """
MATCH (n:NewsItem)-[:IN_TOPIC]->(t:Topic), (n)-[:PUBLISHED_IN]->(m:Month)
RETURN t.name AS topic, m.year AS year, count(n) AS news
"""

LINKS = """
MATCH (a:Company)-[r:RELATED_TO]->(b:Company)
RETURN a.name AS first, b.name AS second, r.weight AS similarity, r.co_movement AS co_movement
ORDER BY r.weight DESC LIMIT 8
"""

HUBS = """
MATCH (c:Company)-[r:RELATED_TO]-()
RETURN c.name AS company, count(r) AS links ORDER BY links DESC LIMIT 5
"""

PANEL = """
MATCH (n:NewsItem)-[:ABOUT]->(c:Company), (n)-[:PUBLISHED_IN]->(m:Month), (n)-[:IN_TOPIC]->(t:Topic)
RETURN c.name AS company, m.label AS month, t.name AS topic, count(n) AS news
"""

MONTHLY = """
MATCH (c:Company)-[r:MONTHLY_SENTIMENT]->(m:Month)
RETURN c.name AS company, m.label AS month, r.avg_impact AS impact
"""

SHOCKS = """
MATCH (c:Company)-[s:HAD_SHOCK]->(m:Month)
RETURN c.name AS company, m.label AS month, s.direction AS direction, s.z AS z
ORDER BY abs(s.z) DESC LIMIT 8
"""

SYSTEM_PROMPT = (
    "You are a financial data analyst. Using ONLY the facts below, write 6 specific insights. "
    "Each insight must quote at least one number from the facts. Prefer surprising findings over obvious ones. "
    "Do not use outside knowledge, and say when evidence is weak."
)


def table(rows, **formats):
    frame = pd.DataFrame(rows)
    for column, pattern in formats.items():
        frame[column] = frame[column].map(pattern.format)
    return frame.to_string(index=False)


def topic_trends(session):
    frame = pd.DataFrame(session.run(TOPIC_YEARS).data())
    share = frame.pivot_table(index="topic", columns="year", values="news", aggfunc="sum", fill_value=0)
    share = share / share.sum(axis=0)
    years = list(share.columns)
    edge = max(1, len(years) // 3)
    change = (share[years[-edge:]].mean(axis=1) - share[years[:edge]].mean(axis=1)) * 100
    change = change.sort_values()
    label = f"{years[0]}-{years[edge - 1]} vs {years[-edge]}-{years[-1]}"
    ranked = pd.concat([change.head(3), change.tail(3)])
    return f"Change in share of coverage, {label} (percentage points):\n" + ranked.round(2).to_string()


def leading_indicators(session):
    """Does heavy coverage of a topic this month predict next month's sentiment?"""
    panel = pd.DataFrame(session.run(PANEL).data())
    monthly = pd.DataFrame(session.run(MONTHLY).data())
    monthly["period"] = pd.PeriodIndex(monthly["month"], freq="M")
    lookup = monthly.set_index(["company", "period"])["impact"]
    monthly["next_impact"] = [lookup.get((c, p + 1)) for c, p in zip(monthly["company"], monthly["period"])]

    share = panel.pivot_table(index=["company", "month"], columns="topic", values="news", fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).reset_index()
    data = share.merge(monthly, on=["company", "month"]).dropna(subset=["next_impact"])
    baseline = data["next_impact"].mean()

    rows = []
    for topic in panel["topic"].unique():
        heavy = data[data[topic] >= data[topic][data[topic] > 0].quantile(0.9)]
        if len(heavy) >= len(data) * 0.02:
            rows.append({"topic": topic, "months": len(heavy), "next_month_impact": heavy["next_impact"].mean() - baseline})
    result = pd.DataFrame(rows).sort_values("next_month_impact")
    result = pd.concat([result.head(3), result.tail(3)])
    result["next_month_impact"] = result["next_month_impact"].map("{:+.3f}".format)
    return (f"Next-month sentiment after a topic's heaviest coverage months, vs baseline {baseline:+.3f}:\n"
            + result.to_string(index=False))


def collect_facts(session):
    overview = session.run(OVERVIEW).single()
    return {
        "Overview": (f"{overview['news']:,} news items about {overview['companies']} companies, "
                     f"{overview['first_year']}-{overview['last_year']}, average impact {overview['avg_impact']:+.3f}"),
        "Topics found by clustering embeddings (size, average impact)": table(
            session.run(TOPICS).data()[:12], avg_impact="{:+.2f}"),
        "Topic trends": topic_trends(session),
        "Leading indicators": leading_indicators(session),
        "Most similar company pairs": table(
            session.run(LINKS).data(), similarity="{:.3f}", co_movement=lambda v: "n/a" if pd.isna(v) else f"{v:+.2f}"),
        "Most connected companies": table(session.run(HUBS).data()),
        "Strongest sentiment shocks": table(session.run(SHOCKS).data(), z="{:+.2f}"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="use the faster model instead of deepseek")
    args = parser.parse_args()

    driver = neo4j_driver()
    with driver.session(database=DATABASE) as session:
        facts = collect_facts(session)
    driver.close()

    facts_text = "\n\n".join(f"## {title}\n{body}" for title, body in facts.items())
    model = FAST_MODEL if args.fast else DEEP_MODEL
    insights = ask(facts_text, model=model, system=SYSTEM_PROMPT)

    report = f"# DAX40 knowledge graph insights\n\n## AI insights ({model})\n{insights}\n\n# Supporting facts\n\n{facts_text}\n"
    Path("insights.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
