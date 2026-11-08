"""Ask the DAX40 knowledge graph anything. The AI writes every Cypher query.

  lookup       one query for counts, rankings and lists
  investigate  for "why" questions: the model plans several queries, runs them, adds semantically
               similar headlines as evidence, then explains using only what came back

A query that fails or returns nothing is sent back to the model with the problem so it can repair it.

Usage: python app.py "Why did Volkswagen's sentiment fall in 2020?" [--mode auto|lookup|investigate] [--deep]
"""

import argparse
import json
import os
import re
from collections import defaultdict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ai import CYPHER_MODEL, DATABASE, DEEP_MODEL, FAST_MODEL, VECTOR_INDEX, ask, embed, neo4j_driver, strip_thinking

MAX_REPAIRS = 3
FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|GRANT|DENY|REVOKE|ALTER|TERMINATE|CALL)\b",
    re.IGNORECASE,
)

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"mode": {"enum": ["lookup", "investigate"]}, "reason": {"type": "string"}},
    "required": ["mode", "reason"],
}
QUERY_SCHEMA = {
    "type": "object",
    "properties": {"cypher": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["cypher", "explanation"],
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {
        "type": "object",
        "properties": {"purpose": {"type": "string"}, "cypher": {"type": "string"}},
        "required": ["purpose", "cypher"],
    }}},
    "required": ["queries"],
}

QUERY_SYSTEM = (
    "Write one read-only Neo4j Cypher query that answers the question. Use only the schema and the real "
    "values below, copying names exactly. Always alias returned columns. Do not use CALL or write clauses."
)
PLAN_SYSTEM = (
    "You are investigating why something happened in a news-sentiment knowledge graph. Write 3 independent "
    "read-only Cypher queries that gather evidence: how big the change was compared with normal, which news "
    "and topics surrounded it, and related companies or similar past events. Use the shock, monthly "
    "sentiment, topic and related-company relationships from the schema when they are relevant. "
    "Use only the schema and the real values below, copy names exactly, alias every returned column, "
    "and do not use CALL or write clauses."
)

NEWS_SEARCH = """
CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node AS n, score
MATCH (n)-[:ABOUT]->(c:Company), (n)-[:PUBLISHED_IN]->(m:Month)
OPTIONAL MATCH (n)-[:IN_TOPIC]->(t:Topic)
RETURN m.label AS month, c.name AS company, n.impact_score AS impact, t.name AS topic,
       n.title AS title, n.sentiment_reason AS reason
ORDER BY score DESC
"""

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Explain why the event happened using ONLY the evidence. Cite evidence as [Q1], [Q2] or [S]. "
               "Use three short sections: Likely causes, Supporting evidence, What is uncertain. "
               "Say plainly that this is attribution from news data, not proof of cause."),
    ("human", "Question: {question}\n\nEvidence:\n{evidence}"),
])


def live_schema(session):
    """Read the real schema from Neo4j so the prompt never goes stale."""
    properties = defaultdict(set)
    for row in session.run("CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName "
                           "RETURN nodeLabels, propertyName"):
        if row["propertyName"] and row["propertyName"] != "embedding":
            properties[row["nodeLabels"][0]].add(row["propertyName"])
    nodes = [f"{label}({', '.join(sorted(names))})" for label, names in sorted(properties.items())]

    def label(node):
        return next(iter(node.labels), node.get("name", "?"))

    relationships = session.run("CALL db.schema.visualization() YIELD relationships "
                                "RETURN relationships").single()["relationships"]
    allowed = {(label(r.start_node), r.type, label(r.end_node)) for r in relationships}
    patterns = sorted(f"(:{a})-[:{t}]->(:{b})" for a, t, b in allowed)
    text = "Nodes:\n" + "\n".join(nodes) + "\nRelationships (arrows show the only valid direction):\n" + "\n".join(patterns)
    return {"context": text, "allowed": allowed, "properties": properties}


def real_values(session):
    """Real names from the graph, so the model copies exact spellings instead of guessing."""
    lines = []
    for label, limit in (("Company", 50), ("Topic", 60), ("Sentiment", 5), ("Horizon", 5)):
        values = session.run(f"MATCH (n:{label}) RETURN n.name AS name ORDER BY name LIMIT {limit}").value()
        if values:
            lines.append(f"{label}.name values: {values}")
    span = session.run("MATCH (m:Month) RETURN min(m.label) AS first, max(m.label) AS last").single()
    lines.append(f"Month.label runs from {span['first']} to {span['last']} (format YYYY-MM)")
    return "\n".join(lines)


def route(question):
    return ask(
        f"Question: {question}\n\nChoose 'lookup' for counts, rankings, averages and lists. "
        "Choose 'investigate' for why / what caused / what happened questions.",
        schema=ROUTE_SCHEMA,
    )["mode"]


NODE = re.compile(r"\((\w*)(?::(\w+))?")
EDGE = re.compile(r"\(([^()]*)\)\s*(<?)-\s*\[([^\]]*)\]\s*-(>?)\s*(?=\(([^()]*)\))")


def check_patterns(query, graph):
    """Compare every relationship pattern and property in the query with the live schema."""
    labels = {var: label for var, label in NODE.findall(query) if var and label}

    def label_of(node_text):
        var, label = NODE.match("(" + node_text).groups()
        return label or labels.get(var)

    problems = []
    for start_text, left, body, right, end_text in EDGE.findall(query):
        found = re.search(r":([A-Za-z_|:]+)", body)
        start, end = label_of(start_text), label_of(end_text)
        if left and not right:
            start, end = end, start
        orientations = [(start, end), (end, start)] if not left and not right else [(start, end)]
        for rel in re.split(r"[|:]", found.group(1)) if found else []:
            options = [(s, e) for s, t, e in graph["allowed"] if t == rel]
            if not options:
                problems.append(f"relationship type {rel} does not exist")
            elif not any((a is None or a == s) and (b is None or b == e)
                         for a, b in orientations for s, e in options):
                valid = ", ".join(f"(:{s})-[:{rel}]->(:{e})" for s, e in options)
                problems.append(f"(:{start})-[:{rel}]->(:{end}) is not a valid pattern; the only valid one is {valid}")
    for var, prop in re.findall(r"\b(\w+)\.(\w+)\b", query):
        label = labels.get(var)
        if label in graph["properties"] and prop not in graph["properties"][label]:
            problems.append(f"{label} has no property '{prop}'; it has {sorted(graph['properties'][label])}")
    return sorted(set(problems))


def validate_cypher(cypher, graph):
    query = cypher.strip().rstrip(";")
    if ";" in query or FORBIDDEN.search(query):
        raise ValueError("only a single read-only query is allowed")
    if not re.search(r"\bMATCH\b", query, re.I) or not re.search(r"\bRETURN\b", query, re.I):
        raise ValueError("query needs MATCH and RETURN")
    problems = check_patterns(query, graph)
    if problems:
        raise ValueError("; ".join(problems))
    return query if re.search(r"\bLIMIT\s+\d+", query, re.I) else query + "\nLIMIT 50"


def run_with_repair(session, question, purpose, cypher, graph):
    """Run a query. If it is invalid, errors or returns nothing, show the problem to the model and retry."""
    problem = ""
    for attempt in range(1, MAX_REPAIRS + 2):
        query = cypher
        try:
            query = validate_cypher(cypher, graph)
            rows = session.execute_read(lambda tx: tx.run(query).data())
            if rows:
                return {"purpose": purpose, "cypher": query, "rows": rows[:30], "attempts": attempt}
            problem = "The query ran but returned no rows. Check spellings against the real values and loosen filters."
        except Exception as error:
            problem = str(error)
        if attempt > MAX_REPAIRS:
            break
        fix = ask(f"Question: {question}\nPurpose: {purpose}\nFailed query:\n{query}\nProblem: {problem}\n"
                  "Write a corrected query.", model=CYPHER_MODEL, schema=QUERY_SCHEMA,
                  system=QUERY_SYSTEM + "\n\n" + graph["context"])
        cypher = fix["cypher"]
    return {"purpose": purpose, "cypher": query, "rows": [], "attempts": attempt, "error": problem}


def lookup(session, question, model, graph):
    first = ask(question, model=CYPHER_MODEL, schema=QUERY_SCHEMA, system=QUERY_SYSTEM + "\n\n" + graph["context"])
    result = run_with_repair(session, question, first["explanation"], first["cypher"], graph)
    if result["rows"]:
        answer = ask(f"Question: {question}\nRows: {json.dumps(result['rows'][:30], default=str)}\n"
                     "Answer in 2-3 sentences using only these rows.", model=model)
    else:
        answer = f"No data found after {result['attempts']} attempts: {result.get('error', '')}"
    return {"mode": "lookup", "answer": answer, "queries": [result]}


def similar_headlines(session, question, k=10):
    vector = embed([question], kind="search_query")[0].tolist()
    return session.run(NEWS_SEARCH, index=VECTOR_INDEX, k=k, vector=vector).data()


def investigate(session, question, model, graph):
    plan = ask(question, model=CYPHER_MODEL, schema=PLAN_SCHEMA, system=PLAN_SYSTEM + "\n\n" + graph["context"])
    results = [run_with_repair(session, question, q["purpose"], q["cypher"], graph) for q in plan["queries"][:4]]

    blocks = [f"[Q{i}] {r['purpose']}\n{json.dumps(r['rows'][:15], default=str)}" for i, r in enumerate(results, 1)]
    headlines = similar_headlines(session, question)
    blocks.append("[S] Semantically similar headlines:\n" + "\n".join(
        f"- {h['month']} {h['company']} ({h['impact']:+.1f}, {h['topic']}): {h['title']} -- {h['reason']}"
        for h in headlines))

    chain = EXPLAIN_PROMPT | ChatOllama(
        model=model, temperature=0, base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ) | StrOutputParser()
    answer = strip_thinking(chain.invoke({"question": question, "evidence": "\n\n".join(blocks)}))
    valid = {f"Q{i}" for i in range(1, len(results) + 1)}
    answer = re.sub(r"\[(Q\d+)\]", lambda m: m.group(0) if m.group(1) in valid else "[?]", answer)
    return {"mode": "investigate", "answer": answer, "queries": results, "similar_headlines": headlines}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", choices=["auto", "lookup", "investigate"], default="auto")
    parser.add_argument("--deep", action="store_true", help="use deepseek to write the final answer")
    args = parser.parse_args()
    question = " ".join(args.question)
    model = DEEP_MODEL if args.deep else FAST_MODEL

    driver = neo4j_driver()
    with driver.session(database=DATABASE) as session:
        graph = live_schema(session)
        graph["context"] += "\n\n" + real_values(session)
        mode = route(question) if args.mode == "auto" else args.mode
        run = investigate if mode == "investigate" else lookup
        result = run(session, question, model, graph)
    driver.close()
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
