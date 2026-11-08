# DAX40 Knowledge Graph

A Neo4j graph of 42,000+ DAX40 news headlines that uses local AI models (Ollama) to find
topics, connections and the probable reasons behind sentiment shocks. Nothing is hand-labelled:
topics, thresholds and relationships all come from the dataset.

## How it works

1. `ingestion.py` validates the CSV and loads the base graph.
2. `enrich.py` embeds every headline with `nomic-embed-text`, then
   - clusters the embeddings into topics (the number of topics is chosen automatically, and `llama3.2` names them)
   - links similar headlines and related companies using data-driven thresholds
   - flags shocks: company-months whose sentiment is a statistical outlier
3. `catalyst.py` explains a shock. It collects the company's news and its related companies' news,
   scores each story, and asks `deepseek-r1` for the likely drivers and what similar past events led to.
   `catalyst.py train` replays history in time order to learn which topics matter for each company.
4. `insights.py` mines the graph for trends and leading indicators and has a model write the findings.
5. `app.py` answers questions, and the AI writes every Cypher query. Lookups ("top 5 companies...") use one
   query. "Why" questions use `investigate` mode: the model plans several queries and adds similar headlines
   from the vector index as evidence. Each query is checked against the live schema, and invalid, failing or
   empty queries go back to the model for repair.

## Setup

    pip install -r requirements.txt
    ollama pull nomic-embed-text
    ollama pull llama3.2
    ollama pull deepseek-r1:8b
    ollama pull qwen2.5:3b

Neo4j 5.11 or newer is needed for the vector index. Set the connection first:

    set NEO4J_PASSWORD=your-password

## Run

    python ingestion.py --replace
    python enrich.py                      # embeds 42k headlines, resumes if interrupted
    python catalyst.py train
    python catalyst.py shocks --ticker siemens
    python catalyst.py explain siemens 2020-03
    python insights.py
    python app.py "Which topics dominate negative news at Siemens?"

Models can be changed with the environment variables `EMBED_MODEL`, `FAST_MODEL`, `DEEP_MODEL` and
`CYPHER_MODEL`. Small models write unreliable Cypher, so a larger one for `CYPHER_MODEL` helps.

Headlines are lowercased before embedding because this Ollama build maps capitalised words to unknown
tokens, which made different Title Case headlines of equal length produce identical vectors.

## Notes

Explanations are statistical attribution from news timing, not proof of cause. Shocks are measured
on dataset sentiment because the dataset has no prices.
