import json
import os
import re

import numpy as np
from neo4j import GraphDatabase
from ollama import Client

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
FAST_MODEL = os.getenv("FAST_MODEL", "llama3.2")
DEEP_MODEL = os.getenv("DEEP_MODEL", "deepseek-r1:8b")
CYPHER_MODEL = os.getenv("CYPHER_MODEL", "qwen2.5:3b")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
VECTOR_INDEX = "news_embeddings"

ollama = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))


def neo4j_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )


def embed(texts, kind="search_document"):
    
    prefixed = [f"{kind}: {text.lower()}" for text in texts]
    vectors = np.array(ollama.embed(model=EMBED_MODEL, input=prefixed).embeddings, dtype="float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask(prompt, model=FAST_MODEL, system=None, schema=None):
    """Ask a local model. With a JSON schema the answer comes back as a dict."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    reply = ollama.chat(model=model, messages=messages, format=schema, options={"temperature": 0})
    text = strip_thinking(reply.message.content)
    return json.loads(text) if schema else text
