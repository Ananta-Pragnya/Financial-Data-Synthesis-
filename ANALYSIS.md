# Dataset analysis

## Scope and quality

- 42,814 labeled headlines covering 40 DAX40 constituents.
- Publication range: 6 February 2017 through 17 February 2026 (UTC).
- No missing fields and no invalid sentiment, horizon, score, confidence, or timestamp values were found.
- 95 rows are exact duplicates. They are retained as separate `NewsItem` nodes and remain traceable through `source_row`.
- The labels are model-generated “silver” labels, not human-verified ground truth. The source README warns that the sentiment factor did not produce statistically significant tradable alpha.

## Label profile

| Sentiment | Rows | Share |
|---|---:|---:|
| Positive | 18,043 | 42.14% |
| Negative | 15,053 | 35.16% |
| Neutral | 9,718 | 22.70% |

Mean impact score is **0.1027**, indicating a mildly positive class/impact skew. Mean annotator confidence is **2.5733 / 3**.

| Expected horizon | Rows | Share |
|---|---:|---:|
| 0–1 month | 32,547 | 76.02% |
| 1–3 months | 7,365 | 17.20% |
| 3–6 months | 2,902 | 6.78% |

The graph is therefore best suited to short-horizon event and coverage analysis; long-horizon examples are comparatively sparse.

## Coverage profile

Coverage rises sharply in recent years: 10,435 rows are from 2025 alone, versus 1,431 in 2017. The most represented companies are Airbus (5,512), Volkswagen (4,654), Deutsche Bank (2,587), Mercedes-Benz Group (2,244), and Commerzbank (2,193). This imbalance matters for cross-company comparisons; raw counts should generally be normalized by company coverage and time period.

## Graph totals

| Element | Count |
|---|---:|
| Dataset nodes | 1 |
| Company nodes | 40 |
| Sentiment nodes | 3 |
| Horizon nodes | 3 |
| Month nodes | 108 |
| NewsItem nodes | 42,814 |
| **All nodes** | **42,969** |
| ABOUT relationships | 42,814 |
| HAS_SENTIMENT relationships | 42,814 |
| HAS_HORIZON relationships | 42,814 |
| PUBLISHED_IN relationships | 42,814 |
| CONTAINS relationships | 42,814 |
| **All relationships** | **214,070** |

The schema keeps categorical concepts as shared nodes, which makes traversals and group comparisons natural in Neo4j while retaining full source-row fidelity.

