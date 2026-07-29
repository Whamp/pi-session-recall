# Query-Planned Recall: Pre-Planning Hybrid Baseline

This report records current hybrid behavior before query-planned recall is implemented. Private queries and unchanged real session snapshots remain in a permission-restricted, Git-ignored local corpus; this report contains only opaque case identities, categories, policy values, counts, ranks, and checksums.

## Identity

- Recorded against commit: `38aab6722a6fc97dd212e704faee4373af8b363e`
- Private manifest SHA-256: `4b3fb8573bcd52a26d66a2f8e60f5bbba57b098530791b23f55505d3e9a1187e`
- Publishable controls canonical JSON SHA-256: `a5b300c736ddfed35e4e65635fe5752b17b53df2c61a97079229c3ff0b44d605`
- Embedding profile: `octen-embed` / `Octen/Octen-Embedding-4B`, 2560 dimensions, Q8_0, last pooling
- Frozen cases: 8 across 8 unchanged snapshots and 44521 indexed documents
- Executed searches: 16 (normal plus retrieval-work-matched original query for every case)

## Aggregate outcomes

- Normal hybrid: 2 success, 1 final rank miss, 5 candidate union miss.
- Retrieval-work-matched original query: 3 success, 1 final rank miss, 4 candidate union miss.

A candidate union miss means the exact expected source was absent after all admitted ranked-list candidates were fused. A final rank miss means the source was admitted but ranked below the fixed final-five cutoff.

## Cases

| Case     | Category                        | Role                        | Normal outcome       | Normal rank | Normal work (examined / unique) | Work-matched outcome | Work-matched rank | Work-matched work (examined / unique) |
| -------- | ------------------------------- | --------------------------- | -------------------- | ----------: | ------------------------------: | -------------------- | ----------------: | ------------------------------------: |
| case-001 | ambiguous decision              | difficult case              | candidate union miss |      absent |                         24 / 20 | candidate union miss |            absent |                               79 / 73 |
| case-002 | symptom to mechanism            | difficult case              | candidate union miss |      absent |                         24 / 20 | final rank miss      |                56 |                              100 / 86 |
| case-003 | vocabulary drift                | successful baseline control | success              |           3 |                         18 / 13 | success              |                 4 |                               69 / 61 |
| case-004 | lexical only tool evidence      | difficult case              | candidate union miss |      absent |                         19 / 11 | candidate union miss |            absent |                               70 / 53 |
| case-005 | outcome to component            | difficult case              | candidate union miss |      absent |                         19 / 13 | candidate union miss |            absent |                               70 / 64 |
| case-006 | branch competition              | difficult case              | candidate union miss |      absent |                         18 / 15 | candidate union miss |            absent |                               69 / 66 |
| case-007 | current versus obsolete summary | difficult case              | final rank miss      |           6 |                         18 / 14 | success              |                 5 |                               69 / 63 |
| case-008 | exact identifier                | successful baseline control | success              |           2 |                         24 / 15 | success              |                 5 |                              100 / 67 |

## Interpretation guardrails

- No query planner was run or inspected while selecting or measuring these cases.
- The larger arm repeats only the original query and matches the anticipated planned arm’s total pre-fusion candidate allowance.
- A future planned-query result earns candidate-generation credit only when it admits the expected source beyond both controls.
- A source already admitted but below the final-five cutoff is a ranking problem, not a candidate-generation win.
- Existing committed recall-quality evidence remains the production rollout gate; this private baseline does not replace it.
