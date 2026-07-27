# Query-Planned Recall: Deterministic Source-Admission Quality

This report compares fixed private agent plans with normal hybrid and retrieval-work-matched original-query controls. It contains only opaque case identities, query hashes, policy values, counts, ranks, and checksums; private query and source text remain local.

## Identity

- Recorded against commit: `030396576c03c705a1f3c84dce1ff639256ed2cf`
- Private manifest SHA-256: `4b3fb8573bcd52a26d66a2f8e60f5bbba57b098530791b23f55505d3e9a1187e`
- Fixed plan source: `agent`
- Fixed plan SHA-256: `245f28cb6f541d336f99a829c187e76e8f50e3a81bb473626f5c21e5b5b640a5`
- Publishable controls canonical JSON SHA-256: `a5b300c736ddfed35e4e65635fe5752b17b53df2c61a97079229c3ff0b44d605`
- Embedding policy: `deterministic-token-hash-v1`, 256 dimensions
- Ranking reranker policy: `neutral-fused-order-v1`
- Admission probe reranker policy: `expected-source-promotion-v1`
- Frozen cases: 8 across 8 snapshots and 44521 indexed documents
- Executed searches: 32 (normal hybrid, retrieval-work-matched original query, neutral planned ranking, and planned admission probe per case)

## Quality gate

- New candidate admission beyond both controls: 4
- Ranking-only promotion of an already admitted source: 1
- Preserved existing success: 2
- No improvement: 2

A planned search receives source-admission credit only when its admission probe finds the expected source and both original-query candidate unions miss. Promotion of a source admitted by either control is reported separately as ranking-only behavior. Existing-success preservation is an independent guard and can overlap a contribution class.

## Ranking policy

- Per-list work: dense 20, lexical 20, identifier 20, planned_lex 20, planned_vec 20
- Fused pool / rerank pool / final results: 40 / 40 / 5
- Fusion: RRF v2, k=60, submitted weight 2, planned weight 1, bonuses 0.05 / 0.02
- QMD reranker policy: v2, active-branch prior 0.01, blend bands 1-3 0.75/0.25, 4-10 0.6/0.4, 11-end 0.4/0.6

## Cases

| Case     | Category                        | Normal hybrid        | Retrieval-work-matched original query | Query-planned   | Contribution               | Planned work (admitted / allowed) |
| -------- | ------------------------------- | -------------------- | ------------------------------------- | --------------- | -------------------------- | --------------------------------: |
| case-001 | ambiguous decision              | candidate union miss | candidate union miss                  | final rank miss | new candidate admission    |                          92 / 100 |
| case-002 | symptom to mechanism            | candidate union miss | final rank miss                       | final rank miss | no improvement             |                         100 / 100 |
| case-003 | vocabulary drift                | final rank miss      | final rank miss                       | success         | ranking-only promotion     |                          82 / 100 |
| case-004 | lexical only tool evidence      | candidate union miss | candidate union miss                  | success         | new candidate admission    |                          83 / 100 |
| case-005 | outcome to component            | candidate union miss | candidate union miss                  | final rank miss | new candidate admission    |                          83 / 100 |
| case-006 | branch competition              | candidate union miss | candidate union miss                  | final rank miss | new candidate admission    |                          82 / 100 |
| case-007 | current versus obsolete summary | success              | success                               | final rank miss | no improvement             |                          82 / 100 |
| case-008 | exact identifier                | success              | success                               | success         | preserved existing success |                         100 / 100 |

## Fixed plan query identities

- case-001: lex `660ee51fc5f6c679fcd601a914bc33f448a91858ed6296b282953439459326aa`, vec `86353199e653f426c0133ec3b15710038ea3b2e5c142b76b469fb03a5a6d9ed9`
- case-002: lex `108b78976e09adee24d88175790c2727e6082106c0571597f8e8a3254c1c4dee`, vec `a42d31e8f6a1d9bd25db354d8d56386bd72800eee244e06d0d00830b3c314a88`
- case-003: lex `e4eae58d6a8e67bc6278aae441c416a2cf93305e8b8c2c2fdb38cc6f23e66427`, vec `9d5fe4791dc1aabdce012a04c7b88bb7ff29327be4c9799e5626b1d037ff08c2`
- case-004: lex `228f64b95b93e7929725380a264a1b1fffd4bc3ec437f256411a18b8a6a4f327`, vec `1ff26c147666b9e472907c8ecc63325fe207ed6ee149b29627ef2432101fd3cb`
- case-005: lex `bb46098d12140cc738d30bebb58e1704361b216e3ee1783a45f25908aa07122b`, vec `15fdd43f0be0ba4a800657398119203fe5013cd17eaf3a69e2040369d919bfff`
- case-006: lex `09690a188e1509806fece9ed71d2ef479f101f1aa5ed24e61d4638d3feabf000`, vec `e2ac063f81c7e0f70ba2e6f97857a9bf13077e0905c3c8f176d8de23610a9924`
- case-007: lex `ea77608e2bcda9e1949849647108d15c9f653704d4da9fc7d85fbd9ffd7f8bcb`, vec `24f60d38bf1846dc0c193040c97bb16305d0fe31e5712e8ef84ec692d7112b8a`
- case-008: lex `cd97a32ca0bad9a28f5872e6d287709b50a3a4391b2b3948da691dc03dcece71`, vec `d6ca855d76a2bda623fe141447459ace8204c367caee0ddd4de558ff10e2c56a`
