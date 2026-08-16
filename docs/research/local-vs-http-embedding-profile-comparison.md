# Local Octen 0.6B versus HTTP Octen 4B

## Verdict

Keep the HTTP Octen 4B profile for Will's production Recall index. It matched or slightly exceeded the local model's retrieval quality, answered queries 19% faster, built the fixed sample index 2.5 times faster, and projected to finish the embedding phase of a full rebuild about three times faster.

The shipped local Octen 0.6B profile remains a strong offline default. Its dense retrieval quality was effectively tied with the HTTP model, and it kept every expected source within the eight dense candidates. Its tradeoff is speed on this desktop CPU, not a large quality loss.

## Quality and search speed

The harder comparison used 32 predeclared query variants over eight source targets. Each profile ran the complete suite five times. All rankings were stable across runs.

| Measure                          | HTTP Octen 4B | Local Octen 0.6B |
| -------------------------------- | ------------: | ---------------: |
| Dense candidate recall at 8      |          100% |             100% |
| Dense mean reciprocal rank       |    **0.8464** |           0.8438 |
| Final recall at 5                |      **100%** |          96.875% |
| Hybrid mean reciprocal rank      |    **0.8255** |           0.7961 |
| Context and source preservation  |      **100%** |          96.875% |
| Query latency, pooled median     |  **26.70 ms** |         32.78 ms |
| Query latency, pooled p95        |  **28.23 ms** |         36.27 ms |
| Fixed 125-document index, median |    **4.42 s** |          11.25 s |

Dense ranking was a practical tie: each model ranked three queries better than the other, and 26 tied. The HTTP model's dense MRR lead was 0.0026.

The end-to-end Recall result favored HTTP. One local result ranked first in the dense channel but fell to seventh after normal hybrid fusion, outside the final five. HTTP returned that source first. This difference reflects the complete embedding profile, including its tokenizer and resulting chunk text, rather than the embedding model alone.

Both profiles also ran the existing 16-case release gate five times. Both preserved 100% candidate recall, final recall, context, source occurrences, and provenance. The local profile scored 0.875 dense MRR there, versus 0.84375 for HTTP; that smaller suite was not strong enough to drive the decision.

## Indexed Recall versus raw JSONL agents

A separate baseline gave the same 32 queries to coding agents that could read only the 15 raw JSONL files in the synthetic evaluation corpus. They could not use Recall, its database, repository documentation, evaluation answers, or prior session memory.

| Path                                      | Strict result | Measured time                   | Raw files examined |
| ----------------------------------------- | ------------: | ------------------------------- | -----------------: |
| HTTP Octen 4B indexed Recall              |         32/32 | 26.70 ms median per query       |                  0 |
| Local Octen 0.6B indexed Recall           |         31/32 | 32.78 ms median per query       |                  0 |
| Fresh raw agent for every query           |         32/32 | 15.715 s median; 24.764 s p95   |       15 per query |
| One raw agent answering all 32, best case |         24/32 | 27.106 s for the complete batch |           15 total |

The cold baseline used 32 fresh GPT-5.6 Sol workers, one per query. Every worker independently discovered and opened all 15 files. Their measured search intervals totaled 520.696 seconds if run sequentially. All 32 answers and citations passed an independent grader using the same required-context and source-occurrence rules as Recall.

The one-worker batch is an amortized lower bound, not normal fresh-query behavior. That worker learned the corpus once and reused it for every question. Its output contract also allowed only one citation, making the four two-source duplicate checks impossible; four more answers omitted adjacent question context. The 24/32 score therefore measures that specific process and contract, not the model alone.

These timings cover different boundaries. Recall latency measures retrieval and evidence loading, while the raw-agent interval includes filesystem investigation and answer composition but excludes model startup before its first tool call and response serialization after its last tool call. The 15-file synthetic corpus also favors raw search compared with a real history containing hundreds or thousands of files. The baseline shows that raw inspection can answer well without setup, but it repeatedly pays schema discovery and file-reading costs that an index pays once.

## Full-rebuild embedding speed

The throughput comparison read the active schema-4 database without writing it. It reduced 406,549 stored documents to the **306,736 distinct texts that the current per-Physical-session memoization would embed**, totaling 76,215,908 stored tokens.

The sample selected 128 texts from each of five token-length bands, for 640 texts and 143,321 tokens. Selection used the lowest SHA-256 values of the Physical-session path and content checksum. No sampled text or path is stored in the report.

Each profile embedded every band three times with production settings:

- HTTP: sequential batches of 16 sent over the LAN to Octen 4B on `endurance`;
- local: four concurrent batch-one operations through native ONNX Runtime on `desktop`.

Per-band median time was weighted by the number of corpus inputs in that band.

| Projection                 | HTTP Octen 4B | Local Octen 0.6B |
| -------------------------- | ------------: | ---------------: |
| Embedding phase            |    **6.90 h** |          20.72 h |
| Observed-range projection  |   6.88–7.55 h |    20.62–21.20 h |
| Estimated complete rebuild |    **7.98 h** |          21.80 h |
| Relative embedding time    |        **1×** |            3.00× |

The complete-rebuild estimate adds the 3,889-second non-embedding remainder measured in the retained schema-4 full rebuild. It is a projection, not a replacement for a staged rebuild measurement.

The API projection agrees with earlier evidence: the historical pre-memoization HTTP rebuild spent 10.31 hours embedding 403,687 documents. Removing exact repeated inputs was expected to reduce that work materially.

## Environment

Local measurements ran on `desktop`:

- AMD Ryzen 7 8845HS, 8 cores and 16 threads;
- 60 GiB RAM;
- Node 24.16.0;
- `onnxruntime-node` 1.27.0 native backend;
- pinned `local-octen-embedding-0.6b-onnx-int8-v1` artifact;
- local cold model load and first query: 762 ms;
- measured process peak RSS: 1.15 GiB.

HTTP measurements used the healthy `octen-embed` endpoint at `endurance`:

- live model metadata reported 4,021,774,336 parameters and 2,560 native dimensions;
- Recall stored the normalized first 1,024 dimensions;
- the documented host is an RTX 3080 Ti 12 GB under a 200 W cap running llama.cpp CUDA;
- exact server GPU utilization and co-resident Plex load were not captured during the benchmark;
- two HTTP samples showed transient contention, so the projection uses per-band medians and retains the full range.

## Limits

- The harder suite has 32 synthetic queries over eight source targets. It can expose ranking differences, but it cannot prove broad equivalence on Will's private real-world questions.
- Raw-agent timing measures a different execution boundary from Recall retrieval. The cold workers ran concurrently for throughput, so only their individual intervals and sequential sum are reported as query-cost evidence.
- The raw-agent corpus has only 15 files. Its success and latency do not predict performance over a large private session history.
- HTTP speed includes LAN transport and the shared `endurance` service. Local speed measures this desktop CPU. This comparison answers the deployment decision, not hardware-normalized model efficiency.
- The rebuild projection holds API-tokenized text inputs fixed. A true local-profile rebuild can produce slightly different token counts and chunk boundaries.
- Client RSS does not include remote server memory.
- Only a real staged rebuild can validate the total rebuild projections.

Machine-readable evidence: [`local-vs-http-embedding-profile-comparison.json`](../evaluation/local-vs-http-embedding-profile-comparison.json).

Fixed query variants: [`embedding-profile-query-variants.json`](../../evaluation/embedding-profile-query-variants.json).
