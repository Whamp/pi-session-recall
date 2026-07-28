# Incremental Recall Ingestion Prototype

**Throwaway prototype. Do not ship this directory.**

This prototype asks whether a small zvec session projection plus durable work markers can support append-only recall ingestion without rereading whole Pi sessions or blocking interactive Pi work. It also exercises the transition rules for compaction, branch exit, departure, quiescence, deletion, worker crashes, and generation cutover.

Run:

```bash
npm run prototype:incremental-ingestion
```

The command runs scratch-only measurements, writes `prototype-results.json`, then opens a terminal state-machine driver. Pass `-- --report-only` to run the measurements and scripted transition scenarios without opening the terminal driver.

Safety boundaries:

- zvec collections are created only under `.prototype-data/incremental-ingestion/` and removed after measurement;
- the production recall index is never opened;
- session corpus inspection reads file metadata and record timestamps only;
- no conversation text, session path, query, embedding, or tool output is written to the report;
- the prototype never calls embedding or reranking services.

The benchmark answers implementation-shape questions, not retrieval quality. Its thresholds are candidates for the later specification, not production defaults.
