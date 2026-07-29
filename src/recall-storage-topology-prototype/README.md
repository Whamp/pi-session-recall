# PROTOTYPE — predictable recall storage topology

## Question

Can the smallest useful embedded topology keep lexical/source evidence vector-free, retain dense rows only when an embedding exists, traverse an exact source neighborhood without reopening session JSONL, and remain understandable across build, append, replay, reopen, validation, deletion, and crash recovery?

This is throwaway evidence for [Prototype predictable storage topology on representative recall evidence](https://github.com/Whamp/pi-session-recall/issues/110), not production architecture. It writes only beneath `/tmp/pi-session-recall-storage-topology-prototype` and deletes the scratch generation at the end. It never opens the live recall collection.

The prototype uses a repeatable automated probe because filesystem sizing and crash faults are not meaningful as manually selected state-machine transitions. Its default terminal UI still exposes the current artifact and lets a reviewer run, inspect, or wipe it.

## Run

```bash
npm run prototype:recall-storage-topology
```

For a noninteractive full run:

```bash
npm run prototype:recall-storage-topology -- --all
```

The run updates `MEASUREMENTS.md` beside this file and prints the complete measured state.

## Candidate topology

```text
generations/<id>/
├── lexical-source/   # vector-free FTS evidence plus immutable entry anchors
├── dense/            # only dense-searchable evidence, keyed by occurrence ID
├── projections/      # small mutable physical/logical ingestion projections
├── index-manifest.json
└── operation-state.json
```

An entry anchor is an immutable source-neighborhood index row. It stores parent linkage, branch endpoint membership, exact occurrence IDs, and source geometry. It is not searchable evidence. This avoids a fourth store while preserving direct ID lookup and one-path traversal.

## Durability scope

The session JSONL files protect the data. Durability work protects the time already spent creating embeddings, which can take many hours.

After a crash, replay the source and reuse rows whose occurrence ID, embedding profile, and content checksum still match. Re-embed only missing or damaged rows. Rebuild the whole generation only when the damage cannot be isolated.

Prefer small checkpoints and simple validation. Add stronger database guarantees only when they save meaningful rebuild time without adding much code or storage.
