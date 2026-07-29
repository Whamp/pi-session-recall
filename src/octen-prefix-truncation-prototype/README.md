# PROTOTYPE: Octen prefix truncation

This throwaway prototype answers one question:

> Do the original leading coordinates of the current 2,560-dimensional Octen embeddings preserve enough similarity and retrieval behavior to justify storing fewer dimensions?

It compares leading prefixes at 1,536, 1,024, 768, and 512 dimensions with:

- the full 2,560-dimensional vectors;
- a deterministic random set of 1,024 coordinates; and
- the final 1,024 coordinates.

The useful evidence is retrieval behavior, not PCA variance. PCA is deliberately excluded because it rotates the vectors into new coordinates and therefore cannot test Octen's documented first-N truncation behavior.

## Run

From this branch's worktree:

```bash
npm run prototype:octen-prefix-truncation
```

The command reads the embedding cache and regular Arrow scalar files under `~/.pi/agent/recall`. It does **not**:

- open zvec;
- start an inference server;
- call Octen's hosted API;
- alter the embedding cache, zvec files, manifests, or Pi sessions; or
- require an Octen account.

The default run uses 20,000 candidate vectors and 100 user-message queries. A user's active child assistant response serves as a rough expected answer. That is useful for comparing widths, but it is not a replacement for the repository's labeled recall-quality corpus.

To capture a report while rerunning:

```bash
npm run prototype:octen-prefix-truncation -- \
  --markdown-output src/octen-prefix-truncation-prototype/MEASUREMENTS.md
```

## Throwaway status

This code is measurement scaffolding, not production architecture. Keep it on the prototype branch. The production decision to retain is the chosen stored width and the rule to take the leading prefix and L2-normalize it.
