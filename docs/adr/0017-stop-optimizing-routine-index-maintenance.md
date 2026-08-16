---
status: accepted
---

# Stop optimizing routine index maintenance without measured pain

Schema 4 projection reuse and the bulk planning-state query made routine index maintenance fast enough. Recent production runs took about 0.7 seconds when nothing changed, about 2 seconds for a normal changed session, and 9.8 seconds for the largest recent update. The 30-minute schedule has ample headroom.

Keep the current simple design:

- `psr index` remains the only writer.
- Canonical session JSONL remains authoritative.
- Changed files still receive full parsing and graph validation.
- One changed Physical session still replaces its complete projection in one transaction.
- Complete-file hashing remains the correctness check for content-identical files.
- The native 30-minute timer remains the freshness mechanism.

Do not add file watchers, a daemon, a second writer, append-graph checkpoints, row-level projection updates, rolling or block hashes, in-place database migrations, cache-repair machinery, or a schedule shorter than 30 minutes without new production evidence. These changes would add state and recovery paths to save time users do not currently notice.

Reopen this decision only when measurements show a concrete problem:

- routine maintenance repeatedly takes long enough to threaten the freshness target;
- one phase alone exceeds 10 seconds or consumes more than half of changed-file time;
- maintenance causes failures, resource contention, or sustained embedding-server load;
- global search exceeds its 500 ms p95 target and users notice the delay; or
- corpus growth makes the planning scan material again.

Full database rebuilds are the one known performance problem left: the schema 4 rebuild took 11 hours 23 minutes. [ADR 0018](0018-reuse-identical-embedding-inputs-within-physical-sessions.md) removes exact repeated embedding inputs within each Physical session without changing routine maintenance state or recovery. Keep further rebuild work isolated from the live read and update paths.
