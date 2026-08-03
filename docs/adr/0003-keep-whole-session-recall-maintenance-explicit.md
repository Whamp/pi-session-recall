---
status: accepted
---

# Keep whole-session recall maintenance outside interactive Pi

Interactive Pi operations never maintain the recall index. Startup, settled turns, shutdown, reload, and `pi-session-recall` search read the existing zvec collection without scanning or rewriting session evidence.

The standalone `psr index` command remains the sole writer and change detector. Operators can run it directly or opt into a native per-user schedule with `psr auto-index install`. Scheduling code only installs or removes native definitions that invoke `psr index`; it does not scan sessions or write recall state. `psr index --rebuild` remains the explicit command for replacing incompatible state.

Whole-session maintenance reparses and retokenizes changed physical session files. Running that work in Pi lifecycle or search paths duplicated active-context work and blocked the interactive process. Keeping maintenance in the standalone command preserves one visible writer and leaves append-aware ingestion as a separate design problem.
