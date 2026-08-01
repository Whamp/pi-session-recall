---
status: accepted
---

# Keep whole-session recall maintenance explicit

Interactive Pi operations never maintain the recall index. Startup, settled turns, shutdown, reload, and `pi-session-recall` search read the existing zvec collection without scanning or rewriting session evidence. Operators run `psr index` to add, change, or remove evidence and `psr index --rebuild` to replace incompatible state.

Whole-session maintenance reparses and retokenizes changed physical session files. Running that work in Pi lifecycle or search paths duplicated active-context work and blocked the interactive process. The explicit CLI keeps one visible writer and leaves future append-aware ingestion as a separate design problem.
