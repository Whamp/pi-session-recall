---
status: accepted
---

# Split scheduled indexing from zvec optimization

Automatic maintenance uses two schedules:

- `psr index --no-optimize` updates changed session evidence at the configured interval, which defaults to one hour;
- `psr optimize` runs every day at 23:00 local time and optimizes the existing collection without scanning or indexing sessions.

Both operations retain the standalone `psr index` writer path and use the same recall operation lock. Scheduler code does not scan sessions or write recall data itself. If the schedules overlap, one command waits for the other instead of opening a second writer.

Zvec optimization improves search performance by merging staged data into its configured indexes. It is not required after every write. Running it after each hourly change pass repeatedly processes the whole collection, consumes space for a near-collection-sized temporary output, and turns small updates into expensive maintenance. Separating the schedules keeps recent session evidence searchable while bounding optimization to one attempt per day.

Plain manual `psr index` retains its existing changed-pass optimization behavior. `--no-optimize` gives scheduled or manual callers an explicit update-only operation. `psr optimize` opens the collection only after the hourly writer has closed its changed segments, so daily optimization does not mix session writes and compaction in one operation.

`psr auto-index install` owns both native definitions. On Linux it installs hourly index and daily optimize services and timers. On macOS it installs separate interval and calendar LaunchAgents. Uninstall removes both schedules.
