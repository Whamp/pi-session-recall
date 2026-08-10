---
status: accepted
---

# Remove legacy recall optimization

The version 7 Recall database has one dense-only FP32 FLAT Zvec store and one WAL-mode SQLite Recall catalog. Normal indexing updates changed Physical session files only. It does not build HNSW, create Zvec full-text indexes, store lexical-only rows, write zero-vector sentinels, or run corpus-wide optimization.

`psr optimize`, `--no-optimize`, `--optimize-daily`, generated optimization jobs, active-vector cutover reuse, and legacy JSON-state import are removed. `psr auto-index install` always installs only update indexing. As an upgrade action, install and uninstall both disable and remove stale systemd optimization units or the stale macOS optimization LaunchAgent without disturbing the index schedule.

The certified version 7 generation remains activatable through the existing atomic pointer. The unversioned version 6 production database is not deleted. If compact activation verification fails, the operator stops the timer, runs `psr rollback`, redeploys release `8402107` to read the restored version 6 schema, verifies recall, and restarts the timer. The old database remains until Will explicitly approves ending that rollback window.

Flat dense latency and storage were accepted by the production certification in `docs/research/compact-production-recall-certification.md`. If future corpus growth violates those bounds, a new measured decision must choose a replacement; dormant optimization machinery is not retained.
