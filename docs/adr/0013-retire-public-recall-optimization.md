---
status: accepted
---

# Retire public recall optimization

> ADR-0014 supersedes this ADR's version 7 storage wording. The decision to retire public and scheduled optimization remains accepted.

The compact version 7 layout makes ordinary maintenance update only new, changed, missing, or newly ignored Physical session files. Corpus-wide optimization is not part of that maintenance contract.

`psr optimize` and `psr auto-index install --optimize-daily` are removed. Automatic installation always creates one update-only index job. Installation and uninstallation continue to disable and remove stale systemd optimization units or the stale macOS optimization LaunchAgent left by older releases.

Legacy storage support remains temporarily available so the unversioned version 6 production database can continue serving search until the certified compact generation is activated. Removing those internal paths and deleting the previous database remain gated on successful activation and explicit rollback-window approval.
