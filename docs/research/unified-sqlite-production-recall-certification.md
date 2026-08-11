# Unified SQLite production recall certification

This report records pre-activation checks only. The harness did not activate the candidate.

## Verdict

The candidate passed every local pre-activation gate.

## Gates

| Gate              | Result |
| ----------------- | ------ |
| storage           | PASS   |
| projectLatency    | PASS   |
| globalLatency     | PASS   |
| invocationLatency | PASS   |
| invocationProbes  | PASS   |
| denseProbes       | PASS   |
| sourceProvenance  | PASS   |
| integrity         | PASS   |
| linuxX64Load      | PASS   |
| candidateInactive | PASS   |
| clone             | PASS   |

Platform runtime loading is verified separately by the PR's SQLite-vec GitHub Actions jobs.

## Activation status

Activation, immediate post-activation indexing, live project/global/Source recall, timer verification, and obsolete-artifact cleanup remain pending until this branch is merged and deployed.
