# Superseded v7 compact production recall certification

> **Superseded historical evidence.** This report certified the incompatible v7 flat-Zvec-plus-SQLite candidate. ADR-0014 supersedes that architecture. These PASS results do not certify the v8 unified SQLite Recall database, and the recorded generation must not be activated.

Measured on `desktop` on 2026-08-10 for issue #172.

## Historical verdict

The staged compact database passes every pre-activation production gate. It remains inactive because the installed production package is still at `8402107` on `master`, which cannot read the version 7 compact layout. Activating before this branch is merged and deployed would break the hourly service and live Pi recall.

Candidate target:

```text
generations/generation-f71da5f6-7ea1-43a8-8412-d0c746c6068d
```

The active pointer remained absent before and after construction and certification, so production still resolves the complete version 6 database at `~/.pi/agent/recall/`. The staged generation and the version 6 database both remain available. No production database was deleted.

The complete machine-readable measurements are in [superseded-v7-compact-production-recall-certification.json](superseded-v7-compact-production-recall-certification.json).

## Results

| Gate                            |               Limit |      Measured | Result |
| ------------------------------- | ------------------: | ------------: | ------ |
| Dense store plus catalog        |           ≤ 2.5 GiB |     2.279 GiB | PASS   |
| Flat dense search p95           |            < 100 ms |      49.03 ms | PASS   |
| Dense top result                | 5/5 control matches |           5/5 | PASS   |
| Dense top-eight overlap         |     ≥ 7/8 per query | 7, 7, 8, 8, 8 | PASS   |
| Invocation search p95           |              < 5 ms |       3.26 ms | PASS   |
| Source search probes            |     exact locations |           4/4 | PASS   |
| Changed-session device writes   |            < 10 MiB |      5.73 MiB | PASS   |
| Unrelated catalog state changes |                   0 |             0 | PASS   |

The staged database contains 341,036 Dense recall documents and 218,139 Invocation records. Allocated storage is 2,178,654,208 bytes for Zvec and 267,890,688 bytes for SQLite, including its WAL and shared-memory files.

## Dense comparison

Each query ran once as warmup and five times for measurement against the version 6 HNSW control and staged FLAT store.

| Query                                                                    | Same top result | Top-eight overlap |
| ------------------------------------------------------------------------ | --------------: | ----------------: |
| Why have recent pi-session-recall optimization attempts failed?          |             yes |               7/8 |
| How is automatic recall indexing scheduled?                              |             yes |               7/8 |
| Which corrupted February session files are ignored?                      |             yes |               8/8 |
| How large is the recall database?                                        |             yes |               8/8 |
| Why would an agent use pi-session-recall instead of searching raw JSONL? |             yes |               8/8 |

The 25 measured staged-search samples had a nearest-rank p95 of 49.03 ms.

## Invocation probes

Each probe ran once as warmup and five times for measurement. Every probe returned at least one result; the tool-name probe also returned an Invocation whose exact tool name was `brain_query`.

| Kind         | Query                             | Results |
| ------------ | --------------------------------- | ------: |
| Tool name    | `brain_query`                     |      20 |
| Path         | `/home/will/.pi/agent/TAILNET.md` |      20 |
| URL          | `http://192.168.0.67:8090/v1`     |      20 |
| Command      | `psr optimize`                    |       4 |
| Issue number | `gh issue view 165`               |       2 |
| Flag         | `--optimize-daily`                |       2 |

The 30 measured Invocation samples had a nearest-rank p95 of 3.26 ms.

## Source probes

Each explicit Source search scanned 3,719 eligible Physical session files with zero read failures. The two configured ignored paths remained excluded.

| Kind                | Query                                                         | Exact Source location                                                                                                                                     | Source role     |
| ------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| Result-only error   | `FtsRocksdbReducer`                                           | `/home/will/.pi/agent/sessions/--home-will-projects-pi-session-recall--/2026-08-08T20-43-57-969Z_019fe31e-3e91-7df4-969a-8c8b1f1ec757.jsonl:176#2165f7ed` | `toolResult`    |
| Hardware identifier | `CT1000P3PSSD8`                                               | `/home/will/.pi/agent/sessions/--home-will--/2026-01-08T21-15-34-355Z_1d4e2cfc-3f56-4c74-958a-78a36a55efd1.jsonl:7#f3016783`                              | `toolResult`    |
| Filename            | `2026-02-02T18-31-25`                                         | `/home/will/.pi/agent/sessions/--home-will--/2026-02-10T15-22-22-670Z_b3772b85-c4ca-41bf-8e42-15e1debfd844.jsonl:13#a8b282ed`                             | `toolResult`    |
| Command output      | `pi - AI coding assistant with read, bash, edit, write tools` | `/home/will/.pi/agent/sessions/--home-will--/2026-01-12T23-52-11-516Z_442dcec9-9f02-4ee2-a45e-862b6b5ec863.jsonl:4#b43c054d`                              | `bashExecution` |

Source scans took 15.1–15.5 seconds each and wrote no persistent Source-search state.

## Changed-session write probe

The probe used the 10,025,616-byte production session named in the result-only error case. It changed only that file's modification time, ran one candidate update, restored the exact original nanosecond timestamp, and ran one cleanup update. Session content never changed.

The measured update:

- indexed one Physical session;
- reused 829 dense vectors and embedded none;
- changed no unrelated catalog state;
- wrote 622,592 process-attributed bytes;
- wrote 13,508,608 gross bytes to `nvme0n1` during the 5.59-second window;
- observed 7,503,872 median background bytes across three equal-duration idle windows;
- attributed 6,004,736 bytes, or 5.73 MiB, to the update.

The catalog now compares document identities and Invocation persistence fields before replacement. When those child rows are unchanged, it updates only the session metadata. This reduced process-attributed writes from 3.2 MiB in the first measurement to 0.59 MiB in the passing run.

## Construction and recovery

`psr index --rebuild --stage` used a separate construction lock. The version 6 active database remained readable, and the hourly timer stayed active. Journal evidence records successful no-op production index runs at 01:33, 02:34, and 03:35 PDT while candidate construction was in progress.

The 04:36 scheduled run overlapped the certification process's read-only HNSW control and failed to acquire Zvec's collection lock. It changed no production data. After certification closed the control reader, an immediate 05:35 run completed successfully with zero failed sessions. The hourly timer is active.

The initial build stopped safely after 1,197 completed sessions when `octen-embed` became unavailable. The candidate remained inactive and retained 105,424 completed embeddings. A resumed attempt also stopped safely during another endpoint interruption. The final command resumed the same candidate and reused active vectors only when the embedding profile, canonical document ID, and checksum matched:

```bash
node --import tsx bin/psr index --rebuild --stage --resume --reuse-active-vectors
```

That pass completed 2,513 remaining sessions in 35 minutes 7 seconds, reused 230,630 vectors, embedded no documents, and reported zero failed sessions. No active-only row was seeded into the candidate; reuse occurred at the canonical document seam during indexing.

## Historical reproduction record

The v7 certification executable and its tests were removed when ADR-0014 superseded the layout. The following command identifies how this historical report was produced; it is not runnable on the current branch:

```bash
node --import tsx src/certify-compact-recall-production.ts \
  --candidate generations/generation-f71da5f6-7ea1-43a8-8412-d0c746c6068d \
  --block-device nvme0n1 \
  --changed-session /home/will/.pi/agent/sessions/--home-will-projects-pi-session-recall--/2026-08-08T20-43-57-969Z_019fe31e-3e91-7df4-969a-8c8b1f1ec757.jsonl \
  --output docs/research/compact-production-recall-certification.json
```

The removed command exited 0 only when every v7 pre-activation gate passed and left the active pointer unchanged.
