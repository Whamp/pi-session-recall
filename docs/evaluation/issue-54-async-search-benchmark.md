# Issue 54 async sequential search benchmark

Date: 2026-07-27

The issue 48 prototype report contains write-window, projection, metadata, and append-preparation measurements. It does not contain a three-route search latency or event-loop-delay baseline.

This ticket therefore compared synchronous and awaited sequential queries on the same isolated scratch collection. The benchmark did not read production recall data or call an inference service.

## Workload

- zvec 0.6.0
- 4,096 generated documents
- 2,560-dimensional vectors
- ordinary and case-preserving FTS fields
- dense, ordinary lexical, then identifier route order
- 40 candidates per route
- 20 measured searches after warm-up

## Results

| Query path             | Latency p50 | Latency p95 | Maximum timer delay p95 |
| ---------------------- | ----------: | ----------: | ----------------------: |
| Synchronous sequential |   23.918 ms |   27.042 ms |               26.082 ms |
| Awaited sequential     |   24.956 ms |   27.332 ms |                0.702 ms |

Awaited sequencing reduced p95 timer delay by 97.3%. Its p95 query latency was 0.290 ms higher (1.1%), which is within the run's sub-millisecond host variation. The result supports awaited sequential queries: foreground latency stayed effectively flat while the event loop remained responsive between zvec routes.

The final rollout ticket still owns a temporary production-copy benchmark because issue 54 is prohibited from reading production recall data.
