# Incremental Ingestion Generation Cutover

**Prototype finding, not a production specification.**

## Question

How can an explicit multi-hour rebuild replace the recall generation without deleting the searchable generation first, losing branch-only markers, or forcing incremental workers to write two databases?

## Rejected prototype shape: dual-generation incremental writes

The initial reducer sent every new marker to both the active and building generations. It can preserve correctness, but it requires per-generation acknowledgements, dual projection commits, dual evidence writes, marker garbage collection across both generations, and failure handling when one generation advances and the other does not.

That complexity does not improve interactive recall. A rebuild is rare and explicit, so duplicating all incremental work during it violates KISS/YAGNI.

## Recommended shape: freeze incremental commits, not searches

1. The rebuild coordinator records a durable `building` generation registry entry.
2. Incremental workers stop opening the active generation for writes. Pi continues publishing immutable markers.
3. Searches continue opening the old active generation read-only.
4. The rebuild creates evidence and projection collections under a new generation directory. It never deletes or mutates the active generation.
5. The rebuild imports a bounded source view. Markers published during the rebuild remain in the spool, including branch transitions absent from JSONL.
6. The rebuild optimizes and closes the new generation.
7. Under a brief cutover lock, the coordinator atomically replaces a small active-generation pointer.
8. Searches immediately use the new generation. The old generation remains available for rollback.
9. The ordinary incremental worker replays the retained markers against the new projection. Deterministic IDs and append cursors make records already observed by the rebuild harmless.
10. The worker deletes each marker only after the new generation’s projection checkpoint covers it.

Recall can be stale while the explicit rebuild runs, but it remains available. Search reports a backlog warning only when eligible work exceeds the service objective or fails.

## Generation registry

The durable registry needs only:

- active generation ID;
- optional building generation ID;
- generation schema, import, tokenizer, embedding, and chunk-policy identity;
- rebuild start marker watermark;
- cutover state: `building`, `ready`, `active`, `rollback`, or `failed`;
- active pointer checksum.

The marker spool stays outside every generation. A generation mismatch never discards markers.

## Crash ordering

| Crash point                                 | Recovery                                                                                |
| ------------------------------------------- | --------------------------------------------------------------------------------------- |
| Before registry records `building`          | No rebuild exists; active generation is unchanged                                       |
| During new-generation build                 | Active searches continue; delete or resume only the failed building directory           |
| After optimize but before pointer swap      | Registry can retry the checksum-verified swap                                           |
| During pointer swap                         | Atomic rename leaves either old or new complete pointer                                 |
| After pointer swap but before marker replay | New generation serves results with backlog warning; worker resumes markers              |
| During marker replay                        | Marker remains until the projection checkpoint commits; deterministic upserts replay    |
| After new activation proves unhealthy       | Atomically restore the retained rollback pointer; markers remain generation-independent |

## Consequence

`/pi-session-recall-index --rebuild` should become a side-by-side generation build rather than deleting the active zvec directory before work begins. Incremental workers need one write target at a time, and foreground searches never open a building generation.
