# Zvec FTS segment repair

## Outcome

A production Zvec collection contained one immutable FTS segment whose postings violated Zvec's BitPacked input requirement. The defect made every later collection optimization fail with:

```text
FtsRocksdbReducer: source postings is not BitPacked. field=content
```

Rebuilding both collection-level FTS indexes repaired the defect. The repaired 1,446,592-document collection then completed full optimization twice. Production was switched to the validated collection on 2026-08-09.

The repair succeeded. The cause of the malformed segment remains unknown. [Issue #164](https://github.com/Whamp/pi-session-recall/issues/164) tracks that investigation and recurrence prevention.

## What was malformed

The failure was isolated to immutable shard 26. Its two FTS fields contained only Roaring postings:

| Field               |  Terms | BitPacked | Malformed |
| ------------------- | -----: | --------: | --------: |
| `content`           | 20,913 |         0 |    20,913 |
| `identifierContent` | 22,954 |         0 |    22,954 |

The active mutable shard also used Roaring postings, as expected. All other sampled immutable shards used BitPacked postings.

Shard 26 contained 4,088 records. Of those records, 4,081 remained live with unchanged checksums, seven had been deleted, and none had been superseded. Removing the shard would have lost searchable evidence.

## What the manifest proves

The shard's RocksDB manifest records the normal FTS side column families for term frequency, maximum term frequency, and document length. Those column families contained SST data and were later dropped.

Zvec drops these side tables after `convert_postings_to_bitpacked()`. Their removal proves the seal path advanced beyond conversion, but the final posting values remained Roaring. Sampled internal-key histories contained merge records and no later BitPacked `Put` values.

The process received `SIGTERM` about four minutes after the manifest recorded the side-table drops. The interruption did not occur between posting flush and conversion. A simple interrupted seal therefore does not explain the persisted state.

## Bounded reproductions

Two small collections used the production FTS schema and Zvec 0.6.0:

1. A writer inserted 1,000 documents and was killed. The collection reopened, optimized, accepted more writes, and optimized again.
2. A production-shaped fixture inserted 4,088 documents in batches of 128. Each FTS field contained roughly 408,800 tokens from a 20,000-term vocabulary. It optimized, accepted more writes, and optimized again.

Neither experiment produced malformed postings. Both stayed below 50 MB. These results rule out simple writer interruption and ordinary repeated optimization as sufficient causes. They do not rule out a large-collection race or an unobserved maintenance path.

## Repair trial

The repair ran on a Btrfs copy-on-write clone. Guards stopped the operation if the clone exceeded 12 GiB of exclusive growth, free space fell below 240 GiB, or runtime exceeded four hours.

The supported Zvec repair sequence was:

1. Drop the `content` FTS index with `dropIndexSync()`.
2. Recreate it with `createIndexSync()` from stored scalar text.
3. Repeat the drop and create for `identifierContent`.
4. Scan every immutable FTS database for non-BitPacked postings.
5. Run `psr optimize` against the repaired clone.
6. Run `psr optimize` again and compare searches.

Each field rebuild took about 299 seconds. Peak exclusive clone growth was about 5.75 GiB. The rebuilt clone contained zero malformed postings across 82 immutable FTS databases:

| Field               |     Terms | Malformed |
| ------------------- | --------: | --------: |
| `content`           | 1,920,922 |         0 |
| `identifierContent` | 2,043,194 |         0 |

The first full optimization completed in 3 minutes 44 seconds. It reached 100% vector-index completeness, preserved all 1,446,592 documents, and left no temporary segment. Peak exclusive clone usage was about 16.13 GiB. A second optimization also completed and preserved every sampled result ID, rank, and score.

These times and sizes describe one host and collection. They are not resource guarantees.

## Ranking effect

The FTS repair preserved all tested identifier-channel results exactly. Lexical top-20 overlap before and after repair ranged from 4/20 to 17/20 across five queries. Maximum sampled score changes reached 14.16.

The malformed postings lacked the side-table payload needed for full BM25 term-frequency and document-length scoring. Rebuilding restored that information. Full optimization then merged FTS segments and changed ranking again. A second optimization produced identical sampled rankings and scores.

Optimization therefore preserves indexed evidence but can change lexical scores and order. Treat it as both a performance and ranking operation.

The comparison used five queries and top-20 result lists. It proves a semantic effect, not a general ranking-quality improvement.

## Production cutover

Before activation, the repaired clone and production had matching document counts and byte-identical index state, index manifest, maintenance status, and ignore state. A reflink candidate consumed zero exclusive bytes before activation.

The production timer was stopped, readers were allowed to close, and the live collection directory was renamed to a rollback path. The candidate was then renamed into place. Direct collection checks and the `pi-session-recall` tool both succeeded against the new path before the rollback was removed.

The final production state was:

- 1,446,592 documents;
- 100% vector-index completeness;
- zero malformed postings in the compacted FTS segment;
- no optimization temp directory;
- one 14 GiB live collection.

A compressed 9.1 MB copy of the malformed shard remains in local forensic storage. It is not committed because its scalar data contains private session content.
