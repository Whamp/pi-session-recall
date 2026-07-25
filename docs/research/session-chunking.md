# Pi session chunking research

> **Implementation decision, 2026-07-25:** Subsequent bounded evaluation and latency measurements selected a 512-token ceiling with 64-token overlap. Deterministic hybrid fusion is the default ranking path; Qwen reranking remains an optional deep-search mode. The 1,024/128 policy below records the earlier research recommendation rather than the final production choice.

## Recommendation

Index **visible semantic text runs** with a hard limit of **1,024 Octen tokens** and **128 tokens of overlap** (12.5%). Treat Pi's structural boundaries as stronger than the size target:

1. Never combine different JSONL entries, roles, or turns.
2. Within a message, combine only adjacent nonempty `text` blocks.
3. End a semantic text run at every `thinking`, `toolCall`, image, or other non-text block.
4. Exclude thinking, tool calls, tool results, and images from embedding.
5. Split a text run only when it exceeds 1,024 tokens. Prefer Markdown sections, paragraphs, fenced code blocks, lines, and sentences before a hard token boundary.
6. Apply overlap only between child chunks from the same text run. Never overlap across message or block boundaries.
7. Store `entryId`, role, text-run index, chunk index, and source path on every child chunk.

Use the exact Octen tokenizer in-process. Cache the tokenizer definition by model revision and checksum rather than estimating tokens from characters or making one HTTP request per text run.

After retrieval, rerank candidate **text**, then optionally expand a winning child chunk with adjacent chunks from the same text run. This keeps embeddings precise without depriving the assistant of surrounding context.

## Why this policy

There is no universal best chunk size. Primary guidance consistently recommends respecting source structure, measuring in the embedding model's tokens, and evaluating several granularities:

- Azure AI Search says fixed chunks with 10–15% overlap can work well, recommends token-aware sizing, and says semantic or source-document boundaries should be preserved. Its generic starting point is 512 tokens with 128 tokens of overlap, not a universal optimum.[^azure]
- Amazon Bedrock's fixed chunker accepts a token cap and overlap percentage while refusing to merge across logical document boundaries. Its semantic chunker also treats the token count as a maximum while honoring sentence boundaries.[^bedrock]
- OpenAI's embedding cookbook measures limits in model tokens and recommends paragraph or sentence boundaries where appropriate.[^openai]
- Pinecone recommends testing 128, 256, 512, and 1,024-token candidates against representative queries. It also recommends chunks that make sense independently and describes neighbor expansion as a way to restore context after precise retrieval.[^pinecone]

A 1,024-token ceiling is a defensible high-context starting point for conversational and code-heavy sessions. It is not justified merely by Octen's much larger context window: a model accepting long input does not mean one broad vector is the best retrieval unit.

## Corpus evidence

Measured on 3,478 Pi JSONL session files on 2026-07-24 with the tokenizer from Octen revision `6e188e3b072c3e3678b235ad84e6e97bcbb71e8f`:

| Measure                                                             |             Result |
| ------------------------------------------------------------------- | -----------------: |
| Entries containing visible user, assistant, custom, or summary text |            111,795 |
| Structurally bounded visible text runs                              |            111,809 |
| Total source tokens before overlap                                  |         43,324,230 |
| Mean tokens per run                                                 |              387.5 |
| Median                                                              |                 27 |
| p90                                                                 |                300 |
| p95                                                                 |                648 |
| p99                                                                 |              5,259 |
| Runs over 1,024 tokens                                              |      3,765 (3.37%) |
| Projected chunks at 1,024/128                                       |            148,563 |
| Projected duplicated overlap tokens                                 | 4,704,512 (10.86%) |

Most conversation entries remain intact. The cap affects a small long tail rather than mechanically slicing ordinary messages.

### Pi block structure matters

The corpus contains 155,230 assistant message entries. Common content arrays interleave visible text, thinking, and tool calls. For example, 33,943 entries have the block-type pattern `text → toolCall → text`, and 33,928 have `thinking → text → toolCall`.

Most text blocks around these boundaries are empty streaming artifacts, but 14 observed entries contain multiple nonempty visible runs separated by thinking or tool boundaries. Joining those runs creates text adjacency that did not exist in the conversation. The parser should preserve the structural rule even though the currently affected count is small.

The present implementation does not split inside a thinking or tool-call block because it excludes those block types. Its defect is that it concatenates all remaining text blocks in an entry after exclusion, erasing the positions of excluded blocks.

## Model constraints

### Octen embedding model

The model card identifies Octen-Embedding-4B as a 2,560-dimensional retrieval model with a 32,768-token advertised context window.[^octen] The live llama.cpp service is deliberately configured with four 8,192-token slots. A 1,024-token chunk is comfortably within both limits.

The local `/tokenize` endpoint matches the model tokenizer, but its `content` array is concatenated into one token sequence rather than processed as a batch. Calling it once per semantic run would require about 111,000 HTTP requests. A pinned local tokenizer is more deterministic and efficient.

`@huggingface/tokenizers` 0.1.3 is a zero-dependency pure-JavaScript tokenizer. Loaded with Octen's pinned `tokenizer.json` and `tokenizer_config.json`, it produced exactly the same token IDs as the live llama.cpp endpoint for 100 deterministic real-session samples, plus targeted prose, code/emoji, and Japanese probes. This is the validated implementation seam; the two model files should be cached with their revision and checksums.

### Qwen reranker

Qwen3-Reranker-4B consumes query/document text pairs; it does not consume or depend on Octen's vectors. Its official model card demonstrates pairing each query with each retrieved document and recommends a task-specific instruction, reporting a typical 1–5% gain from instructions.[^qwen-reranker]

The live endpoint accepted `/v1/rerank` with `query`, `documents`, and `top_n`. It scored an exact source-provenance passage `0.9998946` and an unrelated passage `0.0004051`. The deployed service exposes a 4,096-token slot, so a 1,024-token document leaves substantial room for the query and reranker prompt.

## Proposed retrieval shape

1. Embed the query once with Octen.
2. Retrieve a larger candidate set from zvec by cosine similarity.
3. Deduplicate overlapping siblings before reranking.
4. Rerank the original candidate text with `qwen3-rerank` and a conversation-recall instruction.
5. Return the best results with both vector and reranker scores.
6. Expand adjacent chunks from the same text run only when needed for readable context.

Candidate count and final result count should be measured against a recall evaluation set rather than guessed. Qwen's published evaluation reranks top-100 dense candidates, but that is an accuracy benchmark, not a latency recommendation for this 4B local deployment.[^qwen-reranker]

## Validation before another full backfill

Build a small fixed evaluation set from known past-session facts and compare at least:

- 512 tokens / 64 overlap
- 768 tokens / 96 overlap
- 1,024 tokens / 128 overlap

For each policy, measure retrieval recall before reranking, recall after reranking, duplicate-result rate, reranker latency, and context usefulness. The current partial index should be discarded after the schema records tokenizer revision and chunk-policy version; otherwise old and new chunk geometries can coexist undetected.

[^azure]: [Microsoft Learn, “Chunk Documents”](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)

[^bedrock]: [Amazon Bedrock, “How content chunking works for knowledge bases”](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking.html)

[^openai]: [OpenAI Cookbook, “Embedding texts that are longer than the model's maximum context length”](https://developers.openai.com/cookbook/examples/embedding_long_inputs)

[^pinecone]: [Pinecone, “Chunking Strategies for LLM Applications”](https://www.pinecone.io/learn/chunking-strategies/)

[^octen]: [Octen, “Octen-Embedding-4B” model card](https://huggingface.co/Octen/Octen-Embedding-4B)

[^qwen-reranker]: [Qwen, “Qwen3-Reranker-4B” model card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
