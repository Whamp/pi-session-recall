# Fuse hybrid retrieval in application code

Zvec 0.6.0 can fuse dense and FTS queries through `multiQuerySync()`, but its result exposes only the final score. Recall instead runs bounded dense, ordinary FTS, and case-preserving identifier queries separately, then applies deterministic reciprocal rank fusion with policy version 1 and rank constant 60. This preserves every component rank and score plus a document-ID tie-break for evaluation; native fusion remains unsuitable unless it can expose the same evidence.
