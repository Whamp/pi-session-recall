/** Token ceiling and sibling overlap that define one recall index's chunk geometry. */
export interface RecallChunkPolicy {
  maxTokens: number;
  overlapTokens: number;
}

/** Rejects chunk geometry outside production token and sibling-overlap bounds. */
export function assertRecallChunkPolicy(chunkPolicy: RecallChunkPolicy): void {
  if (
    !Number.isInteger(chunkPolicy.maxTokens) ||
    chunkPolicy.maxTokens < 1 ||
    chunkPolicy.maxTokens > 1_024 ||
    !Number.isInteger(chunkPolicy.overlapTokens) ||
    chunkPolicy.overlapTokens < 0 ||
    chunkPolicy.overlapTokens > 128 ||
    chunkPolicy.overlapTokens >= chunkPolicy.maxTokens
  ) {
    throw new Error(
      'Recall chunk policy invalid: maxTokens must be 1..1024 and overlapTokens must be 0..128 and smaller than maxTokens',
    );
  }
}
