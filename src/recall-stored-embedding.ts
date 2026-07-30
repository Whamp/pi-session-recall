/** Native and stored vector widths plus a source-locational diagnostic label. */
export interface StoredRecallEmbeddingOptions {
  nativeDimensions: number;
  storedDimensions: number;
  source: string;
}

/** Repeatability threshold and source-locational diagnostic label for stored vectors. */
export interface StoredRecallEmbeddingRepeatabilityOptions {
  minimumCosineSimilarity: number;
  source: string;
}

function assertPositiveEmbeddingWidth(width: number, name: string, source: string): void {
  if (!Number.isSafeInteger(width) || width < 1) {
    throw new Error(
      `Recall stored embedding invalid for ${source}: ${name} must be a positive integer`,
    );
  }
}

/** Retains the first N native components and L2-normalizes that exact prefix. */
export function createStoredRecallEmbedding(
  nativeVector: readonly number[],
  options: Readonly<StoredRecallEmbeddingOptions>,
): number[] {
  assertPositiveEmbeddingWidth(options.nativeDimensions, 'native dimensions', options.source);
  assertPositiveEmbeddingWidth(options.storedDimensions, 'stored dimensions', options.source);
  if (options.storedDimensions > options.nativeDimensions) {
    throw new Error(
      `Recall stored embedding invalid for ${options.source}: stored width ${options.storedDimensions} exceeds native width ${options.nativeDimensions}`,
    );
  }
  if (nativeVector.length !== options.nativeDimensions) {
    throw new Error(
      `Recall stored embedding invalid for ${options.source}: expected native width ${options.nativeDimensions}, received ${nativeVector.length}`,
    );
  }

  for (const [index, value] of nativeVector.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(
        `Recall stored embedding invalid for ${options.source}: dimension ${index} is not finite`,
      );
    }
  }
  const retainedPrefix = nativeVector.slice(0, options.storedDimensions);
  const norm = Math.hypot(...retainedPrefix);
  if (!Number.isFinite(norm) || norm <= 0) {
    throw new Error(
      `Recall stored embedding invalid for ${options.source}: retained prefix norm must be positive`,
    );
  }
  return retainedPrefix.map((value) => value / norm);
}

/** Rejects repeated stored-vector canaries whose cosine agreement is below policy. */
export function assertRepeatableStoredRecallEmbeddings(
  first: readonly number[],
  repeated: readonly number[],
  options: Readonly<StoredRecallEmbeddingRepeatabilityOptions>,
): void {
  if (
    !Number.isFinite(options.minimumCosineSimilarity) ||
    options.minimumCosineSimilarity < -1 ||
    options.minimumCosineSimilarity > 1
  ) {
    throw new Error(
      `Recall stored embedding repeatability policy invalid for ${options.source}: cosine threshold must be from -1 through 1`,
    );
  }
  if (first.length === 0 || first.length !== repeated.length) {
    throw new Error(
      `Recall stored embedding repeatability mismatch for ${options.source}: vector widths differ`,
    );
  }

  let dotProduct = 0;
  let firstSquaredNorm = 0;
  let repeatedSquaredNorm = 0;
  for (let index = 0; index < first.length; index += 1) {
    const firstValue = first[index];
    const repeatedValue = repeated[index];
    if (
      firstValue === undefined ||
      repeatedValue === undefined ||
      !Number.isFinite(firstValue) ||
      !Number.isFinite(repeatedValue)
    ) {
      throw new Error(
        `Recall stored embedding repeatability mismatch for ${options.source}: vectors must be finite`,
      );
    }
    dotProduct += firstValue * repeatedValue;
    firstSquaredNorm += firstValue * firstValue;
    repeatedSquaredNorm += repeatedValue * repeatedValue;
  }
  const normProduct = Math.sqrt(firstSquaredNorm * repeatedSquaredNorm);
  const cosineSimilarity = normProduct > 0 ? dotProduct / normProduct : Number.NaN;
  if (!Number.isFinite(cosineSimilarity) || cosineSimilarity < options.minimumCosineSimilarity) {
    throw new Error(
      `Recall stored embedding repeatability mismatch for ${options.source}: expected cosine similarity at least ${options.minimumCosineSimilarity}, received ${cosineSimilarity}`,
    );
  }
}
