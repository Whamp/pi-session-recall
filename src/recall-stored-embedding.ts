/** Converts one native embedding into the configured first-N, L2-normalized FP32 stored vector. */
export function createStoredRecallEmbedding(
  nativeEmbedding: readonly number[],
  nativeDimensions: number,
  storedDimensions: number,
): number[] {
  if (!Number.isInteger(nativeDimensions) || nativeDimensions < 1) {
    throw new Error(
      `Recall native embedding dimensions invalid: expected a positive integer, received ${nativeDimensions}`,
    );
  }
  if (nativeEmbedding.length !== nativeDimensions) {
    throw new Error(
      `Recall native embedding dimension mismatch: expected ${nativeDimensions}, received ${nativeEmbedding.length}`,
    );
  }
  if (
    !Number.isInteger(storedDimensions) ||
    storedDimensions < 1 ||
    storedDimensions > nativeDimensions
  ) {
    throw new Error(
      `Recall stored embedding dimensions invalid: expected an integer from 1 to ${nativeDimensions}, received ${storedDimensions}`,
    );
  }

  const prefix: number[] = [];
  let squaredNorm = 0;
  for (let index = 0; index < storedDimensions; index += 1) {
    const value = nativeEmbedding[index];
    if (value === undefined || !Number.isFinite(value)) {
      throw new Error(`Recall native embedding invalid: value ${index} is not finite`);
    }
    const float32Value = Math.fround(value);
    prefix.push(float32Value);
    squaredNorm += float32Value * float32Value;
  }
  if (squaredNorm === 0) {
    throw new Error('Recall stored embedding invalid: vector norm must be positive');
  }
  const norm = Math.sqrt(squaredNorm);
  return prefix.map((value) => Math.fround(value / norm));
}

/** Converts normalized inner-product similarity into the existing bounded cosine-distance scale. */
export function convertNormalizedRecallInnerProductToCosineDistance(
  innerProductSimilarity: number,
): number {
  if (!Number.isFinite(innerProductSimilarity)) {
    throw new Error('Recall dense similarity invalid: expected a finite inner-product score');
  }
  return 1 - Math.max(-1, Math.min(1, innerProductSimilarity));
}
