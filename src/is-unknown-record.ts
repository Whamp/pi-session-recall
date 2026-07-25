/** Narrows opaque JSON-like input to a non-null, non-array object record. */
export function isUnknownRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
