import { createHash } from 'node:crypto';

import { isUnknownRecord } from './is-unknown-record.js';

function serializeCanonicalIdentityValue(value: unknown): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error('Canonical identity value invalid: numbers must be finite');
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(serializeCanonicalIdentityValue).join(',')}]`;
  }
  if (isUnknownRecord(value)) {
    const properties = Object.keys(value)
      .filter((key) => Reflect.get(value, key) !== undefined)
      .sort()
      .map(
        (key) =>
          `${JSON.stringify(key)}:${serializeCanonicalIdentityValue(Reflect.get(value, key))}`,
      );
    return `{${properties.join(',')}}`;
  }
  throw new Error(`Canonical identity value invalid: unsupported ${typeof value}`);
}

/** Creates a stable namespaced SHA-256 identity independent of object key insertion order. */
export function createCanonicalIdentity(namespace: string, value: unknown): string {
  const normalizedNamespace = namespace.trim();
  if (!normalizedNamespace) {
    throw new Error('Canonical identity namespace invalid: expected a non-blank value');
  }
  const digest = createHash('sha256').update(serializeCanonicalIdentityValue(value)).digest('hex');
  return `${normalizedNamespace}:${digest}`;
}
