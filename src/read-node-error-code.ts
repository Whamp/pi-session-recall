/** Reads a Node.js `error.code` string without assuming the caught value is an Error object. */
export function readNodeErrorCode(error: unknown): string | undefined {
  if (!error || typeof error !== 'object' || !('code' in error)) {
    return undefined;
  }
  const code = Reflect.get(error, 'code');
  return typeof code === 'string' ? code : undefined;
}
