import { accessSync, constants } from 'node:fs';
import { join } from 'node:path';

let cachedFlockPath: string | undefined;

/** Resolves and caches the first executable flock binary found on the system. */
export function resolveRecallFlockExecutable(): string {
  if (cachedFlockPath !== undefined) {
    return cachedFlockPath;
  }
  const candidates = [
    '/usr/bin/flock',
    ...(process.env.PATH ?? '')
      .split(':')
      .filter(Boolean)
      .map((dir) => join(dir, 'flock')),
    '/opt/homebrew/opt/util-linux/bin/flock',
    '/usr/local/opt/util-linux/bin/flock',
  ];
  for (const candidate of candidates) {
    try {
      accessSync(candidate, constants.X_OK);
      cachedFlockPath = candidate;
      return candidate;
    } catch {
      // not found or not executable; try next candidate
    }
  }
  throw new Error(
    'Recall flock executable not found; install util-linux (Linux: apt/yum install util-linux, macOS: brew install util-linux)',
  );
}
