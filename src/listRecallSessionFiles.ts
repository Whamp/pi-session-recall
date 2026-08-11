import { readdir } from 'node:fs/promises';
import { join } from 'node:path';

import { readNodeErrorCode } from './read-node-error-code.js';

/** Lists sorted physical session JSONL files recursively beneath one session-store root. */
export async function listRecallSessionFiles(sessionsDirectory: string): Promise<string[]> {
  const paths: string[] = [];

  async function visit(directory: string): Promise<void> {
    let entries;
    try {
      entries = await readdir(directory, { withFileTypes: true });
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        return;
      }
      throw error;
    }
    for (const entry of entries) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        paths.push(path);
      }
    }
  }

  await visit(sessionsDirectory);
  return paths.sort();
}
