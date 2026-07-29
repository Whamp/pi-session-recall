import { open } from 'node:fs/promises';

/** Durably synchronizes directory-entry changes before a recall spool operation returns. */
export async function syncRecallDirectory(directoryPath: string): Promise<void> {
  const directory = await open(directoryPath, 'r');
  try {
    await directory.sync();
  } finally {
    await directory.close();
  }
}
