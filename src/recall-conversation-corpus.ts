import { readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';

import { readNodeErrorCode } from './read-node-error-code.js';

/** Metadata-only size and path for one physical session file in the recall corpus. */
export interface RecallConversationCorpusFile {
  sessionPath: string;
  sourceByteSize: number;
}

/** Metadata-only physical session count and source bytes; no session parsing or model work. */
export interface RecallConversationCorpusInspection {
  sessionCount: number;
  sourceByteSize: number;
}

/** Maximum physical sessions accepted by one optional first-index measurement sample. */
export const MAX_RECALL_FIRST_INDEX_SAMPLE_SESSION_COUNT = 10;

/** Lists physical session files recursively in deterministic path order without reading contents. */
export async function listRecallConversationSessionFiles(directory: string): Promise<string[]> {
  const files: string[] = [];
  async function visit(current: string): Promise<void> {
    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        return;
      }
      throw error;
    }
    for (const entry of entries) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        files.push(path);
      }
    }
  }
  await visit(directory);
  return files.sort();
}

/** Selects a deterministic bounded sample spanning the corpus file-size distribution. */
export function selectRecallConversationCorpusSample(
  files: readonly RecallConversationCorpusFile[],
  maximumSessionCount: number,
): RecallConversationCorpusFile[] {
  if (
    !Number.isInteger(maximumSessionCount) ||
    maximumSessionCount < 1 ||
    maximumSessionCount > MAX_RECALL_FIRST_INDEX_SAMPLE_SESSION_COUNT
  ) {
    throw new Error(
      `Recall first-index sample bound invalid: expected an integer from 1 through ${MAX_RECALL_FIRST_INDEX_SAMPLE_SESSION_COUNT}, received ${maximumSessionCount}`,
    );
  }
  if (files.length <= maximumSessionCount) {
    return [...files];
  }
  const bySize = [...files].sort(
    (left, right) =>
      left.sourceByteSize - right.sourceByteSize ||
      left.sessionPath.localeCompare(right.sessionPath, 'en'),
  );
  function readRequiredSampleFile(index: number): RecallConversationCorpusFile {
    const file = bySize[index];
    if (!file) {
      throw new Error(`Recall first-index sample selection missing file at index ${index}`);
    }
    return file;
  }
  if (maximumSessionCount === 1) {
    return [readRequiredSampleFile(Math.floor((bySize.length - 1) / 2))];
  }
  return Array.from({ length: maximumSessionCount }, (value, index) => {
    void value;
    const sampleIndex = Math.round((index * (bySize.length - 1)) / (maximumSessionCount - 1));
    return readRequiredSampleFile(sampleIndex);
  });
}

/** Inspects exact session-file sizes without parsing, tokenizing, embedding, or opening zvec. */
export async function inspectRecallConversationCorpus(sessionsDirectory: string): Promise<{
  inspection: RecallConversationCorpusInspection;
  files: RecallConversationCorpusFile[];
}> {
  const sessionPaths = await listRecallConversationSessionFiles(sessionsDirectory);
  const files = await Promise.all(
    sessionPaths.map(async (sessionPath) => ({
      sessionPath,
      sourceByteSize: (await stat(sessionPath)).size,
    })),
  );
  return {
    inspection: {
      sessionCount: files.length,
      sourceByteSize: files.reduce((total, file) => total + file.sourceByteSize, 0),
    },
    files,
  };
}
