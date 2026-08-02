import { performance } from 'node:perf_hooks';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
  type RecallConversationService,
} from './recall-conversation-service.js';

const PSR_USAGE = 'psr usage: psr index [--rebuild]';

/** Replaceable process boundaries for the standalone `psr` command. */
export interface PsrCliDependencies {
  loadConfig: () => Promise<RecallConversationConfig>;
  createService: (config: RecallConversationConfig) => RecallConversationService;
  writeOutput: (text: string) => void;
  writeProgress: (text: string) => void;
  getMonotonicTimeMs: () => number;
}

const DEFAULT_PSR_CLI_DEPENDENCIES: PsrCliDependencies = {
  loadConfig: loadRecallConversationConfig,
  createService: createRecallConversationService,
  writeOutput(text) {
    process.stdout.write(text);
  },
  writeProgress(text) {
    process.stderr.write(text);
  },
  getMonotonicTimeMs: performance.now.bind(performance),
};

function formatRecallIndexProgress(event: RecallIndexProgressEvent): string {
  switch (event.kind) {
    case 'preparing':
      return 'Preparing recall index...';
    case 'waiting-for-write-lock':
      return 'Waiting for recall index write lock...';
    case 'discovering-physical-session-files':
      return 'Discovering physical session files...';
    case 'planning-maintenance-workset':
      return 'Planning maintenance workset...';
    case 'maintenance-workset-planned': {
      const plannedFiles = event.newFiles + event.changedFiles + event.missingFiles;
      if (event.rebuild) {
        return `Rebuild maintenance workset: ${event.discoveredFiles} physical session files discovered; all ${event.discoveredFiles} scheduled for indexing.`;
      }
      if (plannedFiles === 0) {
        return `Maintenance workset: ${event.discoveredFiles} physical session files discovered; no files require indexing or removal.`;
      }
      return `Maintenance workset: ${event.discoveredFiles} physical session files discovered; ${event.newFiles} new, ${event.changedFiles} changed, ${event.missingFiles} missing.`;
    }
    case 'indexing-changed-physical-session-files':
      return 'Indexing changed physical session files...';
    case 'indexing-maintenance-workset':
      return `Indexing ${event.completedFiles}/${event.totalFiles} files · ${event.indexedSessions} indexed sessions · ${event.newlyEmbeddedDocuments} embedded documents · ${event.reusedVectors} reused vectors · ${event.deletedDocuments} deleted documents · ${event.failedSessions} failed sessions · ${event.sessionPath}`;
    case 'physical-session-file-failed':
      return `Warning: physical session file failed: ${event.sessionPath}`;
    case 'optimizing-collection':
      return 'Optimizing recall collection...';
    case 'completed':
      return 'Recall index maintenance completed.';
  }
}

/** Runs the complete standalone CLI; only explicit incremental indexing and rebuild can write. */
export async function runPsrCli(
  argumentsList: readonly string[],
  dependencies: PsrCliDependencies = DEFAULT_PSR_CLI_DEPENDENCIES,
): Promise<number> {
  if (argumentsList.length === 0 || argumentsList[0] === '--help') {
    dependencies.writeOutput(`${PSR_USAGE}\n`);
    return 0;
  }
  const rebuild = argumentsList.length === 2 && argumentsList[1] === '--rebuild';
  if (argumentsList[0] !== 'index' || (argumentsList.length !== 1 && !rebuild)) {
    throw new Error(PSR_USAGE);
  }

  let lastProgressKind: RecallIndexProgressEvent['kind'] | undefined;
  let lastProgressTimeMs = Number.NEGATIVE_INFINITY;
  const reportProgress = (event: RecallIndexProgressEvent): void => {
    const currentTimeMs = dependencies.getMonotonicTimeMs();
    if (
      event.kind === 'indexing-maintenance-workset' &&
      event.completedFiles !== event.totalFiles &&
      lastProgressKind === event.kind &&
      currentTimeMs - lastProgressTimeMs < 1_000
    ) {
      return;
    }
    dependencies.writeProgress(`${formatRecallIndexProgress(event)}\n`);
    lastProgressKind = event.kind;
    lastProgressTimeMs = currentTimeMs;
  };

  reportProgress({ kind: 'preparing' });
  const config = await dependencies.loadConfig();
  const result = await dependencies.createService(config).index({
    rebuild,
    optimize: true,
    onProgress: reportProgress,
  });
  const summary = result.indexSummary;
  dependencies.writeOutput(
    [
      `Indexed ${summary.indexedSessions} of ${summary.scannedSessions} sessions`,
      `removed ${summary.removedSessions}`,
      `embedded ${summary.newlyEmbeddedChunks}`,
      `reused ${summary.reusedVectors} vectors`,
      `deleted ${summary.deletedChunks} documents`,
      `${result.totalChunks} searchable documents`,
      `${summary.failedSessions.length} failed sessions`,
    ].join(' · ') + '\n',
  );
  for (const failure of summary.failedSessions) {
    dependencies.writeOutput(`Failed: ${failure.sessionPath}: ${failure.error}\n`);
  }
  return summary.failedSessions.length === 0 ? 0 : 1;
}
