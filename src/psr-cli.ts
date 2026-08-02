import { performance } from 'node:perf_hooks';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
  type RecallConversationIndexResult,
  type RecallConversationService,
} from './recall-conversation-service.js';

const PSR_USAGE = 'psr usage: psr index [--rebuild] [--compact]';
const ENGLISH_INTEGER_FORMAT = new Intl.NumberFormat('en-US');

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

interface RecallIndexProgressTiming {
  elapsedMs: number;
  indexingElapsedMs?: number;
}

function formatCountedNoun(count: number, singularNoun: string): string {
  const noun = count === 1 ? singularNoun : `${singularNoun}s`;
  return `${ENGLISH_INTEGER_FORMAT.format(count)} ${noun}`;
}

function formatDuration(durationMs: number): string {
  const totalSeconds = Math.max(0, Math.round(durationMs / 1_000));
  const hours = Math.floor(totalSeconds / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, '0')}m ${String(seconds).padStart(2, '0')}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
  }
  return `${seconds}s`;
}

function formatRemainingTime(
  event: Extract<RecallIndexProgressEvent, { kind: 'indexing-maintenance-workset' }>,
  indexingElapsedMs: number | undefined,
): string {
  if (event.completedFiles === event.totalFiles) {
    return 'finishing';
  }
  const healthyCompletedFiles = event.completedFiles - event.failedSessions;
  if (healthyCompletedFiles === 0 || indexingElapsedMs === undefined || indexingElapsedMs < 5_000) {
    return 'estimating time remaining';
  }
  const remainingMs =
    (indexingElapsedMs / healthyCompletedFiles) * (event.totalFiles - event.completedFiles);
  return `about ${formatDuration(remainingMs)} remaining`;
}

function formatCompactRecallIndexSummary(result: RecallConversationIndexResult): string {
  const summary = result.indexSummary;
  const lines = [
    [
      `Indexed ${summary.indexedSessions} of ${summary.scannedSessions} sessions`,
      `removed ${summary.removedSessions}`,
      `embedded ${summary.newlyEmbeddedChunks}`,
      `reused ${summary.reusedVectors} vectors`,
      `deleted ${summary.deletedChunks} documents`,
      `${result.totalChunks} searchable documents`,
      `${summary.failedSessions.length} failed sessions`,
    ].join(' · '),
    ...summary.failedSessions.map((failure) => `Failed: ${failure.sessionPath}: ${failure.error}`),
  ];
  return `${lines.join('\n')}\n`;
}

function formatReadableRecallIndexSummary(
  result: RecallConversationIndexResult,
  elapsedMs: number,
): string {
  const summary = result.indexSummary;
  const lines = [
    'Summary',
    `  Elapsed: ${formatDuration(elapsedMs)}`,
    `  Sessions: ${ENGLISH_INTEGER_FORMAT.format(summary.indexedSessions)} indexed of ${ENGLISH_INTEGER_FORMAT.format(summary.scannedSessions)} scanned; ${ENGLISH_INTEGER_FORMAT.format(summary.removedSessions)} removed`,
    `  Documents: ${ENGLISH_INTEGER_FORMAT.format(summary.newlyEmbeddedChunks)} embedded; ${ENGLISH_INTEGER_FORMAT.format(summary.reusedVectors)} vectors reused; ${ENGLISH_INTEGER_FORMAT.format(summary.deletedChunks)} deleted`,
    `  Searchable documents: ${ENGLISH_INTEGER_FORMAT.format(result.totalChunks)}`,
    `  Failed sessions: ${ENGLISH_INTEGER_FORMAT.format(summary.failedSessions.length)}`,
  ];
  if (summary.failedSessions.length > 0) {
    lines.push('', 'Failures');
    for (const failure of summary.failedSessions) {
      lines.push(`  ${failure.sessionPath}`, `    ${failure.error}`);
    }
  }
  return `${lines.join('\n')}\n`;
}

function formatRecallIndexProgress(
  event: RecallIndexProgressEvent,
  timing: RecallIndexProgressTiming,
): string {
  switch (event.kind) {
    case 'preparing':
      return 'Recall index maintenance\n\nPreparing recall index...';
    case 'waiting-for-write-lock':
      return 'Waiting for recall index write lock...';
    case 'discovering-physical-session-files':
      return 'Discovering physical session files...';
    case 'planning-maintenance-workset':
      return 'Planning maintenance workset...';
    case 'maintenance-workset-planned': {
      const plannedFiles = event.newFiles + event.changedFiles + event.missingFiles;
      const lines = [
        '',
        `Found ${formatCountedNoun(event.discoveredFiles, 'physical session file')}.`,
      ];
      if (event.rebuild) {
        lines.push(
          `Rebuild maintenance workset: ${formatCountedNoun(plannedFiles, 'file')}; all ${ENGLISH_INTEGER_FORMAT.format(plannedFiles)} scheduled for indexing.`,
        );
      } else if (plannedFiles === 0) {
        lines.push('No files require indexing or removal.');
      } else {
        lines.push(
          `Maintenance workset: ${formatCountedNoun(plannedFiles, 'file')} (${ENGLISH_INTEGER_FORMAT.format(event.newFiles)} new, ${ENGLISH_INTEGER_FORMAT.format(event.changedFiles)} changed, ${ENGLISH_INTEGER_FORMAT.format(event.missingFiles)} missing).`,
        );
      }
      if (plannedFiles > 0) {
        lines.push('Estimated time: calculating after the first file completes.');
      }
      return lines.join('\n');
    }
    case 'indexing-changed-physical-session-files':
      return '\nIndexing maintenance workset...';
    case 'indexing-maintenance-workset':
      return `  ${event.completedFiles}/${event.totalFiles} files · ${formatDuration(timing.elapsedMs)} elapsed · ${formatRemainingTime(event, timing.indexingElapsedMs)} · ${ENGLISH_INTEGER_FORMAT.format(event.newlyEmbeddedDocuments)} embedded · ${ENGLISH_INTEGER_FORMAT.format(event.reusedVectors)} reused · ${ENGLISH_INTEGER_FORMAT.format(event.deletedDocuments)} deleted · ${ENGLISH_INTEGER_FORMAT.format(event.failedSessions)} failed`;
    case 'physical-session-file-failed':
      return `  Warning: physical session file failed: ${event.sessionPath}`;
    case 'optimizing-collection':
      return '\nOptimizing searchable collection...';
    case 'completed':
      return `\nCompleted in ${formatDuration(timing.elapsedMs)}.`;
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
  const flags = argumentsList.slice(1);
  const distinctFlags = new Set(flags);
  const validFlags = flags.every((flag) => flag === '--rebuild' || flag === '--compact');
  if (argumentsList[0] !== 'index' || !validFlags || distinctFlags.size !== flags.length) {
    throw new Error(PSR_USAGE);
  }
  const rebuild = distinctFlags.has('--rebuild');
  const compact = distinctFlags.has('--compact');

  let commandStartedAtMs: number | undefined;
  let indexingStartedAtMs: number | undefined;
  let lastProgressKind: RecallIndexProgressEvent['kind'] | undefined;
  let lastProgressTimeMs = Number.NEGATIVE_INFINITY;
  const reportProgress = (event: RecallIndexProgressEvent): void => {
    const currentTimeMs = dependencies.getMonotonicTimeMs();
    commandStartedAtMs ??= currentTimeMs;
    if (
      indexingStartedAtMs === undefined &&
      (event.kind === 'indexing-changed-physical-session-files' ||
        event.kind === 'indexing-maintenance-workset')
    ) {
      indexingStartedAtMs = currentTimeMs;
    }
    if (
      event.kind === 'indexing-maintenance-workset' &&
      event.completedFiles !== event.totalFiles &&
      lastProgressKind === event.kind &&
      currentTimeMs - lastProgressTimeMs < 1_000
    ) {
      return;
    }
    dependencies.writeProgress(
      `${formatRecallIndexProgress(event, {
        elapsedMs: currentTimeMs - commandStartedAtMs,
        ...(indexingStartedAtMs === undefined
          ? {}
          : { indexingElapsedMs: currentTimeMs - indexingStartedAtMs }),
      })}\n`,
    );
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
    compact
      ? formatCompactRecallIndexSummary(result)
      : formatReadableRecallIndexSummary(
          result,
          lastProgressTimeMs - (commandStartedAtMs ?? lastProgressTimeMs),
        ),
  );
  return summary.failedSessions.length === 0 ? 0 : 1;
}
