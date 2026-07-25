import type { RecallQualityGateDecision } from './recall-quality-gate.js';
import type { RecallConversationService } from './recall-conversation-service.js';

interface RecallIndexCommandUi {
  setStatus(status?: string): void;
  notify(message: string, level: 'info' | 'warning'): void;
}

/** Guarded inputs for the production conversation-index slash command. */
export interface RecallIndexCommandOptions {
  argumentsText: string;
  qualityGateDecision: RecallQualityGateDecision;
  service: Pick<RecallConversationService, 'index'>;
  ui: RecallIndexCommandUi;
}

function readRecallIndexRebuildFlag(argumentsText: string): boolean {
  const args = argumentsText.trim();
  if (!args) {
    return false;
  }
  if (args === '--rebuild') {
    return true;
  }
  throw new Error(
    `Recall index command arguments invalid: ${args}; usage: /pi-session-recall-index [--rebuild]`,
  );
}

function assertRecallBackfillGatePassed(decision: RecallQualityGateDecision): void {
  if (decision.automatedGatePassed && decision.selectedPolicy) {
    return;
  }
  const blockers =
    decision.blockers.length > 0
      ? decision.blockers.join('; ')
      : 'no measured chunk, candidate, and final-result policy passed';
  throw new Error(
    `Recall full backfill blocked because the quality gate has not passed: ${blockers}. Run npm run evaluate:recall and review docs/evaluation/recall-quality-report.md before indexing production sessions.`,
  );
}

/** Runs explicit incremental or rebuilding index maintenance only after the quality gate passes. */
export async function runRecallIndexCommand(options: RecallIndexCommandOptions): Promise<void> {
  assertRecallBackfillGatePassed(options.qualityGateDecision);
  const rebuild = readRecallIndexRebuildFlag(options.argumentsText);
  options.ui.setStatus(rebuild ? 'rebuilding conversations…' : 'indexing conversations…');
  try {
    const result = await options.service.index({
      rebuild,
      onProgress(progress) {
        options.ui.setStatus(
          `${rebuild ? 'rebuilding' : 'indexing'} ${progress.scannedSessions}/${progress.totalSessions}`,
        );
      },
      optimize: true,
    });
    const failures = result.indexSummary.failedSessions.length;
    const message = [
      `Recall index ready: ${result.totalChunks} chunks`,
      `${result.indexSummary.cacheHits} cache hits`,
      `${result.indexSummary.newlyEmbeddedChunks} newly embedded`,
      `${result.indexSummary.embeddingRequestCount} embedding requests`,
      `${result.indexSummary.deletedChunks} removed`,
      failures > 0 ? `${failures} failed sessions` : undefined,
    ]
      .filter((part) => part !== undefined)
      .join(' · ');
    options.ui.notify(message, failures > 0 ? 'warning' : 'info');
  } finally {
    options.ui.setStatus();
  }
}
