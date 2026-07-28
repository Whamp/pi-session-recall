import { RecallManualMaintenanceTrigger } from './enums.js';
import type { RecallQualityGateDecision } from './recall-quality-gate.js';
import type {
  RecallConversationIndexOptions,
  RecallConversationService,
} from './recall-conversation-service.js';

interface RecallIndexCommandUi {
  setStatus(status?: string): void;
  notify(message: string, level: 'info' | 'warning'): void;
}

/** Guarded inputs for the production conversation-index slash command. */
export interface RecallIndexCommandOptions {
  argumentsText: string;
  qualityGateDecision: RecallQualityGateDecision;
  service: Pick<RecallConversationService, 'adoptLegacy' | 'collectRetired' | 'index' | 'rollback'>;
  ui: RecallIndexCommandUi;
}

type RecallIndexCommandAction =
  | 'adopt-legacy'
  | 'collect-retired'
  | 'incremental'
  | 'rebuild'
  | 'rollback';

function readRecallIndexCommandAction(argumentsText: string): RecallIndexCommandAction {
  const args = argumentsText.trim();
  if (!args) {
    return 'incremental';
  }
  if (args === '--rebuild') {
    return 'rebuild';
  }
  if (args === '--rollback') {
    return 'rollback';
  }
  if (args === '--adopt-legacy') {
    return 'adopt-legacy';
  }
  if (args === '--collect-retired') {
    return 'collect-retired';
  }
  throw new Error(
    `Recall index command arguments invalid: ${args}; usage: /pi-session-recall-index [--rebuild|--rollback|--adopt-legacy|--collect-retired]`,
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

/** Runs gated indexing or explicit rollback, legacy adoption, and retired-generation collection. */
export async function runRecallIndexCommand(options: RecallIndexCommandOptions): Promise<void> {
  const action = readRecallIndexCommandAction(options.argumentsText);
  if (action === 'rollback') {
    if (!options.service.rollback) {
      throw new Error('Recall generation rollback is unavailable in this service');
    }
    options.ui.setStatus('rolling back recall generation…');
    try {
      await options.service.rollback();
      options.ui.notify(
        'Recall generation rolled back; retained markers are pending replay',
        'warning',
      );
    } finally {
      options.ui.setStatus();
    }
    return;
  }
  if (action === 'collect-retired') {
    if (!options.service.collectRetired) {
      throw new Error('Recall retired generation collection is unavailable in this service');
    }
    options.ui.setStatus('collecting retired recall generations…');
    try {
      await options.service.collectRetired();
      options.ui.notify('Expired recall generations collected', 'info');
    } finally {
      options.ui.setStatus();
    }
    return;
  }
  if (action === 'adopt-legacy') {
    if (!options.service.adoptLegacy) {
      throw new Error('Recall legacy generation adoption is unavailable in this service');
    }
    options.ui.setStatus('adopting legacy recall generation…');
    try {
      await options.service.adoptLegacy();
      options.ui.notify(
        'Legacy recall generation adopted read-only; run --rebuild next',
        'warning',
      );
    } finally {
      options.ui.setStatus();
    }
    return;
  }
  assertRecallBackfillGatePassed(options.qualityGateDecision);
  const rebuild = action === 'rebuild';
  options.ui.setStatus(rebuild ? 'rebuilding conversations…' : 'indexing conversations…');
  try {
    const onProgress: NonNullable<RecallConversationIndexOptions['onProgress']> = (progress) => {
      options.ui.setStatus(
        `${rebuild ? 'rebuilding' : 'indexing'} ${progress.scannedSessions}/${progress.totalSessions}`,
      );
    };
    const indexOptions: RecallConversationIndexOptions = rebuild
      ? {
          rebuild: true,
          manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
          onProgress,
          optimize: true,
        }
      : {
          rebuild: false,
          manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
          onProgress,
        };
    const result = await options.service.index(indexOptions);
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
