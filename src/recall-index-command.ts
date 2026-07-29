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

type RecallIndexCommandService = Pick<
  RecallConversationService,
  | 'adoptLegacy'
  | 'collectRetired'
  | 'discardStagingIndexGeneration'
  | 'index'
  | 'readBackgroundIndexGenerationStatus'
  | 'resumeBackgroundIndexGeneration'
  | 'rollback'
  | 'startBackgroundIndexGeneration'
  | 'stopBackgroundIndexGeneration'
>;

type RecallIndexCommandAction =
  | 'adopt-legacy'
  | 'collect-retired'
  | 'discard'
  | 'incremental'
  | 'rebuild'
  | 'resume'
  | 'rollback'
  | 'status'
  | 'stop';

/** Guarded inputs for the production conversation-index slash command. */
export interface RecallIndexCommandOptions {
  argumentsText: string;
  qualityGateDecision: RecallQualityGateDecision;
  service: RecallIndexCommandService;
  ui: RecallIndexCommandUi;
}

function readRecallIndexCommandAction(argumentsText: string): RecallIndexCommandAction {
  const args = argumentsText.trim();
  if (!args) {
    return 'incremental';
  }
  const actions: Readonly<Record<string, RecallIndexCommandAction>> = {
    '--adopt-legacy': 'adopt-legacy',
    '--collect-retired': 'collect-retired',
    '--discard': 'discard',
    '--rebuild': 'rebuild',
    '--resume': 'resume',
    '--rollback': 'rollback',
    '--status': 'status',
    '--stop': 'stop',
  };
  const action = actions[args];
  if (action) {
    return action;
  }
  throw new Error(
    `Recall index command arguments invalid: ${args}; usage: /pi-session-recall-index [--rebuild|--status|--stop|--resume|--discard|--rollback|--adopt-legacy|--collect-retired]`,
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

function formatBackgroundIndexStatus(
  status: Awaited<ReturnType<RecallIndexCommandService['readBackgroundIndexGenerationStatus']>>,
): string {
  if (!status) {
    return 'Recall background index: no build recorded';
  }
  const parts = [
    `Recall background index ${status.processState}`,
    `generation ${status.generationId ?? 'pending'}`,
    `process ${status.processId}`,
    status.progress
      ? `progress ${status.progress.scannedSessions}/${status.progress.totalSessions}`
      : undefined,
    status.latestCheckpoint
      ? `checkpoint ${status.latestCheckpoint.checkpointedSessions}/${status.latestCheckpoint.totalSessions}`
      : undefined,
    status.latestActionableError ?? undefined,
  ];
  return parts.filter((part) => part !== undefined).join(' · ');
}

/** Runs incremental maintenance, detached rebuild control, or explicit generation recovery. */
export async function runRecallIndexCommand(options: RecallIndexCommandOptions): Promise<void> {
  const action = readRecallIndexCommandAction(options.argumentsText);

  if (action === 'status') {
    const status = await options.service.readBackgroundIndexGenerationStatus();
    options.ui.notify(
      formatBackgroundIndexStatus(status),
      status?.latestActionableError ? 'warning' : 'info',
    );
    return;
  }
  if (action === 'stop') {
    const status = await options.service.stopBackgroundIndexGeneration();
    options.ui.notify(formatBackgroundIndexStatus(status), 'info');
    return;
  }
  if (action === 'discard') {
    const discarded = await options.service.discardStagingIndexGeneration();
    options.ui.notify(
      discarded
        ? 'Recall staging generation discarded'
        : 'Recall staging generation: nothing to discard',
      'info',
    );
    return;
  }
  if (action === 'rollback') {
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
  if (action === 'rebuild' || action === 'resume') {
    options.ui.setStatus(
      action === 'rebuild' ? 'starting background rebuild…' : 'resuming rebuild…',
    );
    try {
      const status =
        action === 'rebuild'
          ? await options.service.startBackgroundIndexGeneration()
          : await options.service.resumeBackgroundIndexGeneration();
      options.ui.notify(formatBackgroundIndexStatus(status), 'info');
    } finally {
      options.ui.setStatus();
    }
    return;
  }

  options.ui.setStatus('indexing conversations…');
  try {
    const onProgress: NonNullable<RecallConversationIndexOptions['onProgress']> = (progress) => {
      options.ui.setStatus(`indexing ${progress.scannedSessions}/${progress.totalSessions}`);
    };
    const result = await options.service.index({
      rebuild: false,
      manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
      onProgress,
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
