import { writeFile } from 'node:fs/promises';
import { emitKeypressEvents } from 'node:readline';

import { measureIncrementalIngestionPrototype } from './incremental-ingestion-measurements.js';
import { measureIncrementalWorkerPrototype } from './incremental-worker-measurement.js';
import {
  applyIncrementalIngestionPrototypeAction,
  createIncrementalIngestionPrototypeState,
  summarizeIncrementalIngestionPrototypeState,
  type IncrementalIngestionPrototypeAction,
  type IncrementalIngestionPrototypeState,
} from './incremental-ingestion-state.js';

const RESULT_PATH = new URL('./prototype-results.json', import.meta.url);
const BOLD = '\u001b[1m';
const DIM = '\u001b[2m';
const RESET = '\u001b[0m';

interface PrototypeScenarioResult {
  name: string;
  actions: IncrementalIngestionPrototypeAction[];
  finalState: Record<string, unknown>;
  checks: Record<string, boolean>;
}

function runPrototypeActions(
  actions: IncrementalIngestionPrototypeAction[],
): IncrementalIngestionPrototypeState {
  return actions.reduce(
    applyIncrementalIngestionPrototypeAction,
    createIncrementalIngestionPrototypeState(),
  );
}

function runPrototypeScenarios(): PrototypeScenarioResult[] {
  const compactionActions: IncrementalIngestionPrototypeAction[] = [
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'compact', runtimeId: 'runtime_a', keepRecentEntryCount: 1 },
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'compact', runtimeId: 'runtime_a', keepRecentEntryCount: 1 },
    { type: 'worker_commit', generationId: 'generation_1', crashAfterEvidence: false },
  ];
  const compactionState = runPrototypeActions(compactionActions);
  const eligibleCount = Object.values(compactionState.entries).filter(
    (entry) => entry.eligible,
  ).length;

  const branchActions: IncrementalIngestionPrototypeAction[] = [
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'branch', runtimeId: 'runtime_a', newLeafId: 'entry_1' },
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'worker_commit', generationId: 'generation_1', crashAfterEvidence: false },
  ];
  const branchState = runPrototypeActions(branchActions);

  const crashReplayActions: IncrementalIngestionPrototypeAction[] = [
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'depart', runtimeId: 'runtime_a' },
    { type: 'worker_commit', generationId: 'generation_1', crashAfterEvidence: true },
    { type: 'worker_commit', generationId: 'generation_1', crashAfterEvidence: false },
  ];
  const crashReplayState = runPrototypeActions(crashReplayActions);

  const rebuildActions: IncrementalIngestionPrototypeAction[] = [
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'depart', runtimeId: 'runtime_a' },
    { type: 'worker_commit', generationId: 'generation_1', crashAfterEvidence: false },
    { type: 'start_rebuild', generationId: 'generation_2' },
    { type: 'arrive', runtimeId: 'runtime_b' },
    { type: 'append', runtimeId: 'runtime_b' },
    { type: 'depart', runtimeId: 'runtime_b' },
    { type: 'worker_commit', generationId: 'generation_2', crashAfterEvidence: false },
    { type: 'cutover_rebuild' },
  ];
  const rebuildState = runPrototypeActions(rebuildActions);

  const deletionActions: IncrementalIngestionPrototypeAction[] = [
    { type: 'append', runtimeId: 'runtime_a' },
    { type: 'depart', runtimeId: 'runtime_a' },
    { type: 'worker_commit', generationId: 'generation_1', crashAfterEvidence: false },
    { type: 'confirm_source_deletion', runtimeId: 'runtime_a' },
    { type: 'worker_commit', generationId: 'generation_1', crashAfterEvidence: false },
  ];
  const deletionState = runPrototypeActions(deletionActions);

  return [
    {
      name: 'repeated_compaction',
      actions: compactionActions,
      finalState: summarizeIncrementalIngestionPrototypeState(compactionState),
      checks: {
        eligibleEvidenceRemainsMonotonic: eligibleCount >= 4,
        completedMarkersRemoved: compactionState.markers.length === 0,
      },
    },
    {
      name: 'branch_exit',
      actions: branchActions,
      finalState: summarizeIncrementalIngestionPrototypeState(branchState),
      checks: {
        abandonedEntriesEligible:
          branchState.entries.entry_2?.eligible === true &&
          branchState.entries.entry_3?.eligible === true,
        newBranchTailNotEligible: branchState.entries.entry_4?.eligible === false,
      },
    },
    {
      name: 'worker_crash_replay',
      actions: crashReplayActions,
      finalState: summarizeIncrementalIngestionPrototypeState(crashReplayState),
      checks: {
        markerRetainedAcrossCrashThenRemoved: crashReplayState.markers.length === 0,
        evidenceIdempotentlyPresent:
          crashReplayState.generations.generation_1?.evidenceEntryIds.length === 1,
        failureClearedAfterReplay: crashReplayState.workerFailure === null,
      },
    },
    {
      name: 'rebuild_generation_cutover',
      actions: rebuildActions,
      finalState: summarizeIncrementalIngestionPrototypeState(rebuildState),
      checks: {
        newGenerationActivated: rebuildState.activeGenerationId === 'generation_2',
        markersAcknowledgedByBothGenerations: rebuildState.markers.length === 0,
        postBaselineEvidencePresent:
          rebuildState.generations.generation_2?.evidenceEntryIds.includes('entry_2') === true,
      },
    },
    {
      name: 'confirmed_source_deletion',
      actions: deletionActions,
      finalState: summarizeIncrementalIngestionPrototypeState(deletionState),
      checks: {
        sourceDeletionConfirmed: deletionState.sourceDeletionConfirmed,
        searchableEvidenceRemoved:
          deletionState.generations.generation_1?.evidenceEntryIds.length === 0,
      },
    },
  ];
}

function renderPrototypeState(state: IncrementalIngestionPrototypeState): void {
  console.clear();
  process.stdout.write(`${BOLD}Incremental Recall Ingestion Prototype${RESET}\n`);
  process.stdout.write(`${DIM}Throwaway state model; no production data${RESET}\n\n`);
  process.stdout.write(
    `${JSON.stringify(summarizeIncrementalIngestionPrototypeState(state), null, 2)}\n\n`,
  );
  process.stdout.write(
    `${BOLD}[a]${RESET} append  ${BOLD}[c]${RESET} compact  ${BOLD}[b]${RESET} branch to root  ` +
      `${BOLD}[d]${RESET} depart  ${BOLD}[u]${RESET} quiesce\n` +
      `${BOLD}[w]${RESET} worker commit  ${BOLD}[x]${RESET} worker crash  ` +
      `${BOLD}[r]${RESET} start rebuild  ${BOLD}[k]${RESET} drain/cutover  ` +
      `${BOLD}[z]${RESET} confirm deletion  ${BOLD}[q]${RESET} quit\n`,
  );
}

function runInteractivePrototype(): void {
  if (!process.stdin.isTTY) {
    return;
  }
  emitKeypressEvents(process.stdin);
  process.stdin.setRawMode(true);
  process.stdin.resume();
  let state = createIncrementalIngestionPrototypeState();
  renderPrototypeState(state);
  process.stdin.on('keypress', (_input, key) => {
    try {
      if (key.name === 'q' || (key.ctrl && key.name === 'c')) {
        process.stdin.setRawMode(false);
        process.stdin.pause();
        return;
      }
      let action: IncrementalIngestionPrototypeAction | undefined;
      if (key.name === 'a') action = { type: 'append', runtimeId: 'runtime_a' };
      if (key.name === 'c') {
        action = { type: 'compact', runtimeId: 'runtime_a', keepRecentEntryCount: 1 };
      }
      if (key.name === 'b')
        action = { type: 'branch', runtimeId: 'runtime_a', newLeafId: 'entry_1' };
      if (key.name === 'd') action = { type: 'depart', runtimeId: 'runtime_a' };
      if (key.name === 'u') action = { type: 'quiesce', runtimeId: 'runtime_a' };
      if (key.name === 'w') {
        action = {
          type: 'worker_commit',
          generationId: state.activeGenerationId,
          crashAfterEvidence: false,
        };
      }
      if (key.name === 'x') {
        action = {
          type: 'worker_commit',
          generationId: state.activeGenerationId,
          crashAfterEvidence: true,
        };
      }
      if (key.name === 'r') action = { type: 'start_rebuild', generationId: 'generation_2' };
      if (key.name === 'z') action = { type: 'confirm_source_deletion', runtimeId: 'runtime_a' };
      if (key.name === 'k' && state.buildingGenerationId !== null) {
        state = applyIncrementalIngestionPrototypeAction(state, {
          type: 'worker_commit',
          generationId: state.buildingGenerationId,
          crashAfterEvidence: false,
        });
        action = { type: 'cutover_rebuild' };
      }
      if (action) {
        state = applyIncrementalIngestionPrototypeAction(state, action);
      }
    } catch (error) {
      state.workerFailure = error instanceof Error ? error.message : String(error);
    }
    renderPrototypeState(state);
  });
}

async function main(): Promise<void> {
  const measurements = await measureIncrementalIngestionPrototype();
  const workerMeasurements = await measureIncrementalWorkerPrototype();
  const scenarios = runPrototypeScenarios();
  const report = {
    generatedAt: new Date().toISOString(),
    measurements,
    workerMeasurements,
    scenarios,
  };
  await writeFile(RESULT_PATH, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!process.argv.includes('--report-only')) {
    runInteractivePrototype();
  }
}

await main();
