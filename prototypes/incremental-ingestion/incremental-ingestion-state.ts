/** Throwaway pure state model for incremental recall ingestion transitions. */

export type PrototypeMarkerKind =
  | 'activity'
  | 'compaction'
  | 'branch_exit'
  | 'departure'
  | 'arrival'
  | 'deletion';

export interface PrototypeEntryState {
  id: string;
  parentId: string | null;
  eligible: boolean;
  deleted: boolean;
}

export interface PrototypeMarkerState {
  id: string;
  kind: PrototypeMarkerKind;
  runtimeId: string;
  runtimeSequence: number;
  createdAtSeconds: number;
}

export interface PrototypeGenerationState {
  id: string;
  evidenceEntryIds: string[];
  projectedEligibleEntryIds: string[];
  pendingMarkerIds: string[];
}

export interface IncrementalIngestionPrototypeState {
  nowSeconds: number;
  nextEntryNumber: number;
  nextMarkerNumber: number;
  runtimeSequences: Record<string, number>;
  entries: Record<string, PrototypeEntryState>;
  currentLeafId: string | null;
  activeContextEntryIds: string[];
  markers: PrototypeMarkerState[];
  generations: Record<string, PrototypeGenerationState>;
  activeGenerationId: string;
  buildingGenerationId: string | null;
  workerFailure: string | null;
  sourceDeletionConfirmed: boolean;
}

export type IncrementalIngestionPrototypeAction =
  | { type: 'append'; runtimeId: string }
  | { type: 'compact'; runtimeId: string; keepRecentEntryCount: number }
  | { type: 'branch'; runtimeId: string; newLeafId: string | null }
  | { type: 'depart'; runtimeId: string }
  | { type: 'arrive'; runtimeId: string }
  | { type: 'quiesce'; runtimeId: string }
  | { type: 'advance_time'; seconds: number }
  | { type: 'worker_commit'; generationId: string; crashAfterEvidence: boolean }
  | { type: 'start_rebuild'; generationId: string }
  | { type: 'cutover_rebuild' }
  | { type: 'confirm_source_deletion'; runtimeId: string };

/** Creates the smallest state needed to exercise incremental recall transitions. */
export function createIncrementalIngestionPrototypeState(): IncrementalIngestionPrototypeState {
  return {
    nowSeconds: 0,
    nextEntryNumber: 1,
    nextMarkerNumber: 1,
    runtimeSequences: {},
    entries: {},
    currentLeafId: null,
    activeContextEntryIds: [],
    markers: [],
    generations: {
      generation_1: {
        id: 'generation_1',
        evidenceEntryIds: [],
        projectedEligibleEntryIds: [],
        pendingMarkerIds: [],
      },
    },
    activeGenerationId: 'generation_1',
    buildingGenerationId: null,
    workerFailure: null,
    sourceDeletionConfirmed: false,
  };
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function publishPrototypeMarker(
  state: IncrementalIngestionPrototypeState,
  runtimeId: string,
  kind: PrototypeMarkerKind,
): void {
  const runtimeSequence = (state.runtimeSequences[runtimeId] ?? 0) + 1;
  state.runtimeSequences[runtimeId] = runtimeSequence;
  const marker: PrototypeMarkerState = {
    id: `marker_${state.nextMarkerNumber}`,
    kind,
    runtimeId,
    runtimeSequence,
    createdAtSeconds: state.nowSeconds,
  };
  state.nextMarkerNumber += 1;
  state.markers.push(marker);
  const targetGenerationId = state.buildingGenerationId ?? state.activeGenerationId;
  const targetGeneration = state.generations[targetGenerationId];
  if (!targetGeneration) {
    throw new Error(`Prototype marker target generation missing: ${targetGenerationId}`);
  }
  targetGeneration.pendingMarkerIds.push(marker.id);
}

function findAncestorEntryIds(
  entries: Record<string, PrototypeEntryState>,
  leafId: string | null,
): string[] {
  const reversePath: string[] = [];
  let entryId = leafId;
  while (entryId !== null) {
    const entry = entries[entryId];
    if (!entry) {
      break;
    }
    reversePath.push(entry.id);
    entryId = entry.parentId;
  }
  return reversePath.reverse();
}

function markEntriesEligible(state: IncrementalIngestionPrototypeState, entryIds: string[]): void {
  for (const entryId of entryIds) {
    const entry = state.entries[entryId];
    if (entry && !entry.deleted) {
      entry.eligible = true;
    }
  }
}

function removeCompletedMarkers(state: IncrementalIngestionPrototypeState): void {
  const pending = new Set(
    Object.values(state.generations).flatMap((generation) => generation.pendingMarkerIds),
  );
  state.markers = state.markers.filter((marker) => pending.has(marker.id));
}

function applyWorkerCommit(
  state: IncrementalIngestionPrototypeState,
  generationId: string,
  crashAfterEvidence: boolean,
): void {
  const generation = state.generations[generationId];
  if (!generation) {
    throw new Error(`Prototype generation missing: ${generationId}`);
  }
  if (state.buildingGenerationId !== null && generationId === state.activeGenerationId) {
    throw new Error('Prototype active-generation commits freeze during rebuild');
  }
  const eligibleEntryIds = Object.values(state.entries)
    .filter((entry) => entry.eligible && !entry.deleted)
    .map((entry) => entry.id);
  generation.evidenceEntryIds = unique([...generation.evidenceEntryIds, ...eligibleEntryIds]);
  if (crashAfterEvidence) {
    state.workerFailure = `worker crashed after evidence upsert for ${generationId}`;
    return;
  }
  generation.projectedEligibleEntryIds = unique(eligibleEntryIds);
  generation.pendingMarkerIds = [];
  state.workerFailure = null;
  removeCompletedMarkers(state);
}

/** Applies one prototype ingestion action and returns a new inspectable state. */
export function applyIncrementalIngestionPrototypeAction(
  previous: IncrementalIngestionPrototypeState,
  action: IncrementalIngestionPrototypeAction,
): IncrementalIngestionPrototypeState {
  const state = structuredClone(previous);
  if (action.type === 'append') {
    const entryId = `entry_${state.nextEntryNumber}`;
    state.nextEntryNumber += 1;
    state.entries[entryId] = {
      id: entryId,
      parentId: state.currentLeafId,
      eligible: false,
      deleted: false,
    };
    state.currentLeafId = entryId;
    state.activeContextEntryIds = findAncestorEntryIds(state.entries, entryId).filter(
      (id) => !state.entries[id]?.eligible,
    );
    publishPrototypeMarker(state, action.runtimeId, 'activity');
    return state;
  }
  if (action.type === 'compact') {
    const keepCount = Math.max(action.keepRecentEntryCount, 0);
    const forgotten =
      keepCount === 0
        ? [...state.activeContextEntryIds]
        : state.activeContextEntryIds.slice(0, -keepCount);
    markEntriesEligible(state, forgotten);
    state.activeContextEntryIds =
      keepCount === 0 ? [] : state.activeContextEntryIds.slice(-keepCount);
    const summaryId = `entry_${state.nextEntryNumber}`;
    state.nextEntryNumber += 1;
    state.entries[summaryId] = {
      id: summaryId,
      parentId: state.currentLeafId,
      eligible: true,
      deleted: false,
    };
    state.currentLeafId = summaryId;
    publishPrototypeMarker(state, action.runtimeId, 'compaction');
    return state;
  }
  if (action.type === 'branch') {
    const oldPath = findAncestorEntryIds(state.entries, state.currentLeafId);
    const newPath = findAncestorEntryIds(state.entries, action.newLeafId);
    const newPathSet = new Set(newPath);
    markEntriesEligible(
      state,
      oldPath.filter((entryId) => !newPathSet.has(entryId)),
    );
    state.currentLeafId = action.newLeafId;
    state.activeContextEntryIds = newPath.filter((entryId) => !state.entries[entryId]?.eligible);
    publishPrototypeMarker(state, action.runtimeId, 'branch_exit');
    return state;
  }
  if (action.type === 'depart' || action.type === 'quiesce') {
    markEntriesEligible(state, state.activeContextEntryIds);
    state.activeContextEntryIds = [];
    publishPrototypeMarker(
      state,
      action.runtimeId,
      action.type === 'depart' ? 'departure' : 'activity',
    );
    return state;
  }
  if (action.type === 'arrive') {
    publishPrototypeMarker(state, action.runtimeId, 'arrival');
    return state;
  }
  if (action.type === 'advance_time') {
    state.nowSeconds += Math.max(action.seconds, 0);
    return state;
  }
  if (action.type === 'worker_commit') {
    applyWorkerCommit(state, action.generationId, action.crashAfterEvidence);
    return state;
  }
  if (action.type === 'start_rebuild') {
    if (state.buildingGenerationId !== null) {
      throw new Error('Prototype rebuild already active');
    }
    const activeGeneration = state.generations[state.activeGenerationId];
    if (!activeGeneration) {
      throw new Error(`Prototype active generation missing: ${state.activeGenerationId}`);
    }
    state.generations[action.generationId] = {
      id: action.generationId,
      evidenceEntryIds: [...activeGeneration.evidenceEntryIds],
      projectedEligibleEntryIds: [...activeGeneration.projectedEligibleEntryIds],
      pendingMarkerIds: state.markers.map((marker) => marker.id),
    };
    state.buildingGenerationId = action.generationId;
    return state;
  }
  if (action.type === 'cutover_rebuild') {
    if (state.buildingGenerationId === null) {
      throw new Error('Prototype rebuild cutover missing building generation');
    }
    const buildingGeneration = state.generations[state.buildingGenerationId];
    if (!buildingGeneration || buildingGeneration.pendingMarkerIds.length > 0) {
      throw new Error('Prototype rebuild cutover blocked by pending markers');
    }
    const previousActiveGenerationId = state.activeGenerationId;
    state.activeGenerationId = state.buildingGenerationId;
    state.buildingGenerationId = null;
    delete state.generations[previousActiveGenerationId];
    removeCompletedMarkers(state);
    return state;
  }
  state.sourceDeletionConfirmed = true;
  for (const entry of Object.values(state.entries)) {
    entry.deleted = true;
  }
  for (const generation of Object.values(state.generations)) {
    generation.evidenceEntryIds = [];
    generation.projectedEligibleEntryIds = [];
  }
  publishPrototypeMarker(state, action.runtimeId, 'deletion');
  return state;
}

/** Summarizes the prototype state without source conversation content. */
export function summarizeIncrementalIngestionPrototypeState(
  state: IncrementalIngestionPrototypeState,
): Record<string, unknown> {
  return {
    nowSeconds: state.nowSeconds,
    entries: Object.values(state.entries),
    currentLeafId: state.currentLeafId,
    activeContextEntryIds: state.activeContextEntryIds,
    markers: state.markers,
    generations: state.generations,
    activeGenerationId: state.activeGenerationId,
    buildingGenerationId: state.buildingGenerationId,
    workerFailure: state.workerFailure,
    sourceDeletionConfirmed: state.sourceDeletionConfirmed,
  };
}
