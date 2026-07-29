import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import { readRecallGenerationRegistry } from './recall-generation-state.js';
import { tryAcquireRecallRebuildOwnershipLock } from './recall-rebuild-ownership-lock.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const RECALL_INFERENCE_CONFIGURATION_VERSION = 2;
const RECALL_INFERENCE_CONFIGURATION_LOCK_RETRY_MILLISECONDS = 25;

const RECALL_INFERENCE_CAPABILITY_SCHEMA = Type.Enum(RecallInferenceCapability);
const RECALL_INFERENCE_BACKEND_SCHEMA = Type.Enum(RecallInferenceBackend);
const RECALL_INFERENCE_ARTIFACT_STATE_SCHEMA = Type.Enum(RecallInferenceArtifactState);
const RECALL_INFERENCE_ARTIFACT_SCHEMA = Type.Object(
  {
    path: Type.String({ minLength: 1 }),
    repository: Type.String({ minLength: 1 }),
    revision: Type.String({ minLength: 1 }),
    sha256: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    byteSize: Type.Integer({ minimum: 1 }),
    state: RECALL_INFERENCE_ARTIFACT_STATE_SCHEMA,
  },
  { additionalProperties: false },
);
const RECALL_INFERENCE_DEVICE_SCHEMA = Type.Object(
  {
    policy: Type.String({ minLength: 1 }),
    computeBackend: Type.String({ minLength: 1 }),
    names: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);
const RECALL_INFERENCE_CONFORMANCE_V1_SCHEMA = Type.Object(
  {
    verifiedAt: Type.String({ format: 'date-time' }),
    cacheIdentity: Type.String({ minLength: 1 }),
    measurement: Type.Record(Type.String({ minLength: 1 }), Type.Number({ minimum: 0 })),
  },
  { additionalProperties: false },
);
const RECALL_INFERENCE_CONFORMANCE_SCHEMA = Type.Object(
  {
    verifiedAt: Type.String({ format: 'date-time' }),
    cacheIdentity: Type.String({ minLength: 1 }),
    embeddingProfileId: Type.Union([Type.Null(), Type.String({ minLength: 1 })]),
    measurement: Type.Record(Type.String({ minLength: 1 }), Type.Number({ minimum: 0 })),
  },
  { additionalProperties: false },
);
const RECALL_CONFIGURED_INFERENCE_CAPABILITY_V1_SCHEMA = Type.Object(
  {
    capability: RECALL_INFERENCE_CAPABILITY_SCHEMA,
    candidateId: Type.String({ minLength: 1 }),
    profileId: Type.String({ minLength: 1 }),
    backend: RECALL_INFERENCE_BACKEND_SCHEMA,
    adapterId: Type.String({ minLength: 1 }),
    endpoint: Type.Union([Type.Null(), Type.String({ minLength: 1 })]),
    device: Type.Union([Type.Null(), RECALL_INFERENCE_DEVICE_SCHEMA]),
    artifact: Type.Union([Type.Null(), RECALL_INFERENCE_ARTIFACT_SCHEMA]),
    conformance: RECALL_INFERENCE_CONFORMANCE_V1_SCHEMA,
  },
  { additionalProperties: false },
);
const RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA = Type.Object(
  {
    capability: RECALL_INFERENCE_CAPABILITY_SCHEMA,
    candidateId: Type.String({ minLength: 1 }),
    profileId: Type.String({ minLength: 1 }),
    backend: RECALL_INFERENCE_BACKEND_SCHEMA,
    adapterId: Type.String({ minLength: 1 }),
    endpoint: Type.Union([Type.Null(), Type.String({ minLength: 1 })]),
    device: Type.Union([Type.Null(), RECALL_INFERENCE_DEVICE_SCHEMA]),
    artifact: Type.Union([Type.Null(), RECALL_INFERENCE_ARTIFACT_SCHEMA]),
    conformance: RECALL_INFERENCE_CONFORMANCE_SCHEMA,
  },
  { additionalProperties: false },
);
const RECALL_INFERENCE_CONFIGURATION_V1_SCHEMA = Type.Object(
  {
    version: Type.Literal(1),
    embedding: Type.Union([Type.Null(), RECALL_CONFIGURED_INFERENCE_CAPABILITY_V1_SCHEMA]),
    reranking: Type.Union([Type.Null(), RECALL_CONFIGURED_INFERENCE_CAPABILITY_V1_SCHEMA]),
    queryPlanning: Type.Union([Type.Null(), RECALL_CONFIGURED_INFERENCE_CAPABILITY_V1_SCHEMA]),
  },
  { additionalProperties: false },
);
const RECALL_PENDING_EMBEDDING_REPLACEMENT_SCHEMA = Type.Object(
  {
    embeddingProfileId: Type.String({ minLength: 1 }),
    selection: RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA,
  },
  { additionalProperties: false },
);
const RECALL_INFERENCE_CONFIGURATION_SCHEMA = Type.Object(
  {
    version: Type.Literal(RECALL_INFERENCE_CONFIGURATION_VERSION),
    embedding: Type.Union([Type.Null(), RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA]),
    reranking: Type.Union([Type.Null(), RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA]),
    queryPlanning: Type.Union([Type.Null(), RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA]),
    pendingEmbeddingReplacement: Type.Union([
      Type.Null(),
      RECALL_PENDING_EMBEDDING_REPLACEMENT_SCHEMA,
    ]),
  },
  { additionalProperties: false },
);

/** Immutable artifact identity recorded beside one selected embedded capability. */
export interface RecallInferenceArtifactIdentity {
  path: string;
  repository: string;
  revision: string;
  sha256: string;
  byteSize: number;
}

/** Last detected embedded compute backend and device names for operator diagnostics. */
export interface RecallInferenceDeviceStatus {
  policy: string;
  computeBackend: string;
  names: readonly string[];
}

/** Health result that status and doctor can obtain without substituting another adapter. */
export interface RecallInferenceCandidateHealth {
  artifactState: RecallInferenceArtifactState;
  requiredRepair: string | null;
}

/** Independent conformance evidence returned only after the relevant capability suite passes. */
export interface RecallInferenceCandidateConformance {
  profileId: string;
  adapterId: string;
  backend: RecallInferenceBackend;
  cacheIdentity: string;
  embeddingProfileId: string | null;
  measurement: Readonly<Record<string, number>>;
}

/** Public generation operations used only when an embedding profile replacement needs staging. */
export interface RecallInferenceReplacementGenerationService {
  readIndexGenerationStatus(): Promise<{
    active: unknown;
    staging: { embeddingProfileId: string } | null;
  }>;
  startBackgroundIndexGeneration(): Promise<unknown>;
  resumeBackgroundIndexGeneration(): Promise<unknown>;
}

/** One selectable adapter whose preparation and conformance boundaries remain injectable. */
export interface RecallInferenceConfigurationCandidate {
  capability: RecallInferenceCapability;
  candidateId: string;
  profileId: string;
  backend: RecallInferenceBackend;
  adapterId: string;
  endpoint: string | null;
  device: RecallInferenceDeviceStatus | null;
  artifact: RecallInferenceArtifactIdentity | null;
  inspectHealth(this: void): Promise<RecallInferenceCandidateHealth>;
  prepareArtifact?(this: void, approved: boolean): Promise<void>;
  repairArtifact?(this: void, approved: boolean): Promise<void>;
  verifyCapabilityConformance(this: void): Promise<RecallInferenceCandidateConformance>;
  generationService?: RecallInferenceReplacementGenerationService;
}

/** Persisted mixed inference selections; optional capabilities remain null until selected. */
export type RecallInferenceConfiguration = ReturnType<
  typeof Value.Parse<typeof RECALL_INFERENCE_CONFIGURATION_SCHEMA>
>;

/** Clock and explicit artifact consent used while changing one capability. */
export interface ConfigureRecallInferenceCapabilityOptions {
  approvedArtifactChange?: boolean;
  approvedEmbeddingReplacement?: boolean;
  generationRegistryPath?: string;
  nowIsoTimestamp?: () => string;
}

/** One status/doctor row explaining selection, health, conformance, and required repair. */
export interface RecallInferenceCapabilityStatus {
  capability: RecallInferenceCapability;
  required: boolean;
  configured: boolean;
  selection: RecallInferenceConfiguration['embedding'];
  health: RecallInferenceCandidateHealth | null;
  conformanceStatus: 'passed' | 'failed' | 'not-configured' | 'adapter-unavailable';
  requiredRepair: string | null;
}

/** Complete mixed-capability readiness and per-capability operator evidence. */
export interface RecallInferenceConfigurationStatus {
  ready: boolean;
  capabilities: readonly RecallInferenceCapabilityStatus[];
}

/** Optional active-generation selector used to resolve an activated pending embedding replacement. */
export interface ReadRecallInferenceConfigurationOptions {
  generationRegistryPath?: string;
}

/** Controls whether doctor reruns model semantics instead of reading stored conformance. */
export interface InspectRecallInferenceConfigurationOptions {
  verifyConformance: boolean;
  generationRegistryPath?: string;
}

async function acquireRecallInferenceConfigurationLock(
  statePath: string,
): Promise<() => Promise<void>> {
  const lockPath = `${statePath}.configuration-lock`;
  while (true) {
    const ownership = await tryAcquireRecallRebuildOwnershipLock(lockPath);
    if (ownership) {
      return () => ownership.release();
    }
    await sleep(RECALL_INFERENCE_CONFIGURATION_LOCK_RETRY_MILLISECONDS);
  }
}

async function withRecallInferenceConfigurationLock<T>(
  statePath: string,
  operation: () => Promise<T>,
): Promise<T> {
  const releaseLock = await acquireRecallInferenceConfigurationLock(statePath);
  try {
    return await operation();
  } finally {
    await releaseLock();
  }
}

function createEmptyRecallInferenceConfiguration(): RecallInferenceConfiguration {
  return {
    version: RECALL_INFERENCE_CONFIGURATION_VERSION,
    embedding: null,
    reranking: null,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  };
}

function readConfiguredCapability(
  configuration: RecallInferenceConfiguration,
  capability: RecallInferenceCapability,
): RecallInferenceConfiguration['embedding'] {
  if (capability === RecallInferenceCapability.EMBEDDING) {
    return configuration.embedding;
  }
  if (capability === RecallInferenceCapability.RERANKING) {
    return configuration.reranking;
  }
  return configuration.queryPlanning;
}

function replaceConfiguredCapability(
  configuration: RecallInferenceConfiguration,
  capability: RecallInferenceCapability,
  selection: RecallInferenceConfiguration['embedding'],
): RecallInferenceConfiguration {
  if (capability === RecallInferenceCapability.EMBEDDING) {
    return { ...configuration, embedding: selection };
  }
  if (capability === RecallInferenceCapability.RERANKING) {
    return { ...configuration, reranking: selection };
  }
  return { ...configuration, queryPlanning: selection };
}

function findSelectedCandidate(
  selection: NonNullable<RecallInferenceConfiguration['embedding']>,
  candidates: readonly RecallInferenceConfigurationCandidate[],
): RecallInferenceConfigurationCandidate | undefined {
  return candidates.find(
    (candidate) =>
      candidate.capability === selection.capability &&
      candidate.candidateId === selection.candidateId &&
      candidate.profileId === selection.profileId &&
      candidate.backend === selection.backend &&
      candidate.adapterId === selection.adapterId,
  );
}

function assertCandidateHealthAcceptable(
  candidate: RecallInferenceConfigurationCandidate,
  health: RecallInferenceCandidateHealth,
): void {
  const expectedArtifactState = candidate.artifact
    ? RecallInferenceArtifactState.VALID
    : RecallInferenceArtifactState.NOT_REQUIRED;
  if (health.artifactState !== expectedArtifactState || health.requiredRepair) {
    throw new Error(
      `Recall ${candidate.capability} configuration requires repair for ${candidate.candidateId}: ${health.requiredRepair ?? `expected artifact state ${expectedArtifactState}, received ${health.artifactState}`}`,
    );
  }
}

function assertCandidateConformanceMatchesSelection(
  candidate: RecallInferenceConfigurationCandidate,
  conformance: RecallInferenceCandidateConformance,
): void {
  if (
    conformance.profileId !== candidate.profileId ||
    conformance.adapterId !== candidate.adapterId ||
    conformance.backend !== candidate.backend
  ) {
    throw new Error(
      `Recall ${candidate.capability} conformance identity mismatch for ${candidate.candidateId}: expected ${candidate.profileId}/${candidate.backend}/${candidate.adapterId}, received ${conformance.profileId}/${conformance.backend}/${conformance.adapterId}`,
    );
  }
  if (!conformance.cacheIdentity.trim()) {
    throw new Error(
      `Recall ${candidate.capability} conformance cache identity invalid for ${candidate.candidateId}`,
    );
  }
  if (
    candidate.capability === RecallInferenceCapability.EMBEDDING &&
    !conformance.embeddingProfileId?.trim()
  ) {
    throw new Error(
      `Recall embedding conformance generation identity invalid for ${candidate.candidateId}`,
    );
  }
  if (
    candidate.capability !== RecallInferenceCapability.EMBEDDING &&
    conformance.embeddingProfileId !== null
  ) {
    throw new Error(
      `Recall ${candidate.capability} conformance unexpectedly returned embedding generation identity for ${candidate.candidateId}`,
    );
  }
}

function migrateRecallInferenceConfigurationV1(
  configuration: ReturnType<typeof Value.Parse<typeof RECALL_INFERENCE_CONFIGURATION_V1_SCHEMA>>,
): RecallInferenceConfiguration {
  function migrateSelection(
    selection: typeof configuration.embedding,
  ): RecallInferenceConfiguration['embedding'] {
    if (!selection) {
      return null;
    }
    return Value.Parse(RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA, {
      ...selection,
      conformance: {
        ...selection.conformance,
        embeddingProfileId:
          selection.capability === RecallInferenceCapability.EMBEDDING
            ? selection.conformance.cacheIdentity
            : null,
      },
    });
  }
  return {
    version: RECALL_INFERENCE_CONFIGURATION_VERSION,
    embedding: migrateSelection(configuration.embedding),
    reranking: migrateSelection(configuration.reranking),
    queryPlanning: migrateSelection(configuration.queryPlanning),
    pendingEmbeddingReplacement: null,
  };
}

function parseRecallInferenceConfiguration(parsed: unknown): RecallInferenceConfiguration {
  if (Value.Check(RECALL_INFERENCE_CONFIGURATION_SCHEMA, parsed)) {
    return Value.Parse(RECALL_INFERENCE_CONFIGURATION_SCHEMA, parsed);
  }
  if (Value.Check(RECALL_INFERENCE_CONFIGURATION_V1_SCHEMA, parsed)) {
    return migrateRecallInferenceConfigurationV1(
      Value.Parse(RECALL_INFERENCE_CONFIGURATION_V1_SCHEMA, parsed),
    );
  }
  return Value.Parse(RECALL_INFERENCE_CONFIGURATION_SCHEMA, parsed);
}

class RecallGenerationRegistrySelectionError extends Error {}

async function readActiveRecallGenerationEmbeddingProfileId(
  generationRegistryPath: string,
): Promise<string | null> {
  try {
    const registry = await readRecallGenerationRegistry(generationRegistryPath);
    if (!registry?.activeGenerationId) {
      return null;
    }
    return (
      registry.generations.find(({ generationId }) => generationId === registry.activeGenerationId)
        ?.embeddingProfileId ?? null
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new RecallGenerationRegistrySelectionError(
      `Recall generation registry selection invalid at ${generationRegistryPath}: ${message}`,
      { cause: error },
    );
  }
}

async function resolveActivatedPendingEmbeddingReplacement(
  configuration: RecallInferenceConfiguration,
  generationRegistryPath: string,
): Promise<RecallInferenceConfiguration> {
  const pending = configuration.pendingEmbeddingReplacement;
  if (!pending) {
    return configuration;
  }
  if (
    pending.selection.capability !== RecallInferenceCapability.EMBEDDING ||
    pending.selection.conformance.embeddingProfileId !== pending.embeddingProfileId
  ) {
    throw new Error('Recall pending embedding replacement identity is internally inconsistent');
  }
  const activeEmbeddingProfileId =
    await readActiveRecallGenerationEmbeddingProfileId(generationRegistryPath);
  if (activeEmbeddingProfileId !== pending.embeddingProfileId) {
    return configuration;
  }
  return {
    ...configuration,
    embedding: pending.selection,
    pendingEmbeddingReplacement: null,
  };
}

/** Reads inference configuration and resolves a pending replacement only after generation activation. */
export async function readRecallInferenceConfiguration(
  statePath: string,
  options: ReadRecallInferenceConfigurationOptions = {},
): Promise<RecallInferenceConfiguration> {
  try {
    const parsed: unknown = JSON.parse(await readFile(statePath, 'utf8'));
    const configuration = parseRecallInferenceConfiguration(parsed);
    return resolveActivatedPendingEmbeddingReplacement(
      configuration,
      options.generationRegistryPath ?? join(dirname(statePath), 'generation-registry.json'),
    );
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return createEmptyRecallInferenceConfiguration();
    }
    if (error instanceof RecallGenerationRegistrySelectionError) {
      throw error;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall inference configuration invalid at ${statePath}: ${message}`, {
      cause: error,
    });
  }
}

/** Atomically replaces one validated inference configuration file. */
export async function writeRecallInferenceConfiguration(
  statePath: string,
  configuration: RecallInferenceConfiguration,
): Promise<void> {
  const validated = Value.Parse(RECALL_INFERENCE_CONFIGURATION_SCHEMA, configuration);
  await mkdir(dirname(statePath), { recursive: true });
  const temporaryPath = `${statePath}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(validated)}\n`, 'utf8');
  await rename(temporaryPath, statePath);
}

/** Verifies and atomically selects one capability without changing sibling capabilities. */
export async function configureRecallInferenceCapability(
  statePath: string,
  candidate: RecallInferenceConfigurationCandidate,
  options: ConfigureRecallInferenceCapabilityOptions = {},
): Promise<RecallInferenceConfiguration> {
  let health = await candidate.inspectHealth();
  const expectedArtifactState = candidate.artifact
    ? RecallInferenceArtifactState.VALID
    : RecallInferenceArtifactState.NOT_REQUIRED;
  if (health.artifactState !== expectedArtifactState || health.requiredRepair) {
    const prepareArtifact = candidate.prepareArtifact;
    if (!prepareArtifact || !options.approvedArtifactChange) {
      throw new Error(
        `Recall ${candidate.capability} configuration requires explicit artifact approval for ${candidate.candidateId}: ${health.requiredRepair ?? health.artifactState}`,
      );
    }
    await prepareArtifact(true);
    health = await candidate.inspectHealth();
  }
  assertCandidateHealthAcceptable(candidate, health);
  const conformance = await candidate.verifyCapabilityConformance();
  assertCandidateConformanceMatchesSelection(candidate, conformance);
  const selectedAt = options.nowIsoTimestamp?.() ?? new Date().toISOString();
  const selection = Value.Parse(RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA, {
    capability: candidate.capability,
    candidateId: candidate.candidateId,
    profileId: candidate.profileId,
    backend: candidate.backend,
    adapterId: candidate.adapterId,
    endpoint: candidate.endpoint,
    device: candidate.device
      ? {
          policy: candidate.device.policy,
          computeBackend: candidate.device.computeBackend,
          names: [...candidate.device.names],
        }
      : null,
    artifact: candidate.artifact ? { ...candidate.artifact, state: health.artifactState } : null,
    conformance: {
      verifiedAt: selectedAt,
      cacheIdentity: conformance.cacheIdentity,
      embeddingProfileId: conformance.embeddingProfileId,
      measurement: { ...conformance.measurement },
    },
  });
  return withRecallInferenceConfigurationLock(statePath, async () => {
    const current = await readRecallInferenceConfiguration(statePath, {
      ...(options.generationRegistryPath
        ? { generationRegistryPath: options.generationRegistryPath }
        : {}),
    });
    const previousSelection = readConfiguredCapability(current, candidate.capability);
    if (
      candidate.capability === RecallInferenceCapability.EMBEDDING &&
      previousSelection &&
      previousSelection.profileId !== candidate.profileId
    ) {
      if (!options.approvedEmbeddingReplacement) {
        throw new Error(
          `Recall embedding profile change from ${previousSelection.profileId} to ${candidate.profileId} requires explicit embedding replacement approval`,
        );
      }
      const generationService = candidate.generationService;
      if (!generationService) {
        throw new Error(
          `Recall embedding profile change to ${candidate.profileId} cannot start staging: replacement generation service is unavailable`,
        );
      }
      const embeddingProfileId = conformance.embeddingProfileId;
      if (!embeddingProfileId) {
        throw new Error(
          `Recall embedding profile change to ${candidate.profileId} lacks a verified generation identity`,
        );
      }
      const generations = await generationService.readIndexGenerationStatus();
      if (generations.staging && generations.staging.embeddingProfileId !== embeddingProfileId) {
        throw new Error(
          `Recall embedding profile change cannot reuse staging profile ${generations.staging.embeddingProfileId}; explicitly discard it before selecting ${candidate.profileId}`,
        );
      }
      const pendingConfiguration: RecallInferenceConfiguration = {
        ...current,
        pendingEmbeddingReplacement: {
          embeddingProfileId,
          selection,
        },
      };
      await writeRecallInferenceConfiguration(statePath, pendingConfiguration);
      try {
        if (generations.staging) {
          await generationService.resumeBackgroundIndexGeneration();
        } else {
          await generationService.startBackgroundIndexGeneration();
        }
      } catch (error) {
        await writeRecallInferenceConfiguration(statePath, current);
        throw error;
      }
      return pendingConfiguration;
    }
    const updated = replaceConfiguredCapability(current, candidate.capability, selection);
    await writeRecallInferenceConfiguration(statePath, updated);
    return updated;
  });
}

/** Consent and clock used to repair exactly one already selected capability. */
export interface RepairRecallInferenceCapabilityOptions {
  approvedArtifactRepair?: boolean;
  generationRegistryPath?: string;
  nowIsoTimestamp?: () => string;
}

/** Repairs and reverifies one exact selection without resetting valid sibling capabilities. */
export async function repairRecallInferenceCapability(
  statePath: string,
  capability: RecallInferenceCapability,
  candidate: RecallInferenceConfigurationCandidate,
  options: RepairRecallInferenceCapabilityOptions = {},
): Promise<RecallInferenceConfiguration> {
  const current = await readRecallInferenceConfiguration(statePath, {
    ...(options.generationRegistryPath
      ? { generationRegistryPath: options.generationRegistryPath }
      : {}),
  });
  const selection = readConfiguredCapability(current, capability);
  if (!selection) {
    throw new Error(`Recall ${capability} repair unavailable: capability is not configured`);
  }
  if (!findSelectedCandidate(selection, [candidate])) {
    throw new Error(
      `Recall ${capability} repair candidate mismatch: expected exact adapter ${selection.candidateId}; no substitute was selected`,
    );
  }
  let health = await candidate.inspectHealth();
  const expectedArtifactState = candidate.artifact
    ? RecallInferenceArtifactState.VALID
    : RecallInferenceArtifactState.NOT_REQUIRED;
  if (health.artifactState !== expectedArtifactState || health.requiredRepair) {
    const repairArtifact = candidate.repairArtifact;
    if (!repairArtifact || !options.approvedArtifactRepair) {
      throw new Error(
        `Recall ${capability} repair requires explicit artifact repair approval for ${candidate.candidateId}: ${health.requiredRepair ?? health.artifactState}`,
      );
    }
    await repairArtifact(true);
    health = await candidate.inspectHealth();
  }
  assertCandidateHealthAcceptable(candidate, health);
  const conformance = await candidate.verifyCapabilityConformance();
  assertCandidateConformanceMatchesSelection(candidate, conformance);
  const repairedSelection = Value.Parse(RECALL_CONFIGURED_INFERENCE_CAPABILITY_SCHEMA, {
    ...selection,
    device: candidate.device
      ? {
          policy: candidate.device.policy,
          computeBackend: candidate.device.computeBackend,
          names: [...candidate.device.names],
        }
      : null,
    artifact: candidate.artifact ? { ...candidate.artifact, state: health.artifactState } : null,
    conformance: {
      verifiedAt: options.nowIsoTimestamp?.() ?? new Date().toISOString(),
      cacheIdentity: conformance.cacheIdentity,
      embeddingProfileId: conformance.embeddingProfileId,
      measurement: { ...conformance.measurement },
    },
  });
  return withRecallInferenceConfigurationLock(statePath, async () => {
    const latest = await readRecallInferenceConfiguration(statePath, {
      ...(options.generationRegistryPath
        ? { generationRegistryPath: options.generationRegistryPath }
        : {}),
    });
    const latestSelection = readConfiguredCapability(latest, capability);
    if (!latestSelection || !findSelectedCandidate(latestSelection, [candidate])) {
      throw new Error(
        `Recall ${capability} repair selection changed while conformance was running; retry the exact selected adapter`,
      );
    }
    const updated = replaceConfiguredCapability(latest, capability, {
      ...repairedSelection,
      candidateId: latestSelection.candidateId,
      profileId: latestSelection.profileId,
      backend: latestSelection.backend,
      adapterId: latestSelection.adapterId,
      endpoint: latestSelection.endpoint,
    });
    await writeRecallInferenceConfiguration(statePath, updated);
    return updated;
  });
}

/** Removes an optional capability while preserving embeddings and every sibling selection. */
export async function removeRecallInferenceCapability(
  statePath: string,
  capability: RecallInferenceCapability,
): Promise<RecallInferenceConfiguration> {
  if (capability === RecallInferenceCapability.EMBEDDING) {
    throw new Error(
      'Recall inference configuration cannot remove the required embedding capability',
    );
  }
  return withRecallInferenceConfigurationLock(statePath, async () => {
    const current = await readRecallInferenceConfiguration(statePath);
    const updated = replaceConfiguredCapability(current, capability, null);
    await writeRecallInferenceConfiguration(statePath, updated);
    return updated;
  });
}

/**
 * Clears a pending embedding profile replacement after its staging generation is discarded.
 * Returns true when a pending replacement was removed.
 */
export async function clearPendingRecallEmbeddingReplacement(
  statePath: string,
  options: {
    generationRegistryPath?: string;
  } = {},
): Promise<boolean> {
  return withRecallInferenceConfigurationLock(statePath, async () => {
    const current = await readRecallInferenceConfiguration(statePath, {
      ...(options.generationRegistryPath
        ? { generationRegistryPath: options.generationRegistryPath }
        : {}),
    });
    const pending = current.pendingEmbeddingReplacement;
    if (!pending) {
      return false;
    }
    await writeRecallInferenceConfiguration(statePath, {
      ...current,
      pendingEmbeddingReplacement: null,
    });
    return true;
  });
}

/** Inspects all configured capabilities and optionally reruns their exact conformance probes. */
export async function inspectRecallInferenceConfiguration(
  statePath: string,
  candidates: readonly RecallInferenceConfigurationCandidate[],
  options: InspectRecallInferenceConfigurationOptions,
): Promise<RecallInferenceConfigurationStatus> {
  const configuration = await readRecallInferenceConfiguration(statePath, {
    ...(options.generationRegistryPath
      ? { generationRegistryPath: options.generationRegistryPath }
      : {}),
  });
  const capabilities: RecallInferenceCapability[] = [
    RecallInferenceCapability.EMBEDDING,
    RecallInferenceCapability.RERANKING,
    RecallInferenceCapability.QUERY_PLANNING,
  ];
  const statuses: RecallInferenceCapabilityStatus[] = [];
  for (const capability of capabilities) {
    const required = capability === RecallInferenceCapability.EMBEDDING;
    const selection = readConfiguredCapability(configuration, capability);
    if (!selection) {
      statuses.push({
        capability,
        required,
        configured: false,
        selection: null,
        health: null,
        conformanceStatus: 'not-configured',
        requiredRepair: required ? 'select and verify an embedding capability' : null,
      });
      continue;
    }
    const candidate = findSelectedCandidate(selection, candidates);
    if (!candidate) {
      statuses.push({
        capability,
        required,
        configured: true,
        selection,
        health: null,
        conformanceStatus: 'adapter-unavailable',
        requiredRepair: `restore exact adapter ${selection.candidateId}; no substitute was selected`,
      });
      continue;
    }
    let health: RecallInferenceCandidateHealth;
    try {
      health = await candidate.inspectHealth();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      statuses.push({
        capability,
        required,
        configured: true,
        selection,
        health: null,
        conformanceStatus: 'failed',
        requiredRepair: `inspect ${candidate.candidateId}: ${message}`,
      });
      continue;
    }
    let conformanceStatus: RecallInferenceCapabilityStatus['conformanceStatus'] = 'passed';
    let requiredRepair = health.requiredRepair;
    if (options.verifyConformance && !requiredRepair) {
      try {
        const conformance = await candidate.verifyCapabilityConformance();
        assertCandidateConformanceMatchesSelection(candidate, conformance);
      } catch (error) {
        conformanceStatus = 'failed';
        const message = error instanceof Error ? error.message : String(error);
        requiredRepair = `rerun ${candidate.capability} conformance for ${candidate.candidateId}: ${message}`;
      }
    }
    const expectedArtifactState = candidate.artifact
      ? RecallInferenceArtifactState.VALID
      : RecallInferenceArtifactState.NOT_REQUIRED;
    if (health.artifactState !== expectedArtifactState && !requiredRepair) {
      requiredRepair = `restore ${candidate.candidateId} artifact: expected ${expectedArtifactState}, received ${health.artifactState}`;
    }
    statuses.push({
      capability,
      required,
      configured: true,
      selection,
      health,
      conformanceStatus,
      requiredRepair,
    });
  }
  return {
    ready: statuses.every((status) =>
      status.configured
        ? status.conformanceStatus === 'passed' && status.requiredRepair === null
        : !status.required,
    ),
    capabilities: statuses,
  };
}
