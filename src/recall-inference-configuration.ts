import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { readNodeErrorCode } from './read-node-error-code.js';

const RECALL_INFERENCE_CONFIGURATION_VERSION = 1;

const recallInferenceCapabilitySchema = Type.Union([
  Type.Literal('embedding'),
  Type.Literal('reranking'),
  Type.Literal('query-planning'),
]);
const recallInferenceBackendSchema = Type.Union([
  Type.Literal('embedded'),
  Type.Literal('llama-cpp-http'),
  Type.Literal('custom'),
]);
const recallInferenceArtifactStateSchema = Type.Union([
  Type.Literal('valid'),
  Type.Literal('missing'),
  Type.Literal('corrupt'),
  Type.Literal('partial'),
  Type.Literal('incompatible'),
  Type.Literal('not-required'),
]);
const recallInferenceArtifactSchema = Type.Object(
  {
    path: Type.String({ minLength: 1 }),
    repository: Type.String({ minLength: 1 }),
    revision: Type.String({ minLength: 1 }),
    sha256: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    byteSize: Type.Integer({ minimum: 1 }),
    state: recallInferenceArtifactStateSchema,
  },
  { additionalProperties: false },
);
const recallInferenceDeviceSchema = Type.Object(
  {
    policy: Type.String({ minLength: 1 }),
    computeBackend: Type.String({ minLength: 1 }),
    names: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);
const recallInferenceConformanceSchema = Type.Object(
  {
    verifiedAt: Type.String({ format: 'date-time' }),
    cacheIdentity: Type.String({ minLength: 1 }),
    measurement: Type.Record(Type.String({ minLength: 1 }), Type.Number({ minimum: 0 })),
  },
  { additionalProperties: false },
);
const recallConfiguredInferenceCapabilitySchema = Type.Object(
  {
    capability: recallInferenceCapabilitySchema,
    candidateId: Type.String({ minLength: 1 }),
    profileId: Type.String({ minLength: 1 }),
    backend: recallInferenceBackendSchema,
    adapterId: Type.String({ minLength: 1 }),
    endpoint: Type.Union([Type.Null(), Type.String({ minLength: 1 })]),
    device: Type.Union([Type.Null(), recallInferenceDeviceSchema]),
    artifact: Type.Union([Type.Null(), recallInferenceArtifactSchema]),
    conformance: recallInferenceConformanceSchema,
  },
  { additionalProperties: false },
);
const recallInferenceConfigurationSchema = Type.Object(
  {
    version: Type.Literal(RECALL_INFERENCE_CONFIGURATION_VERSION),
    embedding: Type.Union([Type.Null(), recallConfiguredInferenceCapabilitySchema]),
    reranking: Type.Union([Type.Null(), recallConfiguredInferenceCapabilitySchema]),
    queryPlanning: Type.Union([Type.Null(), recallConfiguredInferenceCapabilitySchema]),
  },
  { additionalProperties: false },
);

/** Inference capability configured and verified independently by guided setup. */
export type RecallInferenceCapability = 'embedding' | 'reranking' | 'query-planning';

/** Supported execution location without changing model-profile semantics. */
export type RecallInferenceBackend = 'embedded' | 'llama-cpp-http' | 'custom';

/** Current local artifact health; HTTP and artifact-free custom adapters use not-required. */
export type RecallInferenceArtifactState =
  | 'valid'
  | 'missing'
  | 'corrupt'
  | 'partial'
  | 'incompatible'
  | 'not-required';

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
  typeof Value.Parse<typeof recallInferenceConfigurationSchema>
>;

/** Clock and explicit artifact consent used while changing one capability. */
export interface ConfigureRecallInferenceCapabilityOptions {
  approvedArtifactChange?: boolean;
  approvedEmbeddingReplacement?: boolean;
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

/** Controls whether doctor reruns model semantics instead of reading stored conformance. */
export interface InspectRecallInferenceConfigurationOptions {
  verifyConformance: boolean;
}

function createEmptyRecallInferenceConfiguration(): RecallInferenceConfiguration {
  return {
    version: RECALL_INFERENCE_CONFIGURATION_VERSION,
    embedding: null,
    reranking: null,
    queryPlanning: null,
  };
}

function readConfiguredCapability(
  configuration: RecallInferenceConfiguration,
  capability: RecallInferenceCapability,
): RecallInferenceConfiguration['embedding'] {
  if (capability === 'embedding') {
    return configuration.embedding;
  }
  if (capability === 'reranking') {
    return configuration.reranking;
  }
  return configuration.queryPlanning;
}

function replaceConfiguredCapability(
  configuration: RecallInferenceConfiguration,
  capability: RecallInferenceCapability,
  selection: RecallInferenceConfiguration['embedding'],
): RecallInferenceConfiguration {
  if (capability === 'embedding') {
    return { ...configuration, embedding: selection };
  }
  if (capability === 'reranking') {
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
  const expectedArtifactState = candidate.artifact ? 'valid' : 'not-required';
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
}

/** Reads atomic inference configuration; a missing file is an unconfigured installation. */
export async function readRecallInferenceConfiguration(
  statePath: string,
): Promise<RecallInferenceConfiguration> {
  try {
    const parsed: unknown = JSON.parse(await readFile(statePath, 'utf8'));
    return Value.Parse(recallInferenceConfigurationSchema, parsed);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return createEmptyRecallInferenceConfiguration();
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
  const validated = Value.Parse(recallInferenceConfigurationSchema, configuration);
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
  const expectedArtifactState = candidate.artifact ? 'valid' : 'not-required';
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
  const current = await readRecallInferenceConfiguration(statePath);
  const previousSelection = readConfiguredCapability(current, candidate.capability);
  if (
    candidate.capability === 'embedding' &&
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
    const generations = await generationService.readIndexGenerationStatus();
    if (
      generations.staging &&
      generations.staging.embeddingProfileId !== conformance.cacheIdentity
    ) {
      throw new Error(
        `Recall embedding profile change cannot reuse staging profile ${generations.staging.embeddingProfileId}; explicitly discard it before selecting ${candidate.profileId}`,
      );
    }
    if (generations.staging) {
      await generationService.resumeBackgroundIndexGeneration();
    } else {
      await generationService.startBackgroundIndexGeneration();
    }
  }
  const selectedAt = options.nowIsoTimestamp?.() ?? new Date().toISOString();
  const selection = Value.Parse(recallConfiguredInferenceCapabilitySchema, {
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
      measurement: { ...conformance.measurement },
    },
  });
  const updated = replaceConfiguredCapability(current, candidate.capability, selection);
  await writeRecallInferenceConfiguration(statePath, updated);
  return updated;
}

/** Consent and clock used to repair exactly one already selected capability. */
export interface RepairRecallInferenceCapabilityOptions {
  approvedArtifactRepair?: boolean;
  nowIsoTimestamp?: () => string;
}

/** Repairs and reverifies one exact selection without resetting valid sibling capabilities. */
export async function repairRecallInferenceCapability(
  statePath: string,
  capability: RecallInferenceCapability,
  candidate: RecallInferenceConfigurationCandidate,
  options: RepairRecallInferenceCapabilityOptions = {},
): Promise<RecallInferenceConfiguration> {
  const current = await readRecallInferenceConfiguration(statePath);
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
  const expectedArtifactState = candidate.artifact ? 'valid' : 'not-required';
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
  const repairedSelection = Value.Parse(recallConfiguredInferenceCapabilitySchema, {
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
      measurement: { ...conformance.measurement },
    },
  });
  const updated = replaceConfiguredCapability(current, capability, repairedSelection);
  await writeRecallInferenceConfiguration(statePath, updated);
  return updated;
}

/** Removes an optional capability while preserving embeddings and every sibling selection. */
export async function removeRecallInferenceCapability(
  statePath: string,
  capability: RecallInferenceCapability,
): Promise<RecallInferenceConfiguration> {
  if (capability === 'embedding') {
    throw new Error(
      'Recall inference configuration cannot remove the required embedding capability',
    );
  }
  const current = await readRecallInferenceConfiguration(statePath);
  const updated = replaceConfiguredCapability(current, capability, null);
  await writeRecallInferenceConfiguration(statePath, updated);
  return updated;
}

/** Inspects all configured capabilities and optionally reruns their exact conformance probes. */
export async function inspectRecallInferenceConfiguration(
  statePath: string,
  candidates: readonly RecallInferenceConfigurationCandidate[],
  options: InspectRecallInferenceConfigurationOptions,
): Promise<RecallInferenceConfigurationStatus> {
  const configuration = await readRecallInferenceConfiguration(statePath);
  const capabilities: RecallInferenceCapability[] = ['embedding', 'reranking', 'query-planning'];
  const statuses: RecallInferenceCapabilityStatus[] = [];
  for (const capability of capabilities) {
    const required = capability === 'embedding';
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
    const expectedArtifactState = candidate.artifact ? 'valid' : 'not-required';
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
