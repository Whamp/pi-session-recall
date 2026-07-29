import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { EmbeddedEmbeddingGemmaExecutionIdentity } from './embedded-embeddinggemma-provider.js';
import {
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import type {
  RecallConversationConfig,
  RecallConversationService,
} from './recall-conversation-service.js';
import {
  createRecallModelArtifactCache,
  type RecallModelArtifactCache,
} from './recall-model-artifact-cache.js';
import {
  configureRecallInferenceCapability,
  readRecallInferenceConfiguration,
  type RecallInferenceConfiguration,
} from './recall-inference-configuration.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  type RecommendedEmbeddingGemmaModelProfile,
} from './recall-model-profiles.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { createRecommendedEmbeddingGemmaConversationRuntime } from './recommended-embeddinggemma-conversation-service.js';

const RECALL_FIRST_INDEX_SETUP_STATE_VERSION = 1;
const RECALL_FIRST_INDEX_SETUP_USAGE =
  'usage: setup:recall [status|select-embeddinggemma --approve-download|estimate [--measure --sample-sessions N]|start --approve-build|defer]';

const CORPUS_INSPECTION_SCHEMA = Type.Object(
  {
    sessionCount: Type.Integer({ minimum: 0 }),
    sourceByteSize: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

const FIRST_INDEX_SAMPLE_MEASUREMENT_SCHEMA = Type.Object(
  {
    corpus: CORPUS_INSPECTION_SCHEMA,
    sampledSessionCount: Type.Integer({ minimum: 0 }),
    sampledSourceByteSize: Type.Integer({ minimum: 0 }),
    sampledDenseDocumentCount: Type.Integer({ minimum: 0 }),
    coldStartMilliseconds: Type.Number({ minimum: 0 }),
    measuredSampleMilliseconds: Type.Number({ minimum: 0 }),
    sourceBytesPerSecond: Type.Number({ minimum: 0 }),
    denseDocumentsPerSecond: Type.Number({ minimum: 0 }),
    cacheHitCount: Type.Integer({ minimum: 0 }),
    newlyEmbeddedDocumentCount: Type.Integer({ minimum: 0 }),
    embeddingRequestCount: Type.Integer({ minimum: 0 }),
    estimatedDurationMilliseconds: Type.Object(
      {
        minimum: Type.Integer({ minimum: 0 }),
        maximum: Type.Integer({ minimum: 0 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

const RECALL_FIRST_INDEX_SETUP_STATE_SCHEMA = Type.Object(
  {
    version: Type.Literal(RECALL_FIRST_INDEX_SETUP_STATE_VERSION),
    embedding: Type.Union([
      Type.Null(),
      Type.Object(
        {
          profileId: Type.String({ minLength: 1 }),
          backend: Type.Literal('embedded'),
          adapterId: Type.Literal('node-llama-cpp-embedded-v2'),
          devicePolicy: Type.Literal('auto'),
          verifiedAt: Type.String({ format: 'date-time' }),
        },
        { additionalProperties: false },
      ),
    ]),
    lastEstimate: Type.Union([
      Type.Null(),
      Type.Object(
        {
          kind: Type.Literal('metadata'),
          measuredAt: Type.String({ format: 'date-time' }),
          corpus: CORPUS_INSPECTION_SCHEMA,
        },
        { additionalProperties: false },
      ),
      Type.Object(
        {
          kind: Type.Literal('measured'),
          measuredAt: Type.String({ format: 'date-time' }),
          measurement: FIRST_INDEX_SAMPLE_MEASUREMENT_SCHEMA,
        },
        { additionalProperties: false },
      ),
    ]),
  },
  { additionalProperties: false },
);

/** Persisted verified embedding selection and latest first-index estimate. */
export type RecallFirstIndexSetupState = ReturnType<
  typeof Value.Parse<typeof RECALL_FIRST_INDEX_SETUP_STATE_SCHEMA>
>;

/** Service operations used by deterministic guided setup and no other runtime surface. */
export type RecallFirstIndexSetupCommandService = Pick<
  RecallConversationService,
  | 'verifyEmbeddingCapability'
  | 'inspectConversationCorpus'
  | 'measureFirstIndexSample'
  | 'readIndexGenerationStatus'
  | 'startBackgroundIndexGeneration'
  | 'resumeBackgroundIndexGeneration'
>;

/** Disposable service that follows the authoritative inference configuration during setup. */
export interface RecallFirstIndexSetupConfiguredRuntime {
  service: RecallFirstIndexSetupCommandService;
  dispose(): Promise<void>;
}

/** Recommended embedded runtime used only to verify the initial EmbeddingGemma selection. */
export interface RecallFirstIndexSetupSelectedRuntime extends RecallFirstIndexSetupConfiguredRuntime {
  executionIdentity: Pick<
    EmbeddedEmbeddingGemmaExecutionIdentity,
    'adapter' | 'computeBackend' | 'deviceNames' | 'devicePolicy'
  >;
}

/** Injectable filesystem, artifact, service, clock, and output boundaries for guided setup. */
export interface RecallFirstIndexSetupCommandOptions {
  config: RecallConversationConfig;
  statePath?: string;
  inferenceConfigurationPath?: string;
  modelCacheDirectory?: string;
  profile?: RecommendedEmbeddingGemmaModelProfile;
  artifactCache?: RecallModelArtifactCache;
  metadataService?: Pick<RecallConversationService, 'inspectConversationCorpus'>;
  createSelectedServiceRuntime?: () =>
    | RecallFirstIndexSetupSelectedRuntime
    | Promise<RecallFirstIndexSetupSelectedRuntime>;
  createConfiguredServiceRuntime?: () =>
    | RecallFirstIndexSetupConfiguredRuntime
    | Promise<RecallFirstIndexSetupConfiguredRuntime>;
  nowIsoTimestamp?: () => string;
  writeOutput?: (value: string) => void;
}

type RecallFirstIndexSetupAction =
  | { action: 'status' }
  | { action: 'select-embeddinggemma'; approvedDownload: boolean }
  | { action: 'estimate'; measure: boolean; maximumSessionCount?: number }
  | { action: 'start'; approvedBuild: boolean }
  | { action: 'defer' };

function parseRecallFirstIndexSetupAction(
  argumentsList: readonly string[],
): RecallFirstIndexSetupAction {
  if (argumentsList.length === 0 || (argumentsList.length === 1 && argumentsList[0] === 'status')) {
    return { action: 'status' };
  }
  const [action, ...flags] = argumentsList;
  if (action === 'select-embeddinggemma') {
    if (flags.length === 0) {
      return { action, approvedDownload: false };
    }
    if (flags.length === 1 && flags[0] === '--approve-download') {
      return { action, approvedDownload: true };
    }
  }
  if (action === 'estimate') {
    let measure = false;
    let maximumSessionCount: number | undefined;
    for (let index = 0; index < flags.length; index += 1) {
      const flag = flags[index];
      if (flag === '--measure' && !measure) {
        measure = true;
        continue;
      }
      if (flag === '--sample-sessions' && maximumSessionCount === undefined) {
        const rawValue = flags[index + 1];
        const parsedValue = Number(rawValue);
        if (!rawValue || !Number.isInteger(parsedValue)) {
          throw new Error(
            `Recall first-index setup sample bound invalid: ${rawValue ?? 'missing'}; ${RECALL_FIRST_INDEX_SETUP_USAGE}`,
          );
        }
        maximumSessionCount = parsedValue;
        index += 1;
        continue;
      }
      throw new Error(
        `Recall first-index setup arguments invalid: ${argumentsList.join(' ')}; ${RECALL_FIRST_INDEX_SETUP_USAGE}`,
      );
    }
    if (maximumSessionCount !== undefined && !measure) {
      throw new Error(
        `Recall first-index setup --sample-sessions requires --measure; ${RECALL_FIRST_INDEX_SETUP_USAGE}`,
      );
    }
    return {
      action,
      measure,
      ...(maximumSessionCount === undefined ? {} : { maximumSessionCount }),
    };
  }
  if (action === 'start') {
    if (flags.length === 0) {
      return { action, approvedBuild: false };
    }
    if (flags.length === 1 && flags[0] === '--approve-build') {
      return { action, approvedBuild: true };
    }
  }
  if (action === 'defer' && flags.length === 0) {
    return { action };
  }
  throw new Error(
    `Recall first-index setup arguments invalid: ${argumentsList.join(' ')}; ${RECALL_FIRST_INDEX_SETUP_USAGE}`,
  );
}

function createUnconfiguredFirstIndexSetupState(): RecallFirstIndexSetupState {
  return {
    version: RECALL_FIRST_INDEX_SETUP_STATE_VERSION,
    embedding: null,
    lastEstimate: null,
  };
}

/** Resolves the guided setup state beside managed recall index data. */
export function resolveRecallFirstIndexSetupStatePath(config: RecallConversationConfig): string {
  return join(dirname(config.manifestPath), 'first-index-setup.json');
}

/** Reads guided setup state, treating a missing file as a fresh unconfigured installation. */
export async function readRecallFirstIndexSetupState(
  statePath: string,
): Promise<RecallFirstIndexSetupState> {
  try {
    const parsed: unknown = JSON.parse(await readFile(statePath, 'utf8'));
    return Value.Parse(RECALL_FIRST_INDEX_SETUP_STATE_SCHEMA, parsed);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return createUnconfiguredFirstIndexSetupState();
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall first-index setup state invalid at ${statePath}: ${message}`, {
      cause: error,
    });
  }
}

async function writeRecallFirstIndexSetupState(
  statePath: string,
  state: RecallFirstIndexSetupState,
): Promise<void> {
  const validated = Value.Parse(RECALL_FIRST_INDEX_SETUP_STATE_SCHEMA, state);
  await mkdir(dirname(statePath), { recursive: true });
  const temporaryPath = `${statePath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(validated)}\n`, 'utf8');
  await rename(temporaryPath, statePath);
}

function formatFirstIndexSetupConfiguration(
  state: RecallFirstIndexSetupState,
  configuredEmbedding: RecallInferenceConfiguration['embedding'],
) {
  const embedding = state.embedding ?? configuredEmbedding;
  return {
    state: embedding ? ('configured' as const) : ('unconfigured' as const),
    embedding,
  };
}

async function runWithConfiguredEmbeddingRuntime<T>(
  configuredEmbedding:
    | RecallInferenceConfiguration['embedding']
    | RecallFirstIndexSetupState['embedding'],
  createRuntime: () =>
    | RecallFirstIndexSetupConfiguredRuntime
    | Promise<RecallFirstIndexSetupConfiguredRuntime>,
  operation: (runtime: RecallFirstIndexSetupConfiguredRuntime) => Promise<T>,
): Promise<T> {
  if (!configuredEmbedding) {
    throw new Error(
      'Recall first-index setup embedding is unconfigured: select and verify an embedding capability first',
    );
  }
  const runtime = await createRuntime();
  try {
    return await operation(runtime);
  } finally {
    await runtime.dispose();
  }
}

/** Runs one deterministic, JSON-emitting first-index setup step with explicit consent gates. */
export async function runRecallFirstIndexSetupCommand(
  argumentsList: readonly string[],
  options: RecallFirstIndexSetupCommandOptions,
): Promise<void> {
  const parsedAction = parseRecallFirstIndexSetupAction(argumentsList);
  const profile = options.profile ?? createRecommendedEmbeddingGemmaModelProfile();
  const dataDirectory = dirname(options.config.manifestPath);
  const statePath = options.statePath ?? resolveRecallFirstIndexSetupStatePath(options.config);
  const modelCacheDirectory = options.modelCacheDirectory ?? join(dataDirectory, 'models');
  const artifactCache =
    options.artifactCache ??
    createRecallModelArtifactCache({ cacheDirectory: modelCacheDirectory, profile });
  const metadataService =
    options.metadataService ?? createRecallConversationService(options.config);
  const createSelectedServiceRuntime =
    options.createSelectedServiceRuntime ??
    (() => createRecommendedEmbeddingGemmaConversationRuntime(options.config));
  const createConfiguredServiceRuntime =
    options.createConfiguredServiceRuntime ?? createSelectedServiceRuntime;
  const nowIsoTimestamp = options.nowIsoTimestamp ?? (() => new Date().toISOString());
  const writeOutput =
    options.writeOutput ?? ((value: string) => process.stdout.write(`${value}\n`));
  const state = await readRecallFirstIndexSetupState(statePath);
  const inferenceConfigurationPath =
    options.inferenceConfigurationPath ?? join(dataDirectory, 'inference-configuration.json');
  const inferenceConfiguration = await readRecallInferenceConfiguration(
    inferenceConfigurationPath,
    {
      ...(options.config.generationRegistryPath
        ? { generationRegistryPath: options.config.generationRegistryPath }
        : {}),
    },
  );
  const configuredEmbedding = inferenceConfiguration.embedding ?? state.embedding;
  const inspection = await artifactCache.inspectArtifact();
  const recommendation = {
    profileId: profile.profileId,
    purpose: profile.purpose,
    source: profile.source,
    license: profile.license,
    exactSizeBytes: profile.source.byteSize,
    cachePath: inspection.status.artifactPath,
    devicePolicy: 'auto' as const,
    selected: configuredEmbedding?.profileId === profile.profileId,
  };

  if (parsedAction.action === 'status') {
    const recallReady = configuredEmbedding
      ? await runWithConfiguredEmbeddingRuntime(
          configuredEmbedding,
          createConfiguredServiceRuntime,
          async (runtime) => (await runtime.service.readIndexGenerationStatus()).active !== null,
        )
      : false;
    writeOutput(
      JSON.stringify({
        action: 'status',
        configuration: formatFirstIndexSetupConfiguration(state, inferenceConfiguration.embedding),
        recommendation,
        artifactStatus: inspection.status.state,
        corpusEstimate: state.lastEstimate,
        recallReady,
      }),
    );
    return;
  }

  if (parsedAction.action === 'select-embeddinggemma') {
    if (inferenceConfiguration.pendingEmbeddingReplacement) {
      throw new Error(
        'Recall first-index setup cannot replace embeddings while another embedding replacement is pending; inspect or resume the existing staging generation',
      );
    }
    if (
      inferenceConfiguration.embedding &&
      inferenceConfiguration.embedding.profileId !== profile.profileId
    ) {
      throw new Error(
        `Recall first-index setup cannot replace different embedding profile ${inferenceConfiguration.embedding.profileId}; use inference configure with explicit replacement approval`,
      );
    }
    if (!parsedAction.approvedDownload) {
      throw new Error(
        'Recall first-index setup selection requires explicit --approve-download after reviewing model purpose, source, license, exact size, cache path, and device policy',
      );
    }
    const downloadedArtifact = await artifactCache.downloadArtifact({ approved: true });
    const selection = await (async () => {
      const runtime = await createSelectedServiceRuntime();
      try {
        const verification = await runtime.service.verifyEmbeddingCapability();
        return {
          verification,
          executionIdentity: runtime.executionIdentity,
        };
      } finally {
        await runtime.dispose();
      }
    })();
    const verifiedAt = nowIsoTimestamp();
    const selectedState: RecallFirstIndexSetupState = {
      ...state,
      embedding: {
        profileId: profile.profileId,
        backend: 'embedded',
        adapterId: selection.executionIdentity.adapter,
        devicePolicy: 'auto',
        verifiedAt,
      },
    };
    await configureRecallInferenceCapability(
      inferenceConfigurationPath,
      {
        capability: RecallInferenceCapability.EMBEDDING,
        candidateId: 'recommended-embeddinggemma-embedded',
        profileId: profile.profileId,
        backend: RecallInferenceBackend.EMBEDDED,
        adapterId: selection.executionIdentity.adapter,
        endpoint: null,
        device: {
          policy: selection.executionIdentity.devicePolicy,
          computeBackend: selection.executionIdentity.computeBackend,
          names: [...selection.executionIdentity.deviceNames],
        },
        artifact: {
          path: downloadedArtifact.artifactPath,
          repository: profile.source.repository,
          revision: profile.source.revision,
          sha256: profile.source.sha256,
          byteSize: profile.source.byteSize,
        },
        async inspectHealth() {
          return { artifactState: RecallInferenceArtifactState.VALID, requiredRepair: null };
        },
        async verifyCapabilityConformance() {
          return {
            profileId: profile.profileId,
            adapterId: selection.executionIdentity.adapter,
            backend: RecallInferenceBackend.EMBEDDED,
            cacheIdentity: selection.verification.embeddingProfileId,
            embeddingProfileId: selection.verification.embeddingProfileId,
            measurement: { verificationOperations: 1 },
          };
        },
      },
      {
        ...(options.config.generationRegistryPath
          ? { generationRegistryPath: options.config.generationRegistryPath }
          : {}),
        nowIsoTimestamp: () => verifiedAt,
      },
    );
    await writeRecallFirstIndexSetupState(statePath, selectedState);
    writeOutput(
      JSON.stringify({
        action: 'select-embeddinggemma',
        configuration: formatFirstIndexSetupConfiguration(
          selectedState,
          inferenceConfiguration.embedding,
        ),
        recommendation: { ...recommendation, selected: true },
        verification: selection.verification,
        executionIdentity: selection.executionIdentity,
        recallReady: false,
      }),
    );
    return;
  }

  if (parsedAction.action === 'estimate') {
    const measuredAt = nowIsoTimestamp();
    const estimate = parsedAction.measure
      ? await runWithConfiguredEmbeddingRuntime(
          configuredEmbedding,
          createConfiguredServiceRuntime,
          async (runtime) => ({
            kind: 'measured' as const,
            measuredAt,
            measurement: await runtime.service.measureFirstIndexSample({
              ...(parsedAction.maximumSessionCount === undefined
                ? {}
                : { maximumSessionCount: parsedAction.maximumSessionCount }),
            }),
          }),
        )
      : {
          kind: 'metadata' as const,
          measuredAt,
          corpus: await metadataService.inspectConversationCorpus(),
        };
    const estimatedState: RecallFirstIndexSetupState = {
      ...state,
      lastEstimate: estimate,
    };
    await writeRecallFirstIndexSetupState(statePath, estimatedState);
    writeOutput(JSON.stringify({ action: 'estimate', estimate, recallReady: false }));
    return;
  }

  if (parsedAction.action === 'defer') {
    writeOutput(
      JSON.stringify({
        action: 'defer',
        configuration: formatFirstIndexSetupConfiguration(state, inferenceConfiguration.embedding),
        recallReady: false,
        message:
          'Recall configuration retained; recall is not ready until the first index generation activates.',
      }),
    );
    return;
  }

  if (!parsedAction.approvedBuild) {
    throw new Error(
      'Recall first-index setup start requires explicit --approve-build after reviewing the estimate',
    );
  }
  if (!state.lastEstimate) {
    throw new Error(
      'Recall first-index setup start requires a completed metadata or measured estimate before build approval',
    );
  }
  const backgroundBuild = await runWithConfiguredEmbeddingRuntime(
    configuredEmbedding,
    createConfiguredServiceRuntime,
    async (runtime) => {
      const generations = await runtime.service.readIndexGenerationStatus();
      if (generations.active) {
        throw new Error(
          `Recall first-index setup already has active generation ${generations.active.generationId}`,
        );
      }
      return generations.staging
        ? runtime.service.resumeBackgroundIndexGeneration()
        : runtime.service.startBackgroundIndexGeneration();
    },
  );
  writeOutput(JSON.stringify({ action: 'start', recallReady: false, backgroundBuild }));
}
