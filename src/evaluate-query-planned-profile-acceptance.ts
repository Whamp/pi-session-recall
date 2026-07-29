import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { createQmdHttpQueryPlanningProvider } from './createQmdHttpQueryPlanningProvider.js';
import { createQwenHttpRerankingProvider } from './createQwenHttpRerankingProvider.js';
import { createEmbeddedQmdQueryPlanningProvider } from './embedded-qmd-query-planning-provider.js';
import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import {
  EmbeddedInferenceDevicePolicy,
  QueryPlannedRecallControlKind,
  RecallInferenceBackend,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  runCheckpointedLiveProfileEvaluationMatrix,
  type CheckpointedLiveProfileEvaluation,
  type LiveProfileEvaluationCheckpointIdentity,
} from './live-query-planned-profile-checkpoints.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  removeStaleRecallEvaluationTemporaryFiles,
  writeAtomicRecallEvaluationFile,
} from './recall-evaluation-file-system.js';
import {
  assertRecallEvaluationGitRevisionCurrent,
  readCleanRecallEvaluationGitRevision,
} from './recall-evaluation-git-revision.js';
import { verifyRequiredRecallEvaluationSemanticChecks } from './recall-evaluation-semantic-checks.js';
import {
  loadPrivateQueryPlannedRecallCorpus,
  type LoadedPrivateQueryPlannedRecallCorpus,
} from './query-planned-recall-baseline.js';
import {
  createLiveQueryPlannedEvaluationConfigurationIdentity,
  createLiveQueryPlannedEvaluationCorpusIdentity,
  createLiveQueryPlannedProfileIdentity,
  createPublishableLiveQueryPlannedProfileAcceptance,
  formatPublishableLiveQueryPlannedProfileAcceptanceReport,
  loadPrivateQueryPlannedRecallPlans,
  runLiveQueryPlannedProfileEvaluation,
  type CommittedCorpusLiveProfileEvidence,
  type LiveQueryPlannedProfileEvaluationResult,
  type LiveQueryPlannedProfileIdentity,
  type LiveQueryPlannedProfileRunIdentity,
  type LiveQueryPlannedSoftwareIdentity,
  type PublishableLiveQueryPlannedProfileAcceptance,
} from './query-planned-recall-evaluation.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

const REQUIRE = createRequire(import.meta.url);

const LIVE_PROFILE_ACCEPTANCE_HELP = `Usage: npm run evaluate:query-planned-profiles -- \\
  --accelerated-device <metal|cuda|vulkan> \\
  --http-planner-url <llama.cpp-v1-base-url> \\
  --http-reranker-url <llama.cpp-v1-base-url> \\
  --http-device-class <cpu|accelerated> \\
  --http-device <publishable-device-label> \\
  --http-backend-version <publishable-server-revision>

Runs embedded CPU, embedded accelerator, and HTTP planner/reranker profiles through shared
capability conformance and public RecallConversationService searches over the approved private
corpus. It binds those runs to existing committed-corpus files:
  docs/evaluation/embeddinggemma-quality-cpu.json
  docs/evaluation/embeddinggemma-quality-<accelerated-device>.json

Publishes aggregate evidence to:
  docs/evaluation/query-planned-profile-acceptance.json
  docs/evaluation/query-planned-profile-acceptance.md

Completed profiles are checkpointed under the private .recall-data evaluation area. A rerun resumes
only checkpoints with the exact commit, corpus, profile, backend, device, and adapter configuration.
Recognized interrupted-writer temps for the two exact outputs are recovered before the clean-worktree
gate. Unrelated untracked work still fails. HEAD and the fully clean worktree, including existing output
files, are revalidated immediately before publication. The command reports profile and case progress
to stderr.

The command never downloads models, never scans production sessions, and never publishes private query, plan, or source text. Start exact model-bound llama.cpp HTTP planner and reranker servers before running it.
`;

const RERANKER_CONFORMANCE_QUERY = 'source provenance';
const RERANKER_CONFORMANCE_DOCUMENTS = [
  'Source provenance records connect recalled evidence to its original Pi session.',
  'The navigation bar is blue.',
] as const;
const LLAMA_CPP_B8390_CPU_REFERENCE_SCORES = [
  0.998_413_801_193_237_3, 0.000_137_552_677_188_068_63,
] as const;
const LIVE_RERANKER_MAXIMUM_ABSOLUTE_DIFFERENCE = 0.001;

interface LiveProfileAcceptanceCliOptions {
  acceleratedDevice:
    | EmbeddedInferenceDevicePolicy.METAL
    | EmbeddedInferenceDevicePolicy.CUDA
    | EmbeddedInferenceDevicePolicy.VULKAN;
  httpPlannerUrl: string;
  httpRerankerUrl: string;
  httpDeviceClass: 'cpu' | 'accelerated';
  httpDevice: string;
  httpBackendVersion: string;
}

function parseRequiredCliValues(args: readonly string[]): ReadonlyMap<string, string> {
  if (args.length % 2 !== 0) {
    throw new Error(`Live profile acceptance argument missing value: ${args.at(-1) ?? 'unknown'}`);
  }
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index];
    const value = args[index + 1];
    if (!flag?.startsWith('--') || !value?.trim()) {
      throw new Error(
        `Live profile acceptance argument invalid: ${flag ?? 'missing flag'} ${value ?? 'missing value'}`,
      );
    }
    if (values.has(flag)) {
      throw new Error(`Live profile acceptance argument duplicated: ${flag}`);
    }
    values.set(flag, value.trim());
  }
  return values;
}

function parseLiveProfileAcceptanceCliOptions(
  args: readonly string[],
): LiveProfileAcceptanceCliOptions {
  const values = parseRequiredCliValues(args);
  const expectedFlags = [
    '--accelerated-device',
    '--http-planner-url',
    '--http-reranker-url',
    '--http-device-class',
    '--http-device',
    '--http-backend-version',
  ];
  const unknownFlags = [...values.keys()].filter((flag) => !expectedFlags.includes(flag));
  const missingFlags = expectedFlags.filter((flag) => !values.has(flag));
  if (unknownFlags.length > 0 || missingFlags.length > 0) {
    throw new Error(
      `Live profile acceptance arguments invalid: unknown [${unknownFlags.join(', ')}], missing [${missingFlags.join(', ')}]`,
    );
  }
  const acceleratedDevice = values.get('--accelerated-device');
  if (
    acceleratedDevice !== EmbeddedInferenceDevicePolicy.METAL &&
    acceleratedDevice !== EmbeddedInferenceDevicePolicy.CUDA &&
    acceleratedDevice !== EmbeddedInferenceDevicePolicy.VULKAN
  ) {
    throw new Error(
      `Live profile acceptance accelerated device invalid: ${acceleratedDevice ?? 'missing'}`,
    );
  }
  const httpDeviceClass = values.get('--http-device-class');
  if (httpDeviceClass !== 'cpu' && httpDeviceClass !== 'accelerated') {
    throw new Error(
      `Live profile acceptance HTTP device class invalid: ${httpDeviceClass ?? 'missing'}`,
    );
  }
  const httpPlannerUrl = values.get('--http-planner-url');
  const httpRerankerUrl = values.get('--http-reranker-url');
  const httpDevice = values.get('--http-device');
  const httpBackendVersion = values.get('--http-backend-version');
  if (!httpPlannerUrl || !httpRerankerUrl || !httpDevice || !httpBackendVersion) {
    throw new Error('Live profile acceptance arguments incomplete after validation');
  }
  return {
    acceleratedDevice,
    httpPlannerUrl,
    httpRerankerUrl,
    httpDeviceClass,
    httpDevice,
    httpBackendVersion,
  };
}

function createSha256(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}

function readRecallQualityMetrics(selection: Record<string, unknown>): {
  qualityPassed: boolean;
  candidatePoolRecall: number;
  finalRecall: number;
} {
  const qualityPassed = Reflect.get(selection, 'passed') === true;
  const selected = Reflect.get(selection, 'selected');
  const combinations = Reflect.get(selection, 'combinations');
  const unknownCombinations: unknown[] = Array.isArray(combinations) ? combinations : [];
  const firstCombination = unknownCombinations[0];
  const measurement = isUnknownRecord(selected)
    ? selected
    : isUnknownRecord(firstCombination)
      ? firstCombination
      : null;
  const candidatePoolRecall = measurement
    ? Reflect.get(measurement, 'candidatePoolRecall')
    : undefined;
  const finalRecall = measurement ? Reflect.get(measurement, 'finalRecall') : undefined;
  if (
    typeof candidatePoolRecall !== 'number' ||
    !Number.isFinite(candidatePoolRecall) ||
    typeof finalRecall !== 'number' ||
    !Number.isFinite(finalRecall)
  ) {
    throw new Error('Committed-corpus live evidence invalid: missing finite recall metrics');
  }
  return { qualityPassed, candidatePoolRecall, finalRecall };
}

async function loadEmbeddingGemmaCommittedCorpusEvidence(
  path: string,
  deviceClass: 'cpu' | 'accelerated',
  expectedDevicePolicy: string,
): Promise<CommittedCorpusLiveProfileEvidence> {
  const content = await readFile(path, 'utf8');
  const parsed: unknown = JSON.parse(content);
  if (!isUnknownRecord(parsed)) {
    throw new Error(`Committed-corpus live evidence invalid at ${path}: expected an object`);
  }
  const profile = Reflect.get(parsed, 'profile');
  const backend = Reflect.get(parsed, 'backend');
  const quality = Reflect.get(parsed, 'quality');
  if (!isUnknownRecord(profile) || !isUnknownRecord(backend) || !isUnknownRecord(quality)) {
    throw new Error(`Committed-corpus live evidence invalid at ${path}: missing profile data`);
  }
  const selection = Reflect.get(quality, 'selection');
  if (!isUnknownRecord(selection)) {
    throw new Error(`Committed-corpus live evidence invalid at ${path}: missing quality selection`);
  }
  if (
    Reflect.get(profile, 'profileId') !== 'embeddinggemma-300m-q8-0-v1' ||
    Reflect.get(backend, 'requestedDevicePolicy') !== expectedDevicePolicy
  ) {
    throw new Error(`Committed-corpus live evidence failed profile or device gate at ${path}`);
  }
  return {
    evidenceKind: 'live-profile-candidate',
    deviceClass,
    profileId: 'embeddinggemma-300m-q8-0-v1',
    evidenceSha256: createSha256(content),
    ...readRecallQualityMetrics(selection),
  };
}

async function loadOctenCommittedCorpusEvidence(
  path: string,
): Promise<CommittedCorpusLiveProfileEvidence> {
  const content = await readFile(path, 'utf8');
  const parsed: unknown = JSON.parse(content);
  if (!isUnknownRecord(parsed)) {
    throw new Error(`Octen committed-corpus evidence invalid at ${path}: expected an object`);
  }
  const environment = Reflect.get(parsed, 'environment');
  const result = Reflect.get(parsed, 'result');
  if (!isUnknownRecord(environment) || !isUnknownRecord(result)) {
    throw new Error(`Octen committed-corpus evidence invalid at ${path}: missing result data`);
  }
  const selection = Reflect.get(result, 'selection');
  if (Reflect.get(environment, 'embeddingModel') !== 'octen-embed' || !isUnknownRecord(selection)) {
    throw new Error(`Octen committed-corpus evidence identity invalid at ${path}`);
  }
  const metrics = readRecallQualityMetrics(selection);
  if (!metrics.qualityPassed || metrics.candidatePoolRecall !== 1 || metrics.finalRecall !== 1) {
    throw new Error(`Octen committed-corpus evidence quality gate failed at ${path}`);
  }
  return {
    evidenceKind: 'accepted-hybrid-baseline',
    deviceClass: 'baseline',
    profileId: 'octen-embed',
    evidenceSha256: createSha256(content),
    ...metrics,
  };
}

function createRerankerConformanceFixture() {
  return {
    query: RERANKER_CONFORMANCE_QUERY,
    documents: RERANKER_CONFORMANCE_DOCUMENTS,
    expectedScores: LLAMA_CPP_B8390_CPU_REFERENCE_SCORES,
    maximumAbsoluteDifference: LIVE_RERANKER_MAXIMUM_ABSOLUTE_DIFFERENCE,
  };
}

function createLiveProfileCheckpointIdentity(
  recordedAgainstCommit: string,
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
  profileRun: LiveQueryPlannedProfileRunIdentity,
  profileIdentity: LiveQueryPlannedProfileIdentity,
): LiveProfileEvaluationCheckpointIdentity {
  return {
    version: 3,
    recordedAgainstCommit,
    corpusIdentity: createLiveQueryPlannedEvaluationCorpusIdentity(corpus),
    profileRun,
    profileIdentity,
  };
}

function createLiveProfileSoftwareIdentity(
  recordedAgainstCommit: string,
  backendVersion: string,
): LiveQueryPlannedSoftwareIdentity {
  return {
    repositoryCommit: recordedAgainstCommit,
    backendVersion,
    nodeVersion: process.version,
    platform: process.platform,
    architecture: process.arch,
  };
}

function createEmbeddedProfileEvaluation(
  projectDirectory: string,
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
  recordedAgainstCommit: string,
  device: EmbeddedInferenceDevicePolicy,
  deviceClass: 'cpu' | 'accelerated',
  reportProgress: (message: string) => void,
): CheckpointedLiveProfileEvaluation {
  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  const modelCacheDirectory =
    process.env.PI_RECALL_MODEL_CACHE_DIRECTORY ??
    join(homedir(), '.pi', 'agent', 'recall', 'models');
  const requestTimeoutMilliseconds = 300_000;
  const queryPlanner = createEmbeddedQmdQueryPlanningProvider(queryPlanningProfile, {
    modelCacheDirectory,
    device,
    requestTimeoutMilliseconds,
    onWarning(warning) {
      process.stderr.write(`${warning}\n`);
    },
  });
  const reranker = createEmbeddedQwenRerankingProvider(rerankingProfile, {
    modelCacheDirectory,
    device,
    requestTimeoutMilliseconds,
    onWarning(warning) {
      process.stderr.write(`${warning}\n`);
    },
  });
  const profileRun: LiveQueryPlannedProfileRunIdentity = {
    id: `embedded-${device}`,
    backend: RecallInferenceBackend.EMBEDDED,
    deviceClass,
    device,
    backendVersion: 'node-llama-cpp@3.18.1 / llama.cpp b8390',
  };
  const rerankerConformance = createRerankerConformanceFixture();
  const software = createLiveProfileSoftwareIdentity(
    recordedAgainstCommit,
    'node-llama-cpp@3.18.1 / llama.cpp b8390',
  );
  let baseConfigPromise: Promise<Awaited<ReturnType<typeof loadRecallConversationConfig>>>;
  function loadBaseConfig() {
    baseConfigPromise ??= loadRecallConversationConfig();
    return baseConfigPromise;
  }
  async function resolveProfileIdentity(): Promise<LiveQueryPlannedProfileIdentity> {
    const [baseConfig] = await Promise.all([
      loadBaseConfig(),
      queryPlanner.resolveExecutionIdentity(),
      reranker.resolveExecutionIdentity(),
    ]);
    return createLiveQueryPlannedProfileIdentity({
      evaluationConfiguration: createLiveQueryPlannedEvaluationConfigurationIdentity(
        baseConfig,
        rerankerConformance,
      ),
      software,
      queryPlanningProfile,
      queryPlanningExecutionIdentity: queryPlanner.executionIdentity,
      rerankingProfile,
      rerankingExecutionIdentity: reranker.executionIdentity,
    });
  }
  return {
    profileRun,
    async resolveCheckpointIdentity() {
      return createLiveProfileCheckpointIdentity(
        recordedAgainstCommit,
        corpus,
        profileRun,
        await resolveProfileIdentity(),
      );
    },
    async evaluateProfile() {
      const baseConfig = await loadBaseConfig();
      return runLiveQueryPlannedProfileEvaluation({
        corpus,
        baseConfig,
        workDirectory: join(
          projectDirectory,
          '.recall-data',
          'query-planned-recall',
          `live-profile-${device}`,
        ),
        profileRun,
        evaluationConfiguration: createLiveQueryPlannedEvaluationConfigurationIdentity(
          baseConfig,
          rerankerConformance,
        ),
        software,
        queryPlanningProfile,
        queryPlanner,
        rerankingProfile,
        reranker,
        rerankerConformance,
        reportProgress,
      });
    },
    async disposeProfile() {
      await Promise.all([queryPlanner.dispose(), reranker.dispose()]);
    },
  };
}

function createHttpProfileEvaluation(
  projectDirectory: string,
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
  recordedAgainstCommit: string,
  options: LiveProfileAcceptanceCliOptions,
  reportProgress: (message: string) => void,
): CheckpointedLiveProfileEvaluation {
  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  const requestTimeoutMilliseconds = 300_000;
  const queryPlanner = createQmdHttpQueryPlanningProvider(queryPlanningProfile, {
    baseUrl: options.httpPlannerUrl,
    requestTimeoutMilliseconds,
  });
  const reranker = createQwenHttpRerankingProvider(rerankingProfile, {
    baseUrl: options.httpRerankerUrl,
    requestTimeoutMilliseconds,
  });
  const profileRun: LiveQueryPlannedProfileRunIdentity = {
    id: `http-${options.httpDeviceClass}`,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
    deviceClass: options.httpDeviceClass,
    device: options.httpDevice,
    backendVersion: options.httpBackendVersion,
  };
  const rerankerConformance = createRerankerConformanceFixture();
  const software = createLiveProfileSoftwareIdentity(
    recordedAgainstCommit,
    options.httpBackendVersion,
  );
  let baseConfigPromise: Promise<Awaited<ReturnType<typeof loadRecallConversationConfig>>>;
  function loadBaseConfig() {
    baseConfigPromise ??= loadRecallConversationConfig();
    return baseConfigPromise;
  }
  async function resolveProfileIdentity(): Promise<LiveQueryPlannedProfileIdentity> {
    const baseConfig = await loadBaseConfig();
    return createLiveQueryPlannedProfileIdentity({
      evaluationConfiguration: createLiveQueryPlannedEvaluationConfigurationIdentity(
        baseConfig,
        rerankerConformance,
      ),
      software,
      queryPlanningProfile,
      queryPlanningExecutionIdentity: queryPlanner.executionIdentity,
      rerankingProfile,
      rerankingExecutionIdentity: reranker.executionIdentity,
    });
  }
  return {
    profileRun,
    async resolveCheckpointIdentity() {
      return createLiveProfileCheckpointIdentity(
        recordedAgainstCommit,
        corpus,
        profileRun,
        await resolveProfileIdentity(),
      );
    },
    async evaluateProfile() {
      const baseConfig = await loadBaseConfig();
      return runLiveQueryPlannedProfileEvaluation({
        corpus,
        baseConfig,
        workDirectory: join(
          projectDirectory,
          '.recall-data',
          'query-planned-recall',
          `live-profile-${profileRun.id}`,
        ),
        profileRun,
        evaluationConfiguration: createLiveQueryPlannedEvaluationConfigurationIdentity(
          baseConfig,
          rerankerConformance,
        ),
        software,
        queryPlanningProfile,
        queryPlanner,
        rerankingProfile,
        reranker,
        rerankerConformance,
        reportProgress,
      });
    },
  };
}

function formatRecallEvaluationPublication(path: string, content: string): string {
  return execFileSync(REQUIRE.resolve('oxfmt/bin/oxfmt'), ['--stdin-filepath', path], {
    encoding: 'utf8',
    input: content,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
}

function collectPrivateValuesForAudit(
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
  fixedPlanQueries: readonly string[],
): string[] {
  const values = [
    ...corpus.snapshots.flatMap((snapshot) => [snapshot.path, snapshot.fileName]),
    ...corpus.manifest.cases.flatMap((evaluationCase) => [
      evaluationCase.query,
      ...(evaluationCase.invocationDirectory ? [evaluationCase.invocationDirectory] : []),
      ...evaluationCase.expectedSources.flatMap((source) => [
        source.entryId,
        source.expectedSessionOrigin,
        ...source.requiredText,
      ]),
    ]),
    ...fixedPlanQueries,
  ];
  return [...new Set(values.filter((value) => value.length > 0))];
}

function assertNoPrivateValuesPublished(
  profileRuns: readonly LiveQueryPlannedProfileEvaluationResult[],
  privateValues: readonly string[],
): { checkedValueCount: number; leakCount: 0 } {
  const serialized = JSON.stringify(profileRuns);
  const leakedValues = privateValues.filter((value) => serialized.includes(value));
  if (leakedValues.length > 0) {
    throw new Error(
      `Live profile acceptance privacy audit failed: ${leakedValues.length} private value(s) reached publishable evidence`,
    );
  }
  return { checkedValueCount: privateValues.length, leakCount: 0 };
}

/** Measures the supported live profile matrix and publishes aggregate release evidence. */
export async function evaluateQueryPlannedProfileAcceptance(
  options: LiveProfileAcceptanceCliOptions,
  projectDirectory: string = process.cwd(),
): Promise<PublishableLiveQueryPlannedProfileAcceptance> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const jsonPath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'query-planned-profile-acceptance.json',
  );
  const reportPath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'query-planned-profile-acceptance.md',
  );
  await removeStaleRecallEvaluationTemporaryFiles([jsonPath, reportPath]);
  const recordedAgainstCommit = readCleanRecallEvaluationGitRevision(resolvedProjectDirectory);
  const privateDirectory = join(resolvedProjectDirectory, '.recall-data', 'query-planned-recall');
  const corpus = await loadPrivateQueryPlannedRecallCorpus(join(privateDirectory, 'manifest.json'));
  const fixedPlans = await loadPrivateQueryPlannedRecallPlans(
    join(privateDirectory, 'plans.json'),
    corpus,
  );
  const fixedPlanQueries = fixedPlans.document.cases.flatMap((plannedCase) =>
    plannedCase.queries.map(({ query }) => query),
  );
  const committedCorpus = await Promise.all([
    loadOctenCommittedCorpusEvidence(
      join(resolvedProjectDirectory, 'docs', 'evaluation', 'recall-quality-results.json'),
    ),
    loadEmbeddingGemmaCommittedCorpusEvidence(
      join(resolvedProjectDirectory, 'docs', 'evaluation', 'embeddinggemma-quality-cpu.json'),
      'cpu',
      EmbeddedInferenceDevicePolicy.CPU,
    ),
    loadEmbeddingGemmaCommittedCorpusEvidence(
      join(
        resolvedProjectDirectory,
        'docs',
        'evaluation',
        `embeddinggemma-quality-${options.acceleratedDevice}.json`,
      ),
      'accelerated',
      options.acceleratedDevice,
    ),
  ]);

  const failureSemantics = verifyRequiredRecallEvaluationSemanticChecks(resolvedProjectDirectory);
  const reportProgress = (message: string): void => {
    process.stderr.write(`[${new Date().toISOString()}] ${message}\n`);
  };
  const profiles = [
    createEmbeddedProfileEvaluation(
      resolvedProjectDirectory,
      corpus,
      recordedAgainstCommit,
      EmbeddedInferenceDevicePolicy.CPU,
      'cpu',
      reportProgress,
    ),
    createEmbeddedProfileEvaluation(
      resolvedProjectDirectory,
      corpus,
      recordedAgainstCommit,
      options.acceleratedDevice,
      'accelerated',
      reportProgress,
    ),
    createHttpProfileEvaluation(
      resolvedProjectDirectory,
      corpus,
      recordedAgainstCommit,
      options,
      reportProgress,
    ),
  ];
  const profileRuns = await runCheckpointedLiveProfileEvaluationMatrix({
    checkpointDirectory: join(privateDirectory, 'live-profile-checkpoints'),
    profiles,
    reportProgress,
  });
  const privacyAudit = assertNoPrivateValuesPublished(
    profileRuns,
    collectPrivateValuesForAudit(corpus, fixedPlanQueries),
  );
  const successfulBaselineControlCount = corpus.manifest.cases.filter(
    ({ controlKind }) => controlKind === QueryPlannedRecallControlKind.SUCCESSFUL_BASELINE_CONTROL,
  ).length;
  const evidence = createPublishableLiveQueryPlannedProfileAcceptance({
    recordedAgainstCommit,
    defaultSearchMode: 'hybrid',
    committedCorpus,
    expectedCorpus: createLiveQueryPlannedEvaluationCorpusIdentity(corpus),
    expectedProfileRuns: profiles.map(({ profileRun }) => profileRun),
    profileRuns,
    requiredSuccessfulBaselineControlCount: successfulBaselineControlCount,
    privacyAudit,
    failureSemantics,
  });
  const jsonContent = formatRecallEvaluationPublication(
    jsonPath,
    `${JSON.stringify(evidence, null, 2)}\n`,
  );
  const reportContent = formatRecallEvaluationPublication(
    reportPath,
    formatPublishableLiveQueryPlannedProfileAcceptanceReport(evidence),
  );
  assertRecallEvaluationGitRevisionCurrent(resolvedProjectDirectory, recordedAgainstCommit);
  await writeAtomicRecallEvaluationFile(jsonPath, jsonContent);
  await writeAtomicRecallEvaluationFile(reportPath, reportContent);
  return evidence;
}

async function runLiveProfileAcceptanceCli(args: readonly string[]): Promise<void> {
  if (args.includes('--help') || args.includes('-h')) {
    process.stdout.write(LIVE_PROFILE_ACCEPTANCE_HELP);
    return;
  }
  const options = parseLiveProfileAcceptanceCliOptions(args);
  const evidence = await evaluateQueryPlannedProfileAcceptance(options);
  process.stdout.write(
    `Query-planned explicit fallback acceptance PASS: ${evidence.profileRuns.length} runs, ${evidence.aggregateQuality.newCandidateAdmissionCount} new candidate admission(s), ${evidence.aggregateQuality.plannerFallbackCount} planner fallback(s).\nReport: docs/evaluation/query-planned-profile-acceptance.md\n`,
  );
}

const EXECUTABLE_PATH = process.argv[1];
if (EXECUTABLE_PATH && import.meta.url === pathToFileURL(resolve(EXECUTABLE_PATH)).href) {
  void runLiveProfileAcceptanceCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
