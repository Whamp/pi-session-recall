import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { arch, cpus, homedir, platform } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

import {
  createEmbeddedEmbeddingGemmaProvider,
  createEmbeddingGemmaTokenizerManifestIdentity,
  type EmbeddedEmbeddingGemmaExecutionIdentity,
} from './embedded-embeddinggemma-provider.js';
import { EmbeddedInferenceDevicePolicy } from './enums.js';
import {
  createEmbeddingVectorCacheIdentity,
  type EmbeddingVectorCacheIdentity,
} from './embedding-vector-cache.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallEmbeddingCapabilityVerification,
} from './recall-conversation-service.js';
import {
  readRecallIndexManifest,
  type RecallTokenizerManifestIdentity,
} from './recall-index-manifest.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  type RecommendedEmbeddingGemmaModelProfile,
} from './recall-model-profiles.js';
import { loadRecallQualityCorpus } from './recall-quality-corpus.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  runRecallQualityEvaluation,
  type RecallQualityEvaluationIdentity,
  type RecallQualityEvaluationResult,
} from './run-recall-quality-evaluation.js';

const EMBEDDING_GEMMA_QUALITY_HELP = `Usage: npm run evaluate:embeddinggemma -- --device <auto|cpu|metal|cuda|vulkan>

Runs the checksum-fixed recall regression corpus with the pinned embedded EmbeddingGemma profile.
The command never downloads a model, never scans production sessions, and never changes the
existing Octen report. The pinned artifact must already be approved and verified in the model cache.

Outputs:
  docs/evaluation/embeddinggemma-quality-<device>.json
`;

function isEmbeddedInferenceDevicePolicy(value: string): value is EmbeddedInferenceDevicePolicy {
  return (
    value === String(EmbeddedInferenceDevicePolicy.AUTO) ||
    value === String(EmbeddedInferenceDevicePolicy.CPU) ||
    value === String(EmbeddedInferenceDevicePolicy.METAL) ||
    value === String(EmbeddedInferenceDevicePolicy.CUDA) ||
    value === String(EmbeddedInferenceDevicePolicy.VULKAN)
  );
}

/** Real profile, execution, quality, timing, and storage evidence from one device-policy run. */
export interface EmbeddingGemmaQualityEvidence {
  version: 1;
  evidenceKind: 'embeddinggemma-live-quality-candidate';
  releaseDecision: 'maintainer-review-required';
  command: string;
  startedAt: string;
  completedAt: string;
  environment: {
    gitCommit: string;
    gitDirty: boolean;
    nodeVersion: string;
    platform: string;
    architecture: string;
    cpuModel: string;
  };
  profile: RecommendedEmbeddingGemmaModelProfile;
  backend: {
    adapter: 'node-llama-cpp-embedded-v2';
    backend: 'embedded';
    requestedDevicePolicy: EmbeddedInferenceDevicePolicy;
    executionIdentity: Readonly<EmbeddedEmbeddingGemmaExecutionIdentity>;
  };
  tokenizer: RecallTokenizerManifestIdentity;
  embeddingCacheIdentity: EmbeddingVectorCacheIdentity;
  candidatePolicy: RecallQualityEvaluationIdentity;
  capabilityVerification: RecallEmbeddingCapabilityVerification;
  measurements: {
    coldStartMilliseconds: number;
    warmQueryMilliseconds: number;
    warmDocumentBatchMilliseconds: number;
    indexingThroughput: {
      sourceBytesPerSecond: number;
      denseDocumentsPerSecond: number;
    };
    indexSizeBytes: number;
    embeddingCacheSizeBytes: number;
  };
  quality: RecallQualityEvaluationResult;
  limitations: readonly string[];
}

async function writeAtomicJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
    await rename(temporaryPath, path);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

async function measureDirectoryByteSize(path: string): Promise<number> {
  let entries;
  try {
    entries = await readdir(path, { withFileTypes: true });
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return 0;
    }
    throw error;
  }
  let byteSize = 0;
  for (const entry of entries) {
    const entryPath = join(path, entry.name);
    if (entry.isDirectory()) {
      byteSize += await measureDirectoryByteSize(entryPath);
    } else if (entry.isFile()) {
      byteSize += (await stat(entryPath)).size;
    }
  }
  return byteSize;
}

function parseEmbeddingGemmaQualityDevice(args: readonly string[]): EmbeddedInferenceDevicePolicy {
  const deviceFlagIndex = args.indexOf('--device');
  if (deviceFlagIndex < 0 || deviceFlagIndex === args.length - 1) {
    throw new Error(
      'EmbeddingGemma quality device required: pass --device auto, cpu, metal, cuda, or vulkan',
    );
  }
  if (args.length !== 2 || deviceFlagIndex !== 0) {
    throw new Error(`EmbeddingGemma quality argument unsupported: ${args.join(' ')}`);
  }
  const device = args[deviceFlagIndex + 1];
  if (!device || !isEmbeddedInferenceDevicePolicy(device)) {
    throw new Error(`EmbeddingGemma quality device invalid: ${device ?? 'missing'}`);
  }
  return device;
}

/** Runs real EmbeddingGemma over the fixed corpus without downloading or touching Octen evidence. */
export async function evaluateEmbeddingGemmaRecallQuality(
  device: EmbeddedInferenceDevicePolicy,
  projectDirectory: string = process.cwd(),
): Promise<EmbeddingGemmaQualityEvidence> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const corpus = await loadRecallQualityCorpus(
    join(resolvedProjectDirectory, 'evaluation', 'recall-quality-cases.json'),
  );
  const baseConfig = await loadRecallConversationConfig();
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const tokenizerIdentity = createEmbeddingGemmaTokenizerManifestIdentity(profile);
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory:
      process.env.PI_RECALL_MODEL_CACHE_DIRECTORY ??
      join(homedir(), '.pi', 'agent', 'recall', 'models'),
    device,
    onWarning(warning) {
      process.stderr.write(`${warning}\n`);
    },
  });
  const workDirectory = join(
    resolvedProjectDirectory,
    'evaluation',
    '.recall-data',
    `embeddinggemma-quality-${device}`,
    'recall-quality-evaluation',
  );

  try {
    const startedAt = new Date().toISOString();
    const verificationService = createRecallConversationService(baseConfig, {
      embeddingProfile: profile,
      embeddingProvider: provider,
      tokenizerIdentity,
      loadTokenizer: () => provider.loadConversationTokenizer(),
      rerankingProfile: null,
      reranker: null,
    });
    const coldStartStarted = performance.now();
    const capabilityVerification = await verificationService.verifyEmbeddingCapability();
    const coldStartMilliseconds = performance.now() - coldStartStarted;

    const quality = await runRecallQualityEvaluation({
      corpus,
      baseConfig,
      workDirectory,
      dependencies: {
        embeddingProfile: profile,
        embeddingProvider: provider,
        tokenizerIdentity,
        loadTokenizer: () => provider.loadConversationTokenizer(),
      },
    });

    const warmQueryStarted = performance.now();
    await provider.embedQuery('Which source-backed decision did the conversation retain?');
    const warmQueryMilliseconds = performance.now() - warmQueryStarted;
    const warmDocumentBatchStarted = performance.now();
    await provider.embedDocuments([
      'The active branch retained source-backed evidence.',
      'The abandoned branch remains eligible and labeled.',
    ]);
    const warmDocumentBatchMilliseconds = performance.now() - warmDocumentBatchStarted;

    const chunkPolicy = corpus.specification.chunkPolicies[0];
    const indexRun = quality.indexRuns[0];
    if (!chunkPolicy || !indexRun) {
      throw new Error('EmbeddingGemma quality result incomplete: expected one index run');
    }
    const policyDirectory = join(workDirectory, chunkPolicy.id);
    const manifestPath = join(policyDirectory, 'index-manifest.json');
    const manifest = await readRecallIndexManifest(manifestPath);
    if (!manifest) {
      throw new Error(`EmbeddingGemma quality manifest missing at ${manifestPath}`);
    }
    const indexSeconds = indexRun.indexLatencyMilliseconds / 1_000;
    const sourceByteSizes = await Promise.all(
      corpus.sessionFiles.map(async (sessionFile) => (await stat(sessionFile.path)).size),
    );
    const sourceByteSize = sourceByteSizes.reduce((total, byteSize) => total + byteSize, 0);
    const evidence: EmbeddingGemmaQualityEvidence = {
      version: 1,
      evidenceKind: 'embeddinggemma-live-quality-candidate',
      releaseDecision: 'maintainer-review-required',
      command: `npm run evaluate:embeddinggemma -- --device ${device}`,
      startedAt,
      completedAt: new Date().toISOString(),
      environment: {
        gitCommit: execFileSync('git', ['rev-parse', 'HEAD'], {
          cwd: resolvedProjectDirectory,
          encoding: 'utf8',
        }).trim(),
        gitDirty:
          execFileSync('git', ['status', '--porcelain'], {
            cwd: resolvedProjectDirectory,
            encoding: 'utf8',
          }).trim().length > 0,
        nodeVersion: process.version,
        platform: platform(),
        architecture: arch(),
        cpuModel: cpus()[0]?.model ?? 'unknown',
      },
      profile,
      backend: {
        adapter: 'node-llama-cpp-embedded-v2',
        backend: 'embedded',
        requestedDevicePolicy: device,
        executionIdentity: provider.executionIdentity,
      },
      tokenizer: tokenizerIdentity,
      embeddingCacheIdentity: createEmbeddingVectorCacheIdentity(manifest),
      candidatePolicy: quality.evaluationIdentity,
      capabilityVerification,
      measurements: {
        coldStartMilliseconds,
        warmQueryMilliseconds,
        warmDocumentBatchMilliseconds,
        indexingThroughput: {
          sourceBytesPerSecond: indexSeconds > 0 ? sourceByteSize / indexSeconds : 0,
          denseDocumentsPerSecond:
            indexSeconds > 0 ? indexRun.indexSummary.newlyEmbeddedChunks / indexSeconds : 0,
        },
        indexSizeBytes: await measureDirectoryByteSize(join(policyDirectory, 'zvec')),
        embeddingCacheSizeBytes: await measureDirectoryByteSize(
          join(policyDirectory, 'embedding-cache'),
        ),
      },
      quality,
      limitations: [
        'This committed corpus is synthetic-but-session-shaped and cannot establish private-corpus quality.',
        'One device-policy run does not provide both CPU and accelerated-device evidence.',
        'Maintainer distribution and notice approval remains a separate release decision.',
        'Independent live tokenizer, embedding canary, HTTP parity, reranker, and planner fixtures remain separate acceptance evidence.',
      ],
    };
    const outputPath = join(
      resolvedProjectDirectory,
      'docs',
      'evaluation',
      `embeddinggemma-quality-${device}.json`,
    );
    await writeAtomicJson(outputPath, evidence);
    execFileSync(join(resolvedProjectDirectory, 'node_modules', '.bin', 'oxfmt'), [outputPath], {
      cwd: resolvedProjectDirectory,
      stdio: 'pipe',
    });
    return evidence;
  } finally {
    await provider.dispose();
  }
}

async function runEmbeddingGemmaQualityCli(args: readonly string[]): Promise<void> {
  if (args.includes('--help') || args.includes('-h')) {
    process.stdout.write(EMBEDDING_GEMMA_QUALITY_HELP);
    return;
  }
  const device = parseEmbeddingGemmaQualityDevice(args);
  const evidence = await evaluateEmbeddingGemmaRecallQuality(device);
  process.stdout.write(
    `EmbeddingGemma quality gate ${evidence.quality.selection.passed ? 'PASS' : 'FAIL'} on ${evidence.backend.executionIdentity.computeBackend}; maintainer review remains required.\n`,
  );
  if (!evidence.quality.selection.passed) {
    process.exitCode = 2;
  }
}

const EXECUTABLE_PATH = process.argv[1];
if (EXECUTABLE_PATH && import.meta.url === pathToFileURL(resolve(EXECUTABLE_PATH)).href) {
  void runEmbeddingGemmaQualityCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
