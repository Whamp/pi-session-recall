import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { cpus, arch, platform } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  createDeterministicRecallQualityDependencies,
  DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS,
} from './create-deterministic-recall-quality-dependencies.js';
import { RecallDiagnosticsMode } from './enums.js';
import {
  formatRecallQualityReport,
  type RecallQualityReportEnvironment,
} from './format-recall-quality-report.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import { loadRecallQualityCorpus } from './recall-quality-corpus.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import {
  runRecallQualityEvaluation,
  type RecallQualityEvaluationResult,
} from './run-recall-quality-evaluation.js';

const RECALL_QUALITY_HELP = `Usage: npm run evaluate:recall

Builds and activates one disposable target generation over a copied checksum-fixed evaluation corpus and never scans the production session corpus.

Outputs:
  docs/evaluation/recall-quality-report.md
  docs/evaluation/recall-quality-results.json

The command exits 0 when a measured configuration passes every frozen gate and 2 when quality or latency is insufficient. It never launches the full corpus backfill.
`;

async function writeAtomicTextFile(path: string, content: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, content, { encoding: 'utf8', flag: 'wx' });
    await rename(temporaryPath, path);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

function createReportEnvironment(
  projectDirectory: string,
  config: RecallConversationConfig,
): RecallQualityReportEnvironment {
  return {
    command: process.env.PI_RECALL_QUALITY_EVIDENCE_COMMAND ?? 'npm run evaluate:recall',
    gitCommit: execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: projectDirectory,
      encoding: 'utf8',
    }).trim(),
    gitDirty:
      execFileSync('git', ['status', '--porcelain'], {
        cwd: projectDirectory,
        encoding: 'utf8',
      }).trim().length > 0,
    nodeVersion: process.version,
    platform: platform(),
    architecture: arch(),
    cpuModel: cpus()[0]?.model ?? 'unknown',
    embeddingBaseUrl: config.embeddingBaseUrl,
    embeddingModel: config.embeddingModel,
    embeddingServedModelId: config.embeddingServedModelId,
    embeddingArtifact: config.embeddingArtifact,
    embeddingDimensions: config.embeddingDimensions,
    rerankerBaseUrl: config.rerankerBaseUrl,
    rerankerModel: config.rerankerModel,
  };
}

/** Creates the production-isolated deterministic configuration for committed-corpus quality runs. */
export function createDeterministicRecallQualityConfig(
  projectDirectory: string,
): RecallConversationConfig {
  const protectedDirectory = join(projectDirectory, 'evaluation', '.protected-production-sentinel');
  return {
    sessionsDirectory: join(protectedDirectory, 'sessions'),
    dataDirectory: protectedDirectory,
    databasePath: join(protectedDirectory, 'zvec'),
    projectionDatabasePath: join(protectedDirectory, 'session-projections'),
    statePath: join(protectedDirectory, 'index-state.json'),
    manifestPath: join(protectedDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(protectedDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(protectedDirectory, 'embedding-cache'),
    lockPath: join(protectedDirectory, 'operation.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(protectedDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(protectedDirectory, 'diagnostics.previous.jsonl'),
    markerSpoolDirectory: join(protectedDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: join(protectedDirectory, 'markers', 'quarantine'),
    markerControlDirectory: join(protectedDirectory, 'markers', 'control'),
    workerOwnershipLockPath: join(protectedDirectory, 'incremental-worker.lock'),
    generationRootDirectory: join(protectedDirectory, 'generations'),
    activeGenerationPointerPath: join(protectedDirectory, 'active-generation.json'),
    generationRegistryPath: join(protectedDirectory, 'generation-registry.json'),
    backlogSummaryPath: join(protectedDirectory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(protectedDirectory, 'incremental-diagnostics.jsonl'),
    embeddingBaseUrl: 'in-process://deterministic-fixture-v1',
    embeddingModel: 'deterministic-fixture-v1',
    embeddingServedModelId: 'deterministic-fixture-v1',
    embeddingArtifact: 'committed-corpus-concept-hash-fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'deterministic-fixture',
    embeddingDimensions: DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS,
    embeddingBatchSize: 16,
    rerankerBaseUrl: 'disabled://unexpected-request-fails',
    rerankerModel: 'rejecting-fake',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
  };
}

/** Runs the fixed project-scope evaluation and atomically writes its Markdown and JSON evidence. */
export async function evaluateRecallQuality(
  projectDirectory: string = process.cwd(),
): Promise<RecallQualityEvaluationResult> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const corpus = await loadRecallQualityCorpus(
    join(resolvedProjectDirectory, 'evaluation', 'recall-quality-cases.json'),
  );
  const config = createDeterministicRecallQualityConfig(resolvedProjectDirectory);
  const environment = createReportEnvironment(resolvedProjectDirectory, config);
  const workDirectory = join(
    resolvedProjectDirectory,
    'evaluation',
    '.recall-data',
    'recall-quality-evaluation',
  );
  let result: RecallQualityEvaluationResult;
  try {
    result = await runRecallQualityEvaluation({
      corpus,
      baseConfig: config,
      workDirectory,
      dependencies: createDeterministicRecallQualityDependencies(),
    });
  } finally {
    await rm(workDirectory, { recursive: true, force: true });
  }
  const reportPath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'recall-quality-report.md',
  );
  const resultsPath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'recall-quality-results.json',
  );
  await writeAtomicTextFile(
    resultsPath,
    `${JSON.stringify(
      {
        version: 2,
        environment,
        specification: corpus.specification,
        result,
      },
      null,
      2,
    )}\n`,
  );
  await writeAtomicTextFile(reportPath, formatRecallQualityReport(result, corpus, environment));
  execFileSync('npx', ['--no-install', 'oxfmt', reportPath, resultsPath], {
    cwd: resolvedProjectDirectory,
    stdio: 'pipe',
  });
  return result;
}

async function runRecallQualityCli(args: readonly string[]): Promise<void> {
  if (args.includes('--help') || args.includes('-h')) {
    process.stdout.write(RECALL_QUALITY_HELP);
    return;
  }
  if (args.length > 0) {
    throw new Error(`Recall quality CLI argument unsupported: ${args.join(' ')}`);
  }
  const result = await evaluateRecallQuality();
  const selected = result.selection.selected;
  const decision = selected
    ? `${selected.chunkPolicy.id}, ${selected.candidateCount} candidates/channel, ${selected.finalCount} final`
    : result.selection.blockers.join('; ');
  process.stdout.write(
    `Recall quality automated gate ${result.selection.passed ? 'PASS' : 'FAIL'}: ${decision}\nReport: docs/evaluation/recall-quality-report.md\n`,
  );
  if (!result.selection.passed) {
    process.exitCode = 2;
  }
}

const executablePath = process.argv[1];
if (executablePath && import.meta.url === pathToFileURL(resolve(executablePath)).href) {
  void runRecallQualityCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
