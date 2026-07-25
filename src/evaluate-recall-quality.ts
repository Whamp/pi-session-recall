import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { cpus, arch, platform } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  formatRecallQualityReport,
  type RecallQualityReportEnvironment,
} from './format-recall-quality-report.js';
import { loadRecallQualityCorpus } from './recall-quality-corpus.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  runRecallQualityEvaluation,
  type RecallQualityEvaluationResult,
} from './run-recall-quality-evaluation.js';

const RECALL_QUALITY_HELP = `Usage: npm run evaluate:recall

Builds one temporary index over the checksum-fixed evaluation corpus and never scans the production session corpus.

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
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>,
): RecallQualityReportEnvironment {
  return {
    command: 'npm run evaluate:recall',
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

/** Runs the fixed project-scope evaluation and atomically writes its Markdown and JSON evidence. */
export async function evaluateRecallQuality(
  projectDirectory: string = process.cwd(),
): Promise<RecallQualityEvaluationResult> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const corpus = await loadRecallQualityCorpus(
    join(resolvedProjectDirectory, 'evaluation', 'recall-quality-cases.json'),
  );
  const config = await loadRecallConversationConfig();
  const environment = createReportEnvironment(resolvedProjectDirectory, config);
  const result = await runRecallQualityEvaluation({
    corpus,
    baseConfig: config,
    workDirectory: join(
      resolvedProjectDirectory,
      'evaluation',
      '.recall-data',
      'recall-quality-evaluation',
    ),
  });
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
  execFileSync(
    join(resolvedProjectDirectory, 'node_modules', '.bin', 'oxfmt'),
    [reportPath, resultsPath],
    { cwd: resolvedProjectDirectory, stdio: 'pipe' },
  );
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
