import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createPublishableQueryPlannedRecallBaselineEvidence,
  createPublishableQueryPlannedRecallControls,
  formatPublishableQueryPlannedRecallBaselineReport,
  loadPrivateQueryPlannedRecallCorpus,
  runPrivateQueryPlannedRecallBaseline,
  type PublishableQueryPlannedRecallBaselineEvidence,
} from './query-planned-recall-baseline.js';

const QUERY_PLANNED_RECALL_BASELINE_HELP = `Usage: npm run evaluate:query-planned-baseline

Loads permission-restricted inputs from:
  .recall-data/query-planned-recall/manifest.json

Publishes privacy-safe controls and aggregate evidence to:
  evaluation/query-planned-recall-controls.json
  docs/evaluation/query-planned-hybrid-baseline.json
  docs/evaluation/query-planned-hybrid-baseline.md

The command indexes only unchanged snapshots declared by the private manifest and never scans or writes the production recall corpus. Query text, source text, entry IDs, session paths, project details, and relevant distractor identities stay private.
`;

async function writeAtomicBaselineFile(path: string, content: string): Promise<void> {
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

/** Runs the private pre-planning hybrid baseline and writes only privacy-safe repository evidence. */
export async function evaluateQueryPlannedRecallBaseline(
  projectDirectory: string = process.cwd(),
): Promise<PublishableQueryPlannedRecallBaselineEvidence> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const privateDirectory = join(resolvedProjectDirectory, '.recall-data', 'query-planned-recall');
  const corpus = await loadPrivateQueryPlannedRecallCorpus(join(privateDirectory, 'manifest.json'));
  const config = await loadRecallConversationConfig();
  const controls = createPublishableQueryPlannedRecallControls(corpus);
  const baseline = await runPrivateQueryPlannedRecallBaseline({
    corpus,
    baseConfig: config,
    workDirectory: join(privateDirectory, 'baseline-work'),
  });
  const evidence = createPublishableQueryPlannedRecallBaselineEvidence(controls, baseline, {
    recordedAgainstCommit: execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: resolvedProjectDirectory,
      encoding: 'utf8',
    }).trim(),
    embeddingProfile: {
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: config.embeddingDimensions,
      quantization: config.embeddingQuantization,
      pooling: config.embeddingPooling,
    },
  });
  const controlsPath = join(
    resolvedProjectDirectory,
    'evaluation',
    'query-planned-recall-controls.json',
  );
  const evidencePath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'query-planned-hybrid-baseline.json',
  );
  const reportPath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'query-planned-hybrid-baseline.md',
  );
  await Promise.all([
    writeAtomicBaselineFile(controlsPath, `${JSON.stringify(controls, null, 2)}\n`),
    writeAtomicBaselineFile(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`),
    writeAtomicBaselineFile(
      reportPath,
      formatPublishableQueryPlannedRecallBaselineReport(evidence),
    ),
  ]);
  execFileSync(
    join(resolvedProjectDirectory, 'node_modules', '.bin', 'oxfmt'),
    [controlsPath, evidencePath, reportPath],
    { cwd: resolvedProjectDirectory, stdio: 'pipe' },
  );
  return evidence;
}

async function runQueryPlannedRecallBaselineCli(args: readonly string[]): Promise<void> {
  if (args.includes('--help') || args.includes('-h')) {
    process.stdout.write(QUERY_PLANNED_RECALL_BASELINE_HELP);
    return;
  }
  if (args.length > 0) {
    throw new Error(`Query-planned recall baseline CLI argument unsupported: ${args.join(' ')}`);
  }
  const evidence = await evaluateQueryPlannedRecallBaseline();
  process.stdout.write(
    `Query-planned recall hybrid baseline recorded: ${evidence.baseline.cases.length} cases, ${evidence.baseline.executedSearchRequests} searches.\nReport: docs/evaluation/query-planned-hybrid-baseline.md\n`,
  );
}

const executablePath = process.argv[1];
if (executablePath && import.meta.url === pathToFileURL(resolve(executablePath)).href) {
  void runQueryPlannedRecallBaselineCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
