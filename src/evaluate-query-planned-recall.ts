import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { writeAtomicRecallEvaluationFile } from './recall-evaluation-file-system.js';
import {
  createPublishableQueryPlannedRecallControls,
  loadPrivateQueryPlannedRecallCorpus,
} from './query-planned-recall-baseline.js';
import {
  createPublishableQueryPlannedRecallEvaluationEvidence,
  formatPublishableQueryPlannedRecallEvaluationReport,
  loadPrivateQueryPlannedRecallPlans,
  runPrivateQueryPlannedRecallEvaluation,
  type PublishableQueryPlannedRecallEvaluationEvidence,
} from './query-planned-recall-evaluation.js';

const QUERY_PLANNED_RECALL_EVALUATION_HELP = `Usage: npm run evaluate:query-planned

Loads permission-restricted inputs from:
  .recall-data/query-planned-recall/manifest.json
  .recall-data/query-planned-recall/plans.json

Runs normal hybrid, retrieval-work-matched original-query, and fixed query-planned searches with deterministic token-hash embeddings and controlled rerankers.

Publishes aggregate quality evidence to:
  docs/evaluation/query-planned-recall-quality.json
  docs/evaluation/query-planned-recall-quality.md

The command indexes only unchanged snapshots declared by the private manifest, never scans or writes the production recall corpus, and never publishes private query or source text.
`;

/** Runs deterministic fixed-plan recall quality and writes only aggregate repository evidence. */
export async function evaluateQueryPlannedRecall(
  projectDirectory: string = process.cwd(),
): Promise<PublishableQueryPlannedRecallEvaluationEvidence> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const privateDirectory = join(resolvedProjectDirectory, '.recall-data', 'query-planned-recall');
  const corpus = await loadPrivateQueryPlannedRecallCorpus(join(privateDirectory, 'manifest.json'));
  const plans = await loadPrivateQueryPlannedRecallPlans(
    join(privateDirectory, 'plans.json'),
    corpus,
  );
  const controls = createPublishableQueryPlannedRecallControls(corpus);
  const config = await loadRecallConversationConfig();
  const evaluation = await runPrivateQueryPlannedRecallEvaluation({
    corpus,
    plans,
    baseConfig: config,
    workDirectory: join(privateDirectory, 'evaluation-work'),
  });
  const evidence = createPublishableQueryPlannedRecallEvaluationEvidence(controls, evaluation, {
    recordedAgainstCommit: execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: resolvedProjectDirectory,
      encoding: 'utf8',
    }).trim(),
  });
  const evidencePath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'query-planned-recall-quality.json',
  );
  const reportPath = join(
    resolvedProjectDirectory,
    'docs',
    'evaluation',
    'query-planned-recall-quality.md',
  );
  await Promise.all([
    writeAtomicRecallEvaluationFile(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`),
    writeAtomicRecallEvaluationFile(
      reportPath,
      formatPublishableQueryPlannedRecallEvaluationReport(evidence),
    ),
  ]);
  execFileSync(
    join(resolvedProjectDirectory, 'node_modules', '.bin', 'oxfmt'),
    [evidencePath, reportPath],
    { cwd: resolvedProjectDirectory, stdio: 'pipe' },
  );
  return evidence;
}

async function runQueryPlannedRecallEvaluationCli(args: readonly string[]): Promise<void> {
  if (args.includes('--help') || args.includes('-h')) {
    process.stdout.write(QUERY_PLANNED_RECALL_EVALUATION_HELP);
    return;
  }
  if (args.length > 0) {
    throw new Error(`Query-planned recall evaluation CLI argument unsupported: ${args.join(' ')}`);
  }
  const evidence = await evaluateQueryPlannedRecall();
  process.stdout.write(
    `Query-planned recall quality recorded: ${evidence.evaluation.cases.length} cases, ${evidence.evaluation.executedSearchRequests} searches, ${evidence.evaluation.contributionCounts.newCandidateAdmission} new candidate admission(s).\nReport: docs/evaluation/query-planned-recall-quality.md\n`,
  );
}

const executablePath = process.argv[1];
if (executablePath && import.meta.url === pathToFileURL(resolve(executablePath)).href) {
  void runQueryPlannedRecallEvaluationCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
