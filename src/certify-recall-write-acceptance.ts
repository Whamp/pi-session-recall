import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

import { createRecallDiagnosticHostIdentity } from './create-recall-diagnostic-host-identity.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  assertRecallEvaluationGitRevisionCurrent,
  readCleanRecallEvaluationGitRevision,
} from './recall-evaluation-git-revision.js';

const WRITE_ACCEPTANCE_COMMAND = 'npm run evidence:target-writes';
const WRITE_ACCEPTANCE_REPORT_PATH = 'docs/evaluation/recall-write-acceptance.md';
const WRITE_ACCEPTANCE_RESULTS_PATH = 'docs/evaluation/recall-write-acceptance.json';
const BEHAVIOR_TEST_PATHS = [
  'src/recall-generation-mutation-ownership.test.ts',
  'src/transfer-incremental-recall-work-plan.test.ts',
  'src/recall-background-index-conversation-service.test.ts',
  'src/pi-session-recall-cli.test.ts',
  'src/activate-validated-recall-generation.test.ts',
  'src/rollback-recall-generation.test.ts',
  'src/run-recall-incremental-worker.test.ts',
  'src/coordinate-recall-marker-replay.test.ts',
  'src/scan-recall-session-metadata.test.ts',
  'src/recall-session-projection.test.ts',
  'src/coordinate-recall-write-window.test.ts',
] as const;
const DISPOSABLE_SOURCE_INPUT_PATHS = [
  ...BEHAVIOR_TEST_PATHS,
  'src/createRecallBackgroundIndexWorkerFixtureService.ts',
  'src/pi-session-recall-cli.test-utils.ts',
  'src/fixtures/session-import/canonical-unicode-separators.jsonl',
  'src/fixtures/session-import/pi-session-reuse-history.jsonl',
  'src/fixtures/session-import/pi-v1-canonical-equivalent.jsonl',
  'src/fixtures/session-import/pi-v1-linear.jsonl',
] as const;
const INCREMENTAL_FAULT_STAGES = [
  'before-recovery-record',
  'after-recovery-record',
  'after-lexical-source-write',
  'after-dense-write',
  'after-logical-projection-write',
  'after-physical-projection-write',
  'after-store-close',
  'after-reopened-verification',
  'after-recovery-clear',
  'after-marker-acknowledgement',
] as const;

interface EvidenceCommandResult {
  name: string;
  command: string;
  durationMilliseconds: number;
  passed: true;
}

interface MarkerPublicationDiagnostic {
  combinedP95Milliseconds: number;
  acceptanceBoundMilliseconds: number;
  accepted: boolean;
}

interface MetadataSweepDiagnostic {
  fileCount: number;
  p95Milliseconds: number;
  acceptanceBoundMilliseconds: number;
  accepted: boolean;
}

interface WriteWindowDiagnostic {
  sampleCount: number;
  p95: { writeWindowMilliseconds: number };
  targetWriteWindowMilliseconds: number;
}

interface RecallWriteAcceptanceEvidence {
  version: 1;
  candidateCommit: string;
  completedAt: string;
  environment: ReturnType<typeof createRecallDiagnosticHostIdentity>;
  disposableSourceSnapshot: {
    kind: 'repository-fixtures-and-generated-test-sources';
    sha256: string;
    inputPaths: readonly string[];
  };
  commands: EvidenceCommandResult[];
  faultMatrix: {
    incrementalStages: readonly string[];
    deletionStages: readonly string[];
    detachedSigkillResume: true;
    activationFixedReplay: true;
    rollbackFixedReplayAndSwitchBack: true;
  };
  foregroundBounds: {
    markerPublicationAndDetachedSpawn: MarkerPublicationDiagnostic;
    metadataSweep: MetadataSweepDiagnostic;
    projectionPayloadMaximumBytes: 8_388_608;
    evidenceBatchMaximumDocuments: 32;
    writeWindow: WriteWindowDiagnostic;
    searchWriteWindowWaitMaximumMilliseconds: 500;
  };
  passed: true;
  safety: {
    productionRecallDatabaseAccessed: false;
    originalPiSessionFilesAccessed: false;
    storage: 'disposable temporary roots with real zvec stores';
  };
}

function formatCommand(argumentsList: readonly string[]): string {
  return [process.execPath, ...argumentsList].join(' ');
}

async function runEvidenceCommand(options: {
  name: string;
  argumentsList: readonly string[];
  projectDirectory: string;
  environment?: NodeJS.ProcessEnv;
}): Promise<EvidenceCommandResult> {
  const startedAt = performance.now();
  await new Promise<void>((resolvePromise, rejectPromise) => {
    const child = spawn(process.execPath, options.argumentsList, {
      cwd: options.projectDirectory,
      env: { ...process.env, ...options.environment },
      stdio: 'inherit',
    });
    child.once('error', (error) => {
      rejectPromise(
        new Error(`Recall write acceptance command failed to start: ${options.name}`, {
          cause: error,
        }),
      );
    });
    child.once('exit', (code, signal) => {
      if (code === 0) {
        resolvePromise();
        return;
      }
      rejectPromise(
        new Error(
          `Recall write acceptance command failed: ${options.name}; exit=${String(code)} signal=${String(signal)}`,
        ),
      );
    });
  });
  return {
    name: options.name,
    command: formatCommand(options.argumentsList),
    durationMilliseconds: performance.now() - startedAt,
    passed: true,
  };
}

async function fingerprintDisposableSourceInputs(projectDirectory: string): Promise<string> {
  const fingerprint = createHash('sha256');
  for (const relativePath of DISPOSABLE_SOURCE_INPUT_PATHS) {
    fingerprint.update(relativePath);
    fingerprint.update('\0');
    fingerprint.update(await readFile(join(projectDirectory, relativePath)));
    fingerprint.update('\0');
  }
  return fingerprint.digest('hex');
}

async function readDiagnosticReport(path: string): Promise<unknown> {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall write acceptance diagnostic report invalid at ${path}: ${message}`, {
      cause: error,
    });
  }
}

function readDiagnosticNumber(
  report: Record<string, unknown>,
  fieldName: string,
  reportPath: string,
): number {
  const value = report[fieldName];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(
      `Recall write acceptance diagnostic number invalid at ${reportPath}: ${fieldName}`,
    );
  }
  return value;
}

function readDiagnosticBoolean(
  report: Record<string, unknown>,
  fieldName: string,
  reportPath: string,
): boolean {
  const value = report[fieldName];
  if (typeof value !== 'boolean') {
    throw new Error(
      `Recall write acceptance diagnostic boolean invalid at ${reportPath}: ${fieldName}`,
    );
  }
  return value;
}

async function readMarkerPublicationDiagnostic(path: string): Promise<MarkerPublicationDiagnostic> {
  const report = await readDiagnosticReport(path);
  if (!isUnknownRecord(report)) {
    throw new Error(`Recall write acceptance marker diagnostic invalid at ${path}`);
  }
  return {
    combinedP95Milliseconds: readDiagnosticNumber(report, 'combinedP95Milliseconds', path),
    acceptanceBoundMilliseconds: readDiagnosticNumber(report, 'acceptanceBoundMilliseconds', path),
    accepted: readDiagnosticBoolean(report, 'accepted', path),
  };
}

async function readMetadataSweepDiagnostic(path: string): Promise<MetadataSweepDiagnostic> {
  const report = await readDiagnosticReport(path);
  if (!isUnknownRecord(report)) {
    throw new Error(`Recall write acceptance metadata diagnostic invalid at ${path}`);
  }
  return {
    fileCount: readDiagnosticNumber(report, 'fileCount', path),
    p95Milliseconds: readDiagnosticNumber(report, 'p95Milliseconds', path),
    acceptanceBoundMilliseconds: readDiagnosticNumber(report, 'acceptanceBoundMilliseconds', path),
    accepted: readDiagnosticBoolean(report, 'accepted', path),
  };
}

async function readWriteWindowDiagnostic(path: string): Promise<WriteWindowDiagnostic> {
  const report = await readDiagnosticReport(path);
  if (!isUnknownRecord(report) || !isUnknownRecord(report.p95)) {
    throw new Error(`Recall write acceptance write-window diagnostic invalid at ${path}`);
  }
  return {
    sampleCount: readDiagnosticNumber(report, 'sampleCount', path),
    p95: {
      writeWindowMilliseconds: readDiagnosticNumber(report.p95, 'writeWindowMilliseconds', path),
    },
    targetWriteWindowMilliseconds: readDiagnosticNumber(
      report,
      'targetWriteWindowMilliseconds',
      path,
    ),
  };
}

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

function formatRecallWriteAcceptanceReport(evidence: RecallWriteAcceptanceEvidence): string {
  const marker = evidence.foregroundBounds.markerPublicationAndDetachedSpawn;
  const sweep = evidence.foregroundBounds.metadataSweep;
  const writeWindow = evidence.foregroundBounds.writeWindow;
  return `# Target write acceptance evidence

**Result:** PASS

- Candidate commit: \`${evidence.candidateCommit}\`
- Completed: ${evidence.completedAt}
- Source snapshot: \`${evidence.disposableSourceSnapshot.sha256}\`
- Runtime: ${evidence.environment.nodeVersion} on ${evidence.environment.platform}/${evidence.environment.architecture}
- CPU: ${evidence.environment.cpuModel}
- zvec: ${evidence.environment.zvecVersion}

## Recovery and lifecycle

The configured service fault matrix passed ${evidence.faultMatrix.incrementalStages.length} incremental interruption stages, including the pre-intent boundary, every cross-store mutation boundary, uncertain close, reopened verification, recovery clearing, and marker acknowledgement. A 33-document transfer also stopped between its 32-document first batch and final physical projection, then resumed idempotently.

A real detached child received SIGKILL and resumed the same generation identity through complete reopened-store validation. Fixed activation replay, rollback replay, bounded rollback health checks, two-generation rollback, and switch-back passed. Append, no-op replay, duplicate delivery, branch exit, compaction, quiescence, context-exit summary, confirmed deletion, and suspicious mass-loss protection passed.

## Foreground bounds

| Bound | Measured or enforced | Limit | Result |
| --- | ---: | ---: | --- |
| Marker publication plus detached spawn p95 | ${marker.combinedP95Milliseconds.toFixed(3)} ms | ${marker.acceptanceBoundMilliseconds} ms | PASS |
| Metadata sweep p95 at ${sweep.fileCount.toLocaleString('en-US')} files | ${sweep.p95Milliseconds.toFixed(3)} ms | ${sweep.acceptanceBoundMilliseconds} ms | PASS |
| Projection payload | enforced | ${evidence.foregroundBounds.projectionPayloadMaximumBytes.toLocaleString('en-US')} bytes | PASS |
| Evidence batch | enforced | ${evidence.foregroundBounds.evidenceBatchMaximumDocuments} documents | PASS |
| Close/reopen write-window p95 (${writeWindow.sampleCount} samples) | ${writeWindow.p95.writeWindowMilliseconds.toFixed(3)} ms | ${writeWindow.targetWriteWindowMilliseconds} ms | PASS |
| Search wait for current write window | enforced | ${evidence.foregroundBounds.searchWriteWindowWaitMaximumMilliseconds} ms | PASS |

## Reproduction

Run:

\`\`\`bash
${WRITE_ACCEPTANCE_COMMAND}
\`\`\`

The command runs these subprocess checks:

${evidence.commands.map(({ command }) => `- \`${command}\``).join('\n')}

The source snapshot hashes committed fixtures and deterministic generated-test sources listed in the JSON evidence. Every storage check uses a disposable temporary root and real zvec stores. The run accessed neither the production recall database nor original Pi session files. No extra native flush or intermediate checkpoint was required.
`;
}

/** Runs the target write, recovery, lifecycle, and foreground-bound acceptance matrix. */
export async function certifyRecallWriteAcceptance(
  projectDirectory: string = process.cwd(),
): Promise<RecallWriteAcceptanceEvidence> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const candidateCommit = readCleanRecallEvaluationGitRevision(resolvedProjectDirectory);
  const diagnosticOutputDirectory = await mkdtemp(join(tmpdir(), 'recall-write-evidence-'));
  const commands: EvidenceCommandResult[] = [];
  try {
    commands.push(
      await runEvidenceCommand({
        name: 'write, recovery, lifecycle, and bounded-policy matrix',
        argumentsList: ['--import', 'tsx', '--test', ...BEHAVIOR_TEST_PATHS],
        projectDirectory: resolvedProjectDirectory,
      }),
    );
    for (const diagnostic of [
      {
        name: 'marker publication and detached spawn diagnostic',
        path: 'src/publish-recall-work-marker.diagnostic.test.ts',
        environment: { PI_RECALL_RUN_MARKER_DIAGNOSTIC: '1' },
      },
      {
        name: 'metadata sweep diagnostic',
        path: 'src/scan-recall-session-metadata.diagnostic.test.ts',
        environment: { PI_RECALL_RUN_METADATA_DIAGNOSTIC: '1' },
      },
      {
        name: 'close/reopen write-window diagnostic',
        path: 'src/commit-incremental-recall-transfer.diagnostic.test.ts',
        environment: { PI_RECALL_RUN_WRITE_WINDOW_DIAGNOSTIC: '1' },
      },
    ] as const) {
      commands.push(
        await runEvidenceCommand({
          name: diagnostic.name,
          argumentsList: ['--import', 'tsx', '--test', diagnostic.path],
          projectDirectory: resolvedProjectDirectory,
          environment: {
            ...diagnostic.environment,
            PI_RECALL_EVIDENCE_OUTPUT_DIRECTORY: diagnosticOutputDirectory,
          },
        }),
      );
    }

    const markerPublicationAndDetachedSpawn = await readMarkerPublicationDiagnostic(
      join(diagnosticOutputDirectory, 'marker-publication.json'),
    );
    const metadataSweep = await readMetadataSweepDiagnostic(
      join(diagnosticOutputDirectory, 'metadata-sweep.json'),
    );
    const writeWindow = await readWriteWindowDiagnostic(
      join(diagnosticOutputDirectory, 'write-window.json'),
    );
    if (
      !markerPublicationAndDetachedSpawn.accepted ||
      !metadataSweep.accepted ||
      writeWindow.p95.writeWindowMilliseconds > writeWindow.targetWriteWindowMilliseconds
    ) {
      throw new Error('Recall write acceptance foreground diagnostic rejected the candidate');
    }

    const evidence: RecallWriteAcceptanceEvidence = {
      version: 1,
      candidateCommit,
      completedAt: new Date().toISOString(),
      environment: createRecallDiagnosticHostIdentity(),
      disposableSourceSnapshot: {
        kind: 'repository-fixtures-and-generated-test-sources',
        sha256: await fingerprintDisposableSourceInputs(resolvedProjectDirectory),
        inputPaths: DISPOSABLE_SOURCE_INPUT_PATHS,
      },
      commands,
      faultMatrix: {
        incrementalStages: INCREMENTAL_FAULT_STAGES,
        deletionStages: [
          'before-recovery-record',
          'after-recovery-record',
          'after-dense-delete',
          'after-lexical-source-delete',
          'after-projection-delete',
          'after-store-close',
          'after-reopened-verification',
          'after-recovery-clear',
        ],
        detachedSigkillResume: true,
        activationFixedReplay: true,
        rollbackFixedReplayAndSwitchBack: true,
      },
      foregroundBounds: {
        markerPublicationAndDetachedSpawn,
        metadataSweep,
        projectionPayloadMaximumBytes: 8_388_608,
        evidenceBatchMaximumDocuments: 32,
        writeWindow,
        searchWriteWindowWaitMaximumMilliseconds: 500,
      },
      passed: true,
      safety: {
        productionRecallDatabaseAccessed: false,
        originalPiSessionFilesAccessed: false,
        storage: 'disposable temporary roots with real zvec stores',
      },
    };

    assertRecallEvaluationGitRevisionCurrent(resolvedProjectDirectory, candidateCommit);
    const resultsPath = join(resolvedProjectDirectory, WRITE_ACCEPTANCE_RESULTS_PATH);
    const reportPath = join(resolvedProjectDirectory, WRITE_ACCEPTANCE_REPORT_PATH);
    await writeAtomicTextFile(resultsPath, `${JSON.stringify(evidence, null, 2)}\n`);
    await writeAtomicTextFile(reportPath, formatRecallWriteAcceptanceReport(evidence));
    execFileSync('npx', ['--no-install', 'oxfmt', resultsPath, reportPath], {
      cwd: resolvedProjectDirectory,
      stdio: 'pipe',
    });
    return evidence;
  } finally {
    await rm(diagnosticOutputDirectory, { recursive: true, force: true });
  }
}

await certifyRecallWriteAcceptance();
