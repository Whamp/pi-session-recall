import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import { createRecallDiagnosticHostIdentity } from './create-recall-diagnostic-host-identity.js';
import { writeAtomicRecallEvaluationFile } from './recall-evaluation-file-system.js';
import {
  assertRecallEvaluationGitRevisionCurrent,
  readCleanRecallEvaluationGitRevision,
} from './recall-evaluation-git-revision.js';
import {
  runRecallGenerationRecoveryPreflight,
  type RecallGenerationRecoveryPreflightResult,
} from './recall-generation-recovery-preflight.js';

const RECOVERY_CERTIFICATION_COMMAND = 'npm run evidence:generation-recovery';
const RECOVERY_CERTIFICATION_REPORT_PATH =
  'docs/evaluation/recall-generation-recovery-preflight.md';
const RECOVERY_CERTIFICATION_RESULTS_PATH =
  'docs/evaluation/recall-generation-recovery-preflight.json';
const PRODUCTION_FAILURE_BOUNDARY = 119_662;
const PRODUCTION_LOGICAL_SESSION_COUNT = PRODUCTION_FAILURE_BOUNDARY + 1;
const SLOP_SCAN_BASE_COMMIT = 'b04b350939de11ae56b67f8d1e8cce9ab0b12ec8';
const CERTIFICATION_INPUT_PATHS = [
  'src/recall-generation-recovery-preflight.ts',
  'src/recall-generation-recovery-preflight.test.ts',
  'src/certify-recall-generation-recovery.ts',
  'src/certify-recall-generation-recovery.test.ts',
  'src/build-recall-fixed-snapshot-generation.test.ts',
  'src/recall-physical-source-generation.test.ts',
  'src/transfer-incremental-recall-work-plan.test.ts',
  'src/recall-background-index-conversation-service.test.ts',
  'src/pi-session-recall-cli.test.ts',
] as const;

/** One successful command recorded by generation recovery certification. */
export interface RecoveryCertificationCommandResult {
  name: string;
  command: string;
  durationMilliseconds: number;
  passed: true;
}

interface RecallGenerationRecoveryCertificationEvidence {
  version: 1;
  candidateCommit: string;
  completedAt: string;
  environment: ReturnType<typeof createRecallDiagnosticHostIdentity>;
  certificationInputChecksum: string;
  commands: RecoveryCertificationCommandResult[];
  preflightDurationMilliseconds: number;
  productionFailureBoundary: number;
  preflight: RecallGenerationRecoveryPreflightResult;
  incrementalVsRebuildMembershipEquivalent: true;
  detachedWorkerReachedTerminalValidation: true;
  detachedValidationReceiptsEquivalent: true;
  operatorCliStopResumeReachedReady: true;
  passed: true;
  safety: RecallGenerationRecoveryPreflightResult['sourceSafety'] & {
    disposableStorageOnly: true;
  };
}

function formatCommand(executable: string, argumentsList: readonly string[]): string {
  return [executable, ...argumentsList].join(' ');
}

/** Runs one command required by generation recovery certification. */
export async function runRecoveryCertificationCommand(options: {
  name: string;
  executable: string;
  argumentsList: readonly string[];
  projectDirectory: string;
  displayCommand?: string;
  expectedTestNamePattern?: RegExp;
}): Promise<RecoveryCertificationCommandResult> {
  const startedAt = performance.now();
  let capturedOutput = '';
  await new Promise<void>((resolvePromise, rejectPromise) => {
    const childEnvironment = { ...process.env };
    delete childEnvironment.NODE_TEST_CONTEXT;
    const captureTestOutput = options.expectedTestNamePattern !== undefined;
    const child = spawn(options.executable, options.argumentsList, {
      cwd: options.projectDirectory,
      env: childEnvironment,
      stdio: captureTestOutput ? ['ignore', 'pipe', 'pipe'] : 'inherit',
    });
    if (captureTestOutput) {
      child.stdout?.on('data', (chunk: Buffer) => {
        capturedOutput += chunk.toString('utf8');
        process.stdout.write(chunk);
      });
      child.stderr?.on('data', (chunk: Buffer) => {
        capturedOutput += chunk.toString('utf8');
        process.stderr.write(chunk);
      });
    }
    child.once('error', (error) => {
      rejectPromise(
        new Error(
          `Recall generation recovery certification command failed to start: ${options.name}`,
          {
            cause: error,
          },
        ),
      );
    });
    child.once('exit', (code, signal) => {
      if (code === 0) {
        const expectedTestNamePattern = options.expectedTestNamePattern;
        const matchingNamedTestRan =
          expectedTestNamePattern === undefined ||
          capturedOutput.split(/\r?\n/u).some((line) => {
            const testName = line.match(/^# Subtest: (.+)$/u)?.[1];
            return testName !== undefined && expectedTestNamePattern.test(testName);
          });
        if (matchingNamedTestRan) {
          resolvePromise();
          return;
        }
        rejectPromise(
          new Error(
            `Recall generation recovery certification command did not run an expected named test: ${options.name}`,
          ),
        );
        return;
      }
      rejectPromise(
        new Error(
          `Recall generation recovery certification command failed: ${options.name}; exit=${String(code)} signal=${String(signal)}`,
        ),
      );
    });
  });
  return {
    name: options.name,
    command: options.displayCommand ?? formatCommand(options.executable, options.argumentsList),
    durationMilliseconds: performance.now() - startedAt,
    passed: true,
  };
}

async function fingerprintCertificationInputs(projectDirectory: string): Promise<string> {
  const fingerprint = createHash('sha256');
  for (const relativePath of CERTIFICATION_INPUT_PATHS) {
    fingerprint.update(relativePath);
    fingerprint.update('\0');
    fingerprint.update(await readFile(join(projectDirectory, relativePath)));
    fingerprint.update('\0');
  }
  return fingerprint.digest('hex');
}

function readRequiredSlopScanBaseDirectory(): string {
  const configuredPath = process.env.PI_RECALL_SLOP_BASE_DIRECTORY;
  if (configuredPath === undefined || configuredPath.trim() === '') {
    throw new Error(
      'Recall generation recovery certification requires PI_RECALL_SLOP_BASE_DIRECTORY at the exact review base',
    );
  }
  const baseDirectory = resolve(configuredPath);
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: baseDirectory,
    encoding: 'utf8',
  }).trim();
  if (revision !== SLOP_SCAN_BASE_COMMIT) {
    throw new Error(
      `Recall generation recovery certification slop base mismatch: expected ${SLOP_SCAN_BASE_COMMIT}, received ${revision}`,
    );
  }
  const status = execFileSync('git', ['status', '--porcelain'], {
    cwd: baseDirectory,
    encoding: 'utf8',
  });
  if (status !== '') {
    throw new Error(
      `Recall generation recovery certification slop base must be clean: ${baseDirectory}`,
    );
  }
  return baseDirectory;
}

function assertProductionCardinality(preflight: RecallGenerationRecoveryPreflightResult): void {
  for (const [storeName, count] of Object.entries(preflight.interrupted.storeCounts)) {
    if (count <= PRODUCTION_FAILURE_BOUNDARY) {
      throw new Error(
        `Recall generation recovery certification ${storeName} cardinality did not cross ${PRODUCTION_FAILURE_BOUNDARY}: ${count}`,
      );
    }
  }
  if (
    preflight.uninterrupted.manifestFingerprint !== preflight.interrupted.manifestFingerprint ||
    preflight.uninterrupted.embeddingProfileId !== preflight.interrupted.embeddingProfileId ||
    preflight.uninterrupted.startingSnapshotFingerprint !==
      preflight.interrupted.startingSnapshotFingerprint ||
    JSON.stringify(preflight.uninterrupted.receipt) !==
      JSON.stringify(preflight.interrupted.receipt)
  ) {
    throw new Error(
      'Recall generation recovery certification interrupted and uninterrupted identities disagree',
    );
  }
}

function formatRecoveryCertificationReport(
  evidence: RecallGenerationRecoveryCertificationEvidence,
): string {
  const measured = evidence.preflight.interrupted;
  return `# Repaired generation recovery preflight

**Result:** PASS

- Candidate commit: \`${evidence.candidateCommit}\`
- Completed: ${evidence.completedAt}
- Generated source snapshot: \`${evidence.preflight.sourceSnapshotChecksum}\`
- Certification inputs: \`${evidence.certificationInputChecksum}\`
- Runtime: ${evidence.environment.nodeVersion} on ${evidence.environment.platform}/${evidence.environment.architecture}
- CPU: ${evidence.environment.cpuModel}
- zvec: ${evidence.environment.zvecVersion}
- Deterministic embedding profile: \`${measured.embeddingProfileId}\`
- Manifest fingerprint: \`${measured.manifestFingerprint}\`
- Starting snapshot fingerprint: \`${measured.startingSnapshotFingerprint}\`

## Production cardinality

Complete reopened validation crossed the observed ${evidence.productionFailureBoundary.toLocaleString('en-US')}-record failure boundary in every real-zvec store.

| Store | Reopened count | Membership digest |
| --- | ---: | --- |
| Lexical/source | ${measured.storeCounts.lexicalSource.toLocaleString('en-US')} | \`${measured.exactMembership.lexicalSource.digest}\` |
| Dense | ${measured.storeCounts.dense.toLocaleString('en-US')} | \`${measured.exactMembership.dense.digest}\` |
| Session projection | ${measured.storeCounts.sessionProjection.toLocaleString('en-US')} | \`${measured.exactMembership.sessionProjection.digest}\` |

The uninterrupted and twice-resumed in-process builds used the same candidate commit, generated source snapshot, generation ID, manifest, profile, and source snapshot fingerprint. Their immutable comparable validation receipts and all membership digests agree.

## Detached terminal equivalence

Matched uninterrupted and SIGKILL/resumed detached workers used generated source snapshot \`${evidence.preflight.detached.sourceSnapshotChecksum}\`. Both reached terminal succeeded/ready validation. Their compatible embedding profiles, snapshot cardinalities, validation policy, canary results, store counts, and all membership digests agree.

| Store | Uninterrupted digest | SIGKILL/resumed digest |
| --- | --- | --- |
| Lexical/source | \`${evidence.preflight.detached.uninterrupted.exactMembership.lexicalSource.digest}\` | \`${evidence.preflight.detached.interrupted.exactMembership.lexicalSource.digest}\` |
| Dense | \`${evidence.preflight.detached.uninterrupted.exactMembership.dense.digest}\` | \`${evidence.preflight.detached.interrupted.exactMembership.dense.digest}\` |
| Session projection | \`${evidence.preflight.detached.uninterrupted.exactMembership.sessionProjection.digest}\` | \`${evidence.preflight.detached.interrupted.exactMembership.sessionProjection.digest}\` |

## Recovery matrix

| Check | Result |
| --- | --- |
| Original high-cardinality fixture removed after snapshot capture | PASS |
| Original retained fixture changed after snapshot capture | PASS |
| Resume after bootstrap snapshot capture interruption | PASS |
| Resume after durable physical-source checkpoint interruption | PASS |
| Structurally malformed source skipped while later healthy source indexed | PASS |
| Injected operational failure remained fatal | PASS |
| Injected implementation failure remained fatal | PASS |
| Generated incremental append/replay/branch/deletion schedule matched fresh rebuild | PASS |
| Detached worker was interrupted, replaced, and reached terminal succeeded/ready validation | PASS |
| Standalone CLI stop/resume reached terminal succeeded/ready validation | PASS |

## Measurements

- Disposable uninterrupted generation size: ${evidence.preflight.uninterrupted.onDiskBytes.toLocaleString('en-US')} bytes
- Disposable interrupted generation size: ${evidence.preflight.interrupted.onDiskBytes.toLocaleString('en-US')} bytes
- Production-cardinality preflight duration: ${evidence.preflightDurationMilliseconds.toFixed(3)} ms

These values are reported without release thresholds.

## Reproduction

Create one clean worktree at candidate commit \`${evidence.candidateCommit}\`. Create a separate clean worktree at \`${SLOP_SCAN_BASE_COMMIT}\` for the slop-scan base, then run from the candidate worktree:

\`\`\`bash
PI_RECALL_SLOP_BASE_DIRECTORY=/path/to/clean/${SLOP_SCAN_BASE_COMMIT} ${RECOVERY_CERTIFICATION_COMMAND}
\`\`\`

The certifier ran:

${evidence.commands.map(({ command }) => `- \`${command}\``).join('\n')}

All source files and stores were generated beneath disposable temporary roots. The run did not open original Pi session JSONL, a live or existing acceptance generation, the Octen endpoint, or production activation.
`;
}

/** Certifies repaired generation recovery from one clean candidate commit. */
export async function certifyRecallGenerationRecovery(
  projectDirectory: string = process.cwd(),
): Promise<RecallGenerationRecoveryCertificationEvidence> {
  const resolvedProjectDirectory = resolve(projectDirectory);
  const candidateCommit = readCleanRecallEvaluationGitRevision(resolvedProjectDirectory);
  const slopScanBaseDirectory = readRequiredSlopScanBaseDirectory();
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-generation-recovery-evidence-'));
  const commands: RecoveryCertificationCommandResult[] = [];
  try {
    const focusedTests = [
      {
        name: 'bootstrap interruption model',
        pattern: 'replacement generation bootstrap interruption model',
        path: 'src/build-recall-fixed-snapshot-generation.test.ts',
      },
      {
        name: 'malformed versus operational classification',
        pattern:
          'malformed-source skips once|parser-looking operational failures fatal|non-source failure category fatal',
        path: 'src/recall-physical-source-generation.test.ts',
      },
      {
        name: 'incremental versus rebuild membership equivalence',
        pattern:
          'generated incremental append, replay, branch, and deletion schedules match fresh rebuild membership',
        path: 'src/transfer-incremental-recall-work-plan.test.ts',
      },
      {
        name: 'detached worker interruption and terminal resume',
        pattern: 'crashed workers at every staging phase remain resumable and idempotent',
        path: 'src/recall-background-index-conversation-service.test.ts',
      },
      {
        name: 'operator CLI stop and terminal resume',
        pattern: 'standalone rebuild stops, resumes the same snapshot, and discards inactive work',
        path: 'src/pi-session-recall-cli.test.ts',
      },
    ] as const;
    for (const focusedTest of focusedTests) {
      commands.push(
        await runRecoveryCertificationCommand({
          name: focusedTest.name,
          executable: process.execPath,
          argumentsList: [
            '--import',
            'tsx',
            '--test',
            '--test-reporter=tap',
            `--test-name-pattern=${focusedTest.pattern}`,
            focusedTest.path,
          ],
          projectDirectory: resolvedProjectDirectory,
          expectedTestNamePattern: new RegExp(focusedTest.pattern, 'u'),
        }),
      );
    }

    for (const gate of [
      { name: 'full test suite', executable: 'npm', argumentsList: ['test'] },
      { name: 'typecheck', executable: 'npm', argumentsList: ['run', 'typecheck'] },
      { name: 'type-aware lint', executable: 'npm', argumentsList: ['run', 'lint'] },
      { name: 'format check', executable: 'npm', argumentsList: ['run', 'format:check'] },
      { name: 'Git whitespace check', executable: 'git', argumentsList: ['diff', '--check'] },
    ] as const) {
      commands.push(
        await runRecoveryCertificationCommand({
          ...gate,
          projectDirectory: resolvedProjectDirectory,
        }),
      );
    }
    commands.push(
      await runRecoveryCertificationCommand({
        name: 'repository-required slop scan',
        executable: 'slop-scan',
        argumentsList: [
          'delta',
          '--base',
          slopScanBaseDirectory,
          '--head',
          resolvedProjectDirectory,
          '--fail-on',
          'added,worsened',
        ],
        projectDirectory: resolvedProjectDirectory,
        displayCommand:
          'slop-scan delta --base <exact-base-worktree> --head "$PWD" --fail-on added,worsened',
      }),
    );

    const preflightStartedAt = performance.now();
    const preflight = await runRecallGenerationRecoveryPreflight({
      disposableRoot,
      logicalSessionCount: PRODUCTION_LOGICAL_SESSION_COUNT,
    });
    const preflightDurationMilliseconds = performance.now() - preflightStartedAt;
    assertProductionCardinality(preflight);
    commands.push({
      name: 'public service and real-zvec production-cardinality recovery preflight',
      command: RECOVERY_CERTIFICATION_COMMAND,
      durationMilliseconds: preflightDurationMilliseconds,
      passed: true,
    });

    const evidence: RecallGenerationRecoveryCertificationEvidence = {
      version: 1,
      candidateCommit,
      completedAt: new Date().toISOString(),
      environment: createRecallDiagnosticHostIdentity(),
      certificationInputChecksum: await fingerprintCertificationInputs(resolvedProjectDirectory),
      commands,
      preflightDurationMilliseconds,
      productionFailureBoundary: PRODUCTION_FAILURE_BOUNDARY,
      preflight,
      incrementalVsRebuildMembershipEquivalent: true,
      detachedWorkerReachedTerminalValidation:
        preflight.detached.uninterruptedWorkerReachedTerminalValidation &&
        preflight.detached.resumedWorkerReachedTerminalValidation,
      detachedValidationReceiptsEquivalent: preflight.detached.validationReceiptsEquivalent,
      operatorCliStopResumeReachedReady: true,
      passed: true,
      safety: { ...preflight.sourceSafety, disposableStorageOnly: true },
    };

    assertRecallEvaluationGitRevisionCurrent(resolvedProjectDirectory, candidateCommit);
    const resultsPath = join(resolvedProjectDirectory, RECOVERY_CERTIFICATION_RESULTS_PATH);
    const reportPath = join(resolvedProjectDirectory, RECOVERY_CERTIFICATION_REPORT_PATH);
    await writeAtomicRecallEvaluationFile(resultsPath, `${JSON.stringify(evidence, null, 2)}\n`);
    await writeAtomicRecallEvaluationFile(reportPath, formatRecoveryCertificationReport(evidence));
    execFileSync('npx', ['--no-install', 'oxfmt', resultsPath, reportPath], {
      cwd: resolvedProjectDirectory,
      stdio: 'pipe',
    });
    process.stdout.write(
      `Recall generation recovery certification PASS\nReport: ${RECOVERY_CERTIFICATION_REPORT_PATH}\n`,
    );
    return evidence;
  } finally {
    await rm(disposableRoot, { recursive: true, force: true });
  }
}

if (
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))
) {
  await certifyRecallGenerationRecovery();
}
