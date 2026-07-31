import { createHash } from 'node:crypto';
import { open } from 'node:fs/promises';
import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { isAbsolute, join, relative, resolve } from 'node:path';

import {
  loadRecallConversationConfig,
  type RecallConversationConfig,
} from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationDependencies,
} from './recall-conversation-service.js';
import type { OpenedValidatedRecallGeneration } from './recall-coherent-generation.js';
import { readRecallGenerationManifest } from './recall-generation-manifest.js';
import {
  readRecallGenerationValidationReceipt,
  type RecallGenerationValidationReceipt,
} from './recall-generation-validation-receipt.js';

const RECOVERY_PREFLIGHT_GENERATION_ID = 'generation_recovery_preflight';
const CARDINALITY_LOGICAL_SESSIONS_PER_SOURCE = 1_000;
const MALFORMED_SOURCE_FILE_NAME = 'a-malformed.jsonl';
const RETAINED_SOURCE_FILE_NAME = 'b-retained.jsonl';
const ORIGINAL_RETAINED_EVIDENCE = 'fixed snapshot retained evidence';
const CHANGED_RETAINED_EVIDENCE = 'changed source must stay excluded';

interface RecallGenerationRecoveryPreflightOptions {
  disposableRoot: string;
  logicalSessionCount: number;
}

interface ComparableRecallGenerationValidationReceipt {
  manifestFingerprint: string;
  startingSnapshot: RecallGenerationValidationReceipt['startingSnapshot'];
  exactMembership: RecallGenerationValidationReceipt['exactMembership'];
  validationPolicyVersion: RecallGenerationValidationReceipt['validationPolicyVersion'];
  canaryResults: RecallGenerationValidationReceipt['canaryResults'];
}

interface MeasuredRecallGeneration {
  manifestFingerprint: string;
  embeddingProfileId: string;
  startingSnapshotFingerprint: string;
  storeCounts: OpenedValidatedRecallGeneration['storeCounts'];
  exactMembership: RecallGenerationValidationReceipt['exactMembership'];
  receipt: ComparableRecallGenerationValidationReceipt;
  onDiskBytes: number;
}

/** Measured disposable evidence returned by the production-cardinality recovery preflight. */
export interface RecallGenerationRecoveryPreflightResult {
  logicalSessionCount: number;
  sourceSnapshotChecksum: string;
  uninterrupted: MeasuredRecallGeneration;
  interrupted: MeasuredRecallGeneration;
  fixedSnapshot: {
    originalCardinalitySourceRemoved: true;
    originalRetainedSourceChanged: true;
    retainedOriginalEvidenceFound: true;
    changedReplacementEvidenceFound: false;
  };
  interruptions: {
    bootstrapSnapshotCapture: 'resumed';
    physicalSourceCheckpoint: 'resumed';
  };
  failureClassification: {
    malformedSourceSkipped: true;
    operationalFailureFatal: true;
    implementationFailureFatal: true;
  };
  sourceSafety: {
    originalPiSessionFilesAccessed: false;
    liveRecallGenerationAccessed: false;
    productionActivationPerformed: false;
    octenEndpointAccessed: false;
  };
}

function assertDisposablePreflightRoot(disposableRoot: string): string {
  const resolvedRoot = resolve(disposableRoot);
  const temporaryDirectory = resolve(tmpdir());
  const relativePath = relative(temporaryDirectory, resolvedRoot);
  if (relativePath === '' || relativePath.startsWith('..') || isAbsolute(relativePath)) {
    throw new Error(
      `Recall generation recovery preflight root must be a child of ${temporaryDirectory}: ${resolvedRoot}`,
    );
  }
  return resolvedRoot;
}

function createDeterministicRecallDependencies(
  overrides: Partial<RecallConversationDependencies> = {},
): RecallConversationDependencies {
  return {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return {
            ids: text
              .split(/\s+/u)
              .filter(Boolean)
              .map((_, index) => index),
          };
        },
      };
    },
    async resolveProjectIdentity() {
      return null;
    },
    workerSignal: { signalDetachedWorker() {} },
    ...overrides,
  };
}

async function createPreflightConfig(
  disposableRoot: string,
  dataDirectoryName: string,
  sessionsDirectory: string,
): Promise<RecallConversationConfig> {
  return loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(disposableRoot, dataDirectoryName),
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
}

async function writeCardinalitySource(
  path: string,
  startOrdinal: number,
  logicalSessionCount: number,
): Promise<void> {
  const file = await open(path, 'wx');
  try {
    const lines: string[] = [];
    for (let offset = 0; offset < logicalSessionCount; offset += 1) {
      const suffix = String(startOrdinal + offset).padStart(6, '0');
      lines.push(
        JSON.stringify({
          type: 'session',
          version: 3,
          id: `cardinality-session-${suffix}`,
          timestamp: '2026-08-20T00:00:00.000Z',
          cwd: '/generated/recovery-preflight',
        }),
        JSON.stringify({
          type: 'message',
          id: `cardinality-entry-${suffix}`,
          parentId: null,
          timestamp: '2026-08-20T00:00:01.000Z',
          message: {
            role: 'assistant',
            content: `production cardinality evidence ${suffix}`,
          },
        }),
      );
    }
    await file.write(`${lines.join('\n')}\n`);
  } finally {
    await file.close();
  }
}

async function writeCardinalitySources(
  sessionsDirectory: string,
  logicalSessionCount: number,
): Promise<string[]> {
  const paths: string[] = [];
  for (
    let startOrdinal = 0;
    startOrdinal < logicalSessionCount;
    startOrdinal += CARDINALITY_LOGICAL_SESSIONS_PER_SOURCE
  ) {
    const sourceOrdinal = String(paths.length).padStart(3, '0');
    const path = join(sessionsDirectory, `a-cardinality-${sourceOrdinal}.jsonl`);
    await writeCardinalitySource(
      path,
      startOrdinal,
      Math.min(CARDINALITY_LOGICAL_SESSIONS_PER_SOURCE, logicalSessionCount - startOrdinal),
    );
    paths.push(path);
  }
  return paths;
}

async function writeRetainedSource(path: string, content: string): Promise<void> {
  await writeFile(
    path,
    `${JSON.stringify({
      type: 'session',
      version: 3,
      id: 'retained-session',
      timestamp: '2026-08-20T01:00:00.000Z',
      cwd: '/generated/recovery-preflight',
    })}\n${JSON.stringify({
      type: 'message',
      id: 'retained-entry',
      parentId: null,
      timestamp: '2026-08-20T01:00:01.000Z',
      message: { role: 'assistant', content },
    })}\n`,
    'utf8',
  );
}

async function calculateGeneratedSourceSnapshotChecksum(
  sessionsDirectory: string,
  cardinalitySourcePaths: readonly string[],
): Promise<string> {
  const checksum = createHash('sha256');
  const sourcePaths = [
    ...cardinalitySourcePaths,
    join(sessionsDirectory, RETAINED_SOURCE_FILE_NAME),
  ];
  for (const sourcePath of sourcePaths) {
    checksum.update(relative(sessionsDirectory, sourcePath));
    checksum.update('\0');
    checksum.update(await readFile(sourcePath));
    checksum.update('\0');
  }
  return checksum.digest('hex');
}

async function calculateDirectoryByteSize(directory: string): Promise<number> {
  let byteSize = 0;
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      byteSize += await calculateDirectoryByteSize(path);
    } else if (entry.isFile()) {
      byteSize += (await stat(path)).size;
    }
  }
  return byteSize;
}

function comparableReceipt(
  receipt: RecallGenerationValidationReceipt,
): ComparableRecallGenerationValidationReceipt {
  return {
    manifestFingerprint: receipt.manifestFingerprint,
    startingSnapshot: receipt.startingSnapshot,
    exactMembership: receipt.exactMembership,
    validationPolicyVersion: receipt.validationPolicyVersion,
    canaryResults: receipt.canaryResults,
  };
}

async function measureValidatedGeneration(
  service: ReturnType<typeof createRecallConversationService>,
  generation: OpenedValidatedRecallGeneration,
): Promise<MeasuredRecallGeneration> {
  const reopened = await service.openValidatedRecallGeneration(generation.generationId);
  const [receipt, { manifest }] = await Promise.all([
    readRecallGenerationValidationReceipt(reopened.validationReceiptPath),
    readRecallGenerationManifest(reopened.manifestPath),
  ]);
  return {
    manifestFingerprint: reopened.manifestFingerprint,
    embeddingProfileId: manifest.embeddingProfile.profileId,
    startingSnapshotFingerprint: reopened.startingSnapshotFingerprint,
    storeCounts: reopened.storeCounts,
    exactMembership: receipt.exactMembership,
    receipt: comparableReceipt(receipt),
    onDiskBytes: await calculateDirectoryByteSize(reopened.generationDirectory),
  };
}

function assertEquivalentValidatedGenerations(
  uninterrupted: MeasuredRecallGeneration,
  interrupted: MeasuredRecallGeneration,
): void {
  if (JSON.stringify(uninterrupted.receipt) !== JSON.stringify(interrupted.receipt)) {
    throw new Error(
      `Recall generation recovery preflight receipt mismatch between uninterrupted and interrupted builds: ${JSON.stringify({ uninterrupted: uninterrupted.receipt, interrupted: interrupted.receipt })}`,
    );
  }
  if (JSON.stringify(uninterrupted.storeCounts) !== JSON.stringify(interrupted.storeCounts)) {
    throw new Error(
      'Recall generation recovery preflight store-count mismatch between uninterrupted and interrupted builds',
    );
  }
}

async function assertFatalNonSourceFailure(options: {
  disposableRoot: string;
  sessionsDirectory: string;
  dataDirectoryName: string;
  generationId: string;
  dependencies: RecallConversationDependencies;
  expectedMessage: RegExp;
}): Promise<void> {
  const config = await createPreflightConfig(
    options.disposableRoot,
    options.dataDirectoryName,
    options.sessionsDirectory,
  );
  const service = createRecallConversationService(config, options.dependencies);
  let receivedExpectedFailure = false;
  try {
    await service.buildReplacementRecallGeneration({ generationId: options.generationId });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    receivedExpectedFailure = options.expectedMessage.test(message);
  }
  if (!receivedExpectedFailure) {
    throw new Error(
      `Recall generation recovery preflight non-source failure was not fatal: ${options.generationId}`,
    );
  }
}

/** Runs fixed-snapshot and exact-membership recovery against generated sources and real zvec. */
export async function runRecallGenerationRecoveryPreflight(
  options: Readonly<RecallGenerationRecoveryPreflightOptions>,
): Promise<RecallGenerationRecoveryPreflightResult> {
  if (!Number.isSafeInteger(options.logicalSessionCount) || options.logicalSessionCount < 1) {
    throw new Error(
      `Recall generation recovery preflight logical session count invalid: ${options.logicalSessionCount}`,
    );
  }
  const disposableRoot = assertDisposablePreflightRoot(options.disposableRoot);
  const sessionsDirectory = join(disposableRoot, 'generated-sessions');
  await mkdir(sessionsDirectory, { recursive: true });
  const retainedSourcePath = join(sessionsDirectory, RETAINED_SOURCE_FILE_NAME);
  const cardinalitySourcePaths = await writeCardinalitySources(
    sessionsDirectory,
    options.logicalSessionCount,
  );
  await writeRetainedSource(retainedSourcePath, ORIGINAL_RETAINED_EVIDENCE);
  const sourceSnapshotChecksum = await calculateGeneratedSourceSnapshotChecksum(
    sessionsDirectory,
    cardinalitySourcePaths,
  );

  const uninterruptedConfig = await createPreflightConfig(
    disposableRoot,
    'uninterrupted-data',
    sessionsDirectory,
  );
  const uninterruptedService = createRecallConversationService(
    uninterruptedConfig,
    createDeterministicRecallDependencies(),
  );
  const uninterruptedGeneration = await uninterruptedService.buildReplacementRecallGeneration({
    generationId: RECOVERY_PREFLIGHT_GENERATION_ID,
  });
  const uninterrupted = await measureValidatedGeneration(
    uninterruptedService,
    uninterruptedGeneration,
  );

  let interruptSnapshotCapture = true;
  const interruptedConfig = await createPreflightConfig(
    disposableRoot,
    'interrupted-data',
    sessionsDirectory,
  );
  const interruptedService = createRecallConversationService(
    interruptedConfig,
    createDeterministicRecallDependencies({
      fixedSnapshotBuildFault(stage) {
        if (stage === 'after-snapshot-capture' && interruptSnapshotCapture) {
          interruptSnapshotCapture = false;
          throw new Error('fixture bootstrap snapshot capture interruption');
        }
      },
    }),
  );
  try {
    await interruptedService.buildReplacementRecallGeneration({
      generationId: RECOVERY_PREFLIGHT_GENERATION_ID,
    });
    throw new Error('Recall generation recovery preflight bootstrap interruption did not occur');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.includes('fixture bootstrap snapshot capture interruption')) {
      throw error;
    }
  }

  await Promise.all([
    ...cardinalitySourcePaths.map((path) => rm(path)),
    writeRetainedSource(retainedSourcePath, CHANGED_RETAINED_EVIDENCE),
  ]);
  let interruptPhysicalCheckpoint = true;
  try {
    await interruptedService.buildReplacementRecallGeneration({
      generationId: RECOVERY_PREFLIGHT_GENERATION_ID,
      resumeExistingGeneration: true,
      onPhysicalSourceCheckpoint() {
        if (interruptPhysicalCheckpoint) {
          interruptPhysicalCheckpoint = false;
          throw new Error('fixture physical source checkpoint interruption');
        }
      },
    });
    throw new Error(
      'Recall generation recovery preflight physical checkpoint interruption missing',
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.includes('fixture physical source checkpoint interruption')) {
      throw error;
    }
  }

  const interruptedGeneration = await interruptedService.buildReplacementRecallGeneration({
    generationId: RECOVERY_PREFLIGHT_GENERATION_ID,
    resumeExistingGeneration: true,
  });
  const interrupted = await measureValidatedGeneration(interruptedService, interruptedGeneration);
  assertEquivalentValidatedGenerations(uninterrupted, interrupted);

  const retainedOriginalEvidenceFound =
    (
      await interruptedService.searchRecallGenerationLexical(
        RECOVERY_PREFLIGHT_GENERATION_ID,
        ORIGINAL_RETAINED_EVIDENCE,
        5,
      )
    ).length > 0;
  const changedReplacementEvidenceFound =
    (
      await interruptedService.searchRecallGenerationLexical(
        RECOVERY_PREFLIGHT_GENERATION_ID,
        CHANGED_RETAINED_EVIDENCE,
        5,
      )
    ).length > 0;
  if (!retainedOriginalEvidenceFound || changedReplacementEvidenceFound) {
    throw new Error('Recall generation recovery preflight reopened changed original source bytes');
  }

  const malformedSessionsDirectory = join(disposableRoot, 'malformed-sessions');
  await mkdir(malformedSessionsDirectory);
  await Promise.all([
    writeFile(
      join(malformedSessionsDirectory, MALFORMED_SOURCE_FILE_NAME),
      '{"type":"session","version":3\n',
      'utf8',
    ),
    writeRetainedSource(
      join(malformedSessionsDirectory, 'b-healthy-after-malformed.jsonl'),
      'healthy evidence after malformed source',
    ),
  ]);
  const malformedConfig = await createPreflightConfig(
    disposableRoot,
    'malformed-classification-data',
    malformedSessionsDirectory,
  );
  const malformedService = createRecallConversationService(
    malformedConfig,
    createDeterministicRecallDependencies(),
  );
  const malformedGeneration = await malformedService.buildReplacementRecallGeneration({
    generationId: 'generation_malformed_classification',
  });
  await malformedService.openValidatedRecallGeneration(malformedGeneration.generationId);
  if (
    (
      await malformedService.searchRecallGenerationLexical(
        malformedGeneration.generationId,
        'healthy evidence after malformed source',
        5,
      )
    ).length === 0
  ) {
    throw new Error('Recall generation recovery preflight did not continue after malformed source');
  }

  const failureSessionsDirectory = join(disposableRoot, 'failure-sessions');
  await mkdir(failureSessionsDirectory);
  await writeRetainedSource(
    join(failureSessionsDirectory, 'failure.jsonl'),
    'failure classification evidence',
  );
  await assertFatalNonSourceFailure({
    disposableRoot,
    sessionsDirectory: failureSessionsDirectory,
    dataDirectoryName: 'operational-failure-data',
    generationId: 'generation_operational_failure',
    dependencies: createDeterministicRecallDependencies({
      embeddingProvider: {
        async embedQuery() {
          return [1, 0, 0];
        },
        async embedDocuments() {
          throw new Error('generated operational invalid-session-source lookalike');
        },
      },
    }),
    expectedMessage: /generated operational invalid-session-source lookalike/u,
  });
  await assertFatalNonSourceFailure({
    disposableRoot,
    sessionsDirectory: failureSessionsDirectory,
    dataDirectoryName: 'implementation-failure-data',
    generationId: 'generation_implementation_failure',
    dependencies: createDeterministicRecallDependencies({
      async resolveProjectIdentity() {
        throw new Error('generated implementation source classification failure');
      },
    }),
    expectedMessage: /generated implementation source classification failure/u,
  });

  return {
    logicalSessionCount: options.logicalSessionCount,
    sourceSnapshotChecksum,
    uninterrupted,
    interrupted,
    fixedSnapshot: {
      originalCardinalitySourceRemoved: true,
      originalRetainedSourceChanged: true,
      retainedOriginalEvidenceFound: true,
      changedReplacementEvidenceFound: false,
    },
    interruptions: {
      bootstrapSnapshotCapture: 'resumed',
      physicalSourceCheckpoint: 'resumed',
    },
    failureClassification: {
      malformedSourceSkipped: true,
      operationalFailureFatal: true,
      implementationFailureFatal: true,
    },
    sourceSafety: {
      originalPiSessionFilesAccessed: false,
      liveRecallGenerationAccessed: false,
      productionActivationPerformed: false,
      octenEndpointAccessed: false,
    },
  };
}
