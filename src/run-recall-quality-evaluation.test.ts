import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  RecallDiagnosticsMode,
  RecallEvidenceRelation,
  RecallProjectIdentitySource,
  RecallSearchScope,
} from './enums.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { loadRecallQualityCorpus } from './recall-quality-corpus.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import {
  readRecallIndexManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  type RecallTokenizerManifestIdentity,
} from './recall-index-manifest.js';
import type { RecallEmbeddingModelProfile } from './recall-model-profiles.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import { runRecallQualityEvaluation } from './run-recall-quality-evaluation.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

void test('recall quality runner indexes and searches only the bounded declared corpus', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'run-recall-quality-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const evaluationDirectory = join(directory, 'evaluation');
  const corpusDirectory = join(evaluationDirectory, 'corpus');
  await mkdir(corpusDirectory, { recursive: true });
  const sessionContent =
    [
      {
        type: 'session',
        version: 3,
        id: 'bounded-session',
        timestamp: '2026-07-20T10:00:00Z',
        cwd: '/bounded',
      },
      {
        type: 'message',
        id: 'bounded-answer',
        parentId: null,
        timestamp: '2026-07-20T10:01:00Z',
        message: {
          role: 'assistant',
          content: 'The bounded answer is quartz-heron and stays inside this fixture.',
        },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n';
  const sessionFileName = 'bounded.jsonl';
  const sessionPath = join(corpusDirectory, sessionFileName);
  await writeFile(sessionPath, sessionContent);
  const sourceBefore = {
    bytes: await readFile(sessionPath),
    metadata: await stat(sessionPath),
  };
  const sha256 = createHash('sha256').update(sessionContent).digest('hex');
  const specification = {
    version: 3,
    corpus: {
      id: 'bounded-runner-v1',
      sessionDirectory: 'corpus',
      sessionFiles: [{ fileName: sessionFileName, sha256 }],
    },
    projectIdentityFixtures: [
      {
        workingDirectory: '/bounded',
        projectIdentity: 'git-origin:example.test/acme/bounded',
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
        origin: 'https://example.test/acme/bounded.git',
      },
    ],
    projectLineages: {},
    bounds: {
      maximumSessionFiles: 1,
      maximumEvaluationCases: 1,
      maximumChunkPolicies: 1,
      maximumCandidateCounts: 1,
      maximumFinalCounts: 1,
      maximumSearchRequests: 3,
      maximumChunkEmbeddingRequests: 3,
    },
    chunkPolicies: [{ id: '512-64', maxTokens: 512, overlapTokens: 64 }],
    candidateCounts: [8],
    finalCounts: [5],
    warmupQueriesPerCombination: 0,
    qualityGate: {
      minimumCandidatePoolRecall: 1,
      minimumFinalRecall: 1,
      minimumContextUsefulness: 1,
      minimumSourceOccurrencePreservation: 1,
      maximumFinalDuplicateRate: 0,
      maximumQueryP95Milliseconds: 10_000,
    },
    cases: [
      {
        id: 'bounded-answer',
        category: 'exact_identifier',
        query: 'quartz-heron',
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/bounded',
        expectedInvocationProjectIdentity: 'git-origin:example.test/acme/bounded',
        expectedSources: [
          {
            sessionFile: sessionFileName,
            entryId: 'bounded-answer',
            requiredText: ['quartz-heron'],
            expectedSessionOrigin: '/bounded',
            expectedEvidenceRelation: RecallEvidenceRelation.SAME_REPOSITORY,
            requiredContributingEntryIds: ['bounded-answer'],
            expectedBranch: 'active',
          },
        ],
        excludedSessionFiles: [],
        requiredContext: ['quartz-heron'],
        minimumPreservedSourceOccurrences: 1,
      },
    ],
  };
  const specificationPath = join(evaluationDirectory, 'recall-quality-cases.json');
  await writeFile(specificationPath, `${JSON.stringify(specification, null, 2)}\n`);
  const corpus = await loadRecallQualityCorpus(specificationPath);
  const protectedDataDirectory = join(directory, 'must-not-touch-production-recall');
  const baseConfig: RecallConversationConfig = {
    sessionsDirectory: join(directory, 'must-not-scan-production-sessions'),
    dataDirectory: protectedDataDirectory,
    databasePath: join(protectedDataDirectory, 'zvec'),
    projectionDatabasePath: join(protectedDataDirectory, 'session-projections'),
    statePath: join(protectedDataDirectory, 'index-state.json'),
    manifestPath: join(protectedDataDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(protectedDataDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(protectedDataDirectory, 'embedding-cache'),
    lockPath: join(protectedDataDirectory, 'operation.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(protectedDataDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(protectedDataDirectory, 'diagnostics.previous.jsonl'),
    markerSpoolDirectory: join(protectedDataDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: join(protectedDataDirectory, 'markers', 'quarantine'),
    markerControlDirectory: join(protectedDataDirectory, 'markers', 'control'),
    workerOwnershipLockPath: join(protectedDataDirectory, 'incremental-worker.lock'),
    generationRootDirectory: join(protectedDataDirectory, 'generations'),
    activeGenerationPointerPath: join(protectedDataDirectory, 'active-generation.json'),
    generationRegistryPath: join(protectedDataDirectory, 'generation-registry.json'),
    backlogSummaryPath: join(protectedDataDirectory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(protectedDataDirectory, 'incremental-diagnostics.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-embedding',
    embeddingServedModelId: 'test-embedding-served',
    embeddingArtifact: 'test-embedding.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'test-reranker',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
  };
  const canaryAwareEmbeddingProvider: RecallEmbeddingProvider = {
    async embedQuery(query, signal) {
      const vectors = await this.embedDocuments([query], signal);
      const vector = vectors[0];
      if (!vector) {
        throw new Error('missing query vector');
      }
      return vector;
    },
    async embedDocuments(texts) {
      return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
    },
  };
  const embeddingProfile: RecallEmbeddingModelProfile = {
    identity: {
      requestModel: 'bounded-profile',
      servedModelId: 'bounded-profile-served',
      artifact: 'bounded-profile.gguf',
      artifactRepository: 'example.test/bounded-profile',
      artifactRevision: 'bounded-revision',
      artifactSha256: 'a'.repeat(64),
      dimensions: 3,
      quantization: 'fixture',
      pooling: 'mean',
      normalization: 'l2',
    },
    queryInputPrefix: 'query: ',
    documentInputPrefix: 'document: ',
    canary: {
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'bounded canary',
      expectedDimensions: 3,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    },
  };
  const embeddingProvider: RecallEmbeddingProvider = {
    async embedQuery() {
      return [0, 0, 1];
    },
    async embedDocuments(texts) {
      return texts.map(() => [1, 0, 0]);
    },
  };
  const tokenizerIdentity: RecallTokenizerManifestIdentity = {
    model: 'bounded-tokenizer',
    revision: 'bounded-tokenizer-revision',
    library: { name: 'bounded-tokenizer-library', version: '1.0.0' },
    encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
    assets: [{ fileName: 'bounded-tokenizer.gguf', sha256: 'b'.repeat(64) }],
  };
  const tokenizer: ConversationTextTokenizer = {
    encodeConversationText(text) {
      return {
        ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()),
      };
    },
  };

  const result = await runRecallQualityEvaluation({
    corpus,
    baseConfig,
    workDirectory: join(evaluationDirectory, '.recall-data', 'recall-quality-evaluation'),
    dependencies: {
      embeddingProvider: canaryAwareEmbeddingProvider,
      async loadTokenizer() {
        return tokenizer;
      },
    },
  });

  assert.equal(result.version, 5);
  assert.deepEqual(result.storageIdentity, {
    conversationSchemaVersion: 9,
    zvecSchemaVersion: 8,
    indexManifestVersion: 6,
    incrementalEligibilityPolicyVersion: 1,
  });
  assert.equal(result.boundedWork.indexRuns, 1);
  assert.equal(result.boundedWork.executedSearchRequests, 1);
  assert.equal(result.boundedWork.rerankerRequests, 0);
  assert.equal(result.boundedWork.repositoryIdentityResolutions, 1);
  assert.deepEqual(result.evaluationIdentity, {
    defaultScope: RecallSearchScope.PROJECT,
    projectScopePolicyVersion: 1,
    projectIdentityPolicyVersion: 4,
    projectIdentityMetadataSchemaVersion: 3,
    lineagePolicyVersion: 1,
    lineageDigest: '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    rankingMode: 'hybrid',
    rankFusionVersion: 2,
    reciprocalRankConstant: 60,
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    finalResultCount: 5,
  });
  assert.equal(result.indexRuns.length, 1);
  assert.ok(result.indexRuns.every(({ indexSummary }) => indexSummary.scannedSessions === 1));
  assert.equal(result.configurations.length, 1);
  assert.equal(result.selection.passed, true);
  assert.equal(result.selection.selected?.candidateCount, 8);
  assert.equal(result.selection.selected?.finalCount, 5);
  assert.deepEqual(result.configurations[0]?.measurement.policyFailureCaseIds, []);
  assert.deepEqual(result.configurations[0]?.measurement.queryLatencyByScope.global, null);
  assert.ok(result.configurations[0]?.measurement.queryLatencyByScope.project);

  const profileWorkDirectory = join(
    evaluationDirectory,
    '.recall-data',
    'profile-aware',
    'recall-quality-evaluation',
  );
  const profileResult = await runRecallQualityEvaluation({
    corpus,
    baseConfig,
    workDirectory: profileWorkDirectory,
    dependencies: {
      embeddingProfile,
      embeddingProvider,
      tokenizerIdentity,
      async loadTokenizer() {
        return tokenizer;
      },
    },
  });
  const profileManifest = await readRecallIndexManifest(
    join(
      profileWorkDirectory,
      '512-64',
      'generations',
      'generation_quality_active',
      'index-manifest.json',
    ),
  );
  assert.equal(profileResult.selection.passed, true);
  assert.equal(profileManifest?.embedding.requestModel, 'bounded-profile');
  assert.equal(profileManifest?.embedding.dimensions, 3);
  assert.deepEqual(profileManifest?.tokenizer, tokenizerIdentity);

  await assert.rejects(
    () =>
      runRecallQualityEvaluation({
        corpus,
        baseConfig,
        workDirectory: join(directory, 'recall-quality-evaluation'),
        dependencies: {
          embeddingProfile,
          embeddingProvider,
          tokenizerIdentity,
          async loadTokenizer() {
            return tokenizer;
          },
        },
      }),
    /work directory must stay inside evaluation data area/,
  );

  const sourceAfter = {
    bytes: await readFile(sessionPath),
    metadata: await stat(sessionPath),
  };
  assert.deepEqual(sourceAfter.bytes, sourceBefore.bytes);
  assert.deepEqual(
    {
      size: sourceAfter.metadata.size,
      mode: sourceAfter.metadata.mode,
      mtimeMs: sourceAfter.metadata.mtimeMs,
      ino: sourceAfter.metadata.ino,
    },
    {
      size: sourceBefore.metadata.size,
      mode: sourceBefore.metadata.mode,
      mtimeMs: sourceBefore.metadata.mtimeMs,
      ino: sourceBefore.metadata.ino,
    },
  );
  await rm(directory, { recursive: true, force: true });
  await assert.rejects(() => access(directory), /ENOENT/u);
});
