import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readdir, readlink, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { loadRecallQualityCorpus } from './recall-quality-corpus.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
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
  await writeFile(join(corpusDirectory, sessionFileName), sessionContent);
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
  const baseConfig: RecallConversationConfig = {
    sessionsDirectory: join(directory, 'must-not-scan-production-sessions'),
    databasePath: join(directory, 'unused-zvec'),
    catalogPath: join(directory, 'unused-catalog.sqlite'),
    statePath: join(directory, 'unused-state.json'),
    manifestPath: join(directory, 'unused-manifest.json'),
    indexMaintenanceStatusPath: join(directory, 'unused-maintenance-status.json'),
    physicalSessionIgnoreStatePath: join(directory, 'unused-physical-session-ignore.json'),
    tokenizerCacheDirectory: join(directory, 'unused-tokenizers'),
    lockPath: join(directory, 'unused.lock'),
    databaseGenerationRootPath: join(directory, 'unused-generations'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-embedding',
    embeddingServedModelId: 'test-embedding-served',
    embeddingNativeDimensions: 1_024,
    embeddingStoredDimensions: 1_024,
    embeddingBatchSize: 8,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, invocation: 8 },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
  const baseGenerationTarget = 'unused-generations/existing-generation';
  await mkdir(join(directory, baseGenerationTarget), { recursive: true });
  await symlink(baseGenerationTarget, join(directory, 'active'), 'dir');

  const testEmbedding = [1, ...Array.from({ length: 1_023 }, () => 0)];
  const embeddingProvider: RecallEmbeddingProvider = {
    async embedQuery() {
      return testEmbedding;
    },
    async embedDocuments(documents) {
      return documents.map(() => testEmbedding);
    },
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
      embeddingProvider,
      async loadTokenizer() {
        return tokenizer;
      },
    },
  });

  assert.equal(result.version, 6);
  assert.equal(result.boundedWork.indexRuns, 1);
  assert.equal(result.boundedWork.executedSearchRequests, 1);
  assert.equal(result.boundedWork.repositoryIdentityResolutions, 1);
  assert.deepEqual(result.evaluationIdentity, {
    defaultScope: RecallSearchScope.PROJECT,
    projectScopePolicyVersion: 1,
    projectIdentityPolicyVersion: 4,
    projectIdentityMetadataSchemaVersion: 3,
    lineagePolicyVersion: 1,
    lineageDigest: '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
    rankingMode: 'compact',
    mixedResultPolicyVersion: 1,
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 8, invocation: 8 },
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
  assert.equal(await readlink(join(directory, 'active')), baseGenerationTarget);
  assert.deepEqual(await readdir(join(directory, baseGenerationTarget)), []);

  await assert.rejects(
    () =>
      runRecallQualityEvaluation({
        corpus,
        baseConfig,
        workDirectory: join(directory, 'recall-quality-evaluation'),
        dependencies: {
          embeddingProvider,
          async loadTokenizer() {
            return tokenizer;
          },
        },
      }),
    /work directory must stay inside evaluation data area/,
  );
});
