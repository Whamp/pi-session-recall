import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { chmod, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallCaseCategory,
  QueryPlannedRecallControlKind,
  RecallDiagnosticsMode,
} from './enums.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import { RECALL_EMBEDDING_CANARY_TEXT } from './recall-index-manifest.js';
import {
  createPublishableQueryPlannedRecallBaselineEvidence,
  createPublishableQueryPlannedRecallControls,
  formatPublishableQueryPlannedRecallBaselineReport,
  loadPrivateQueryPlannedRecallCorpus,
  runPrivateQueryPlannedRecallBaseline,
} from './query-planned-recall-baseline.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createSha256(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}

void test('private query-planned recall corpus stays checksum-fixed and publishes no private values', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'query-planned-recall-baseline-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const privateDirectory = join(directory, '.recall-data', 'query-planned-recall');
  const snapshotDirectory = join(privateDirectory, 'snapshots');
  await mkdir(snapshotDirectory, { recursive: true, mode: 0o700 });
  await chmod(join(directory, '.recall-data'), 0o700);
  await chmod(privateDirectory, 0o700);
  await chmod(snapshotDirectory, 0o700);

  const snapshotContent =
    [
      {
        type: 'session',
        version: 3,
        id: 'private-session',
        timestamp: '2026-08-01T10:00:00.000Z',
        cwd: '/private/project',
      },
      {
        type: 'message',
        id: 'expected-entry',
        parentId: null,
        timestamp: '2026-08-01T10:01:00.000Z',
        message: {
          role: 'assistant',
          content: 'Private mechanism phrase and outcome token.',
        },
      },
      {
        type: 'message',
        id: 'distractor-entry',
        parentId: 'expected-entry',
        timestamp: '2026-08-01T10:02:00.000Z',
        message: { role: 'assistant', content: 'Private distractor phrase.' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n';
  const snapshotPath = join(snapshotDirectory, 'snapshot-001.jsonl');
  await writeFile(snapshotPath, snapshotContent, { mode: 0o600 });
  const query = 'private symptom wording';
  const privateCase = {
    id: 'case-001',
    category: QueryPlannedRecallCaseCategory.SYMPTOM_TO_MECHANISM,
    controlKind: QueryPlannedRecallControlKind.DIFFICULT_CASE,
    query,
    querySha256: createSha256(query),
    scope: 'global',
    expectedSources: [
      {
        snapshotId: 'snapshot-001',
        entryId: 'expected-entry',
        requiredText: ['Private mechanism phrase'],
        expectedSessionOrigin: '/private/project',
        expectedEvidenceRelation: 'unrestricted_global_evidence',
        expectedEvidenceKind: 'conversation',
        expectedBranch: 'active',
      },
    ],
    relevantDistractors: [{ snapshotId: 'snapshot-001', entryId: 'distractor-entry' }],
    plannedRetrievalLists: { lexical: 1, semantic: 1, hypotheticalAnswer: 0 },
    retrievalWorkMatchedCandidateLimits: { dense: 34, lexical: 33, identifier: 33 },
  };
  const manifest = {
    version: 1,
    corpus: {
      id: 'private-query-planned-recall-v1',
      snapshotDirectory: 'snapshots',
      snapshots: [
        {
          id: 'snapshot-001',
          fileName: 'snapshot-001.jsonl',
          sha256: createSha256(snapshotContent),
        },
      ],
    },
    policy: {
      normalCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      plannedCandidateLimit: 20,
      finalResultLimit: 5,
    },
    cases: [privateCase],
  };
  const manifestPath = join(privateDirectory, 'manifest.json');
  const manifestContent = `${JSON.stringify(manifest, null, 2)}\n`;
  await writeFile(manifestPath, manifestContent, { mode: 0o600 });

  const loaded = await loadPrivateQueryPlannedRecallCorpus(manifestPath);
  const controls = createPublishableQueryPlannedRecallControls(loaded);
  const published = JSON.stringify(controls);
  const changedSnapshotContent = snapshotContent.replace(
    'Private mechanism phrase and outcome token.',
    'Changed bytes must not be indexed under the verified checksum.',
  );
  await writeFile(snapshotPath, changedSnapshotContent, { mode: 0o600 });

  assert.equal(loaded.manifestSha256, createSha256(manifestContent));
  assert.equal(controls.cases[0]?.caseId, 'case-001');
  assert.equal(controls.cases[0]?.querySha256, createSha256(query));
  assert.equal(controls.cases[0]?.privateCaseSha256, createSha256(JSON.stringify(privateCase)));
  assert.equal(controls.cases[0]?.retrievalWork.totalCandidateLimit, 100);
  assert.equal(controls.cases[0]?.relevantDistractorCount, 1);
  assert.deepEqual(controls.cases[0]?.expectedSessionOriginSha256, [
    createSha256('/private/project'),
  ]);
  assert.deepEqual(controls.cases[0]?.expectedEvidenceRelations, ['unrestricted_global_evidence']);
  assert.equal(published.includes(query), false);
  assert.equal(published.includes('Private mechanism phrase'), false);
  assert.equal(published.includes('/private/project'), false);
  assert.equal(published.includes(snapshotPath), false);

  const baseConfig: RecallConversationConfig = {
    sessionsDirectory: join(directory, 'must-not-scan-production-sessions'),
    databasePath: join(directory, 'unused-zvec'),
    statePath: join(directory, 'unused-state.json'),
    manifestPath: join(directory, 'unused-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'unused-tokenizers'),
    embeddingCacheDirectory: join(directory, 'unused-embedding-cache'),
    lockPath: join(directory, 'unused.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(directory, 'unused-diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(directory, 'unused-diagnostics.previous.jsonl'),
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
  };
  const dependencies = {
    embeddings: {
      async embedTexts(texts: readonly string[]) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return [0, 0, 1];
          }
          return text.includes('mechanism') || text === query ? [1, 0, 0] : [0, 1, 0];
        });
      },
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return { ids: Array.from(text.split(/\\s+/u).filter(Boolean).keys()) };
        },
      };
    },
  };
  await assert.rejects(
    () =>
      runPrivateQueryPlannedRecallBaseline({
        corpus: loaded,
        baseConfig,
        workDirectory: join(snapshotDirectory, 'unsafe-work'),
        dependencies,
      }),
    /work directory overlaps immutable snapshots/u,
  );
  await assert.rejects(
    () =>
      runPrivateQueryPlannedRecallBaseline({
        corpus: loaded,
        baseConfig,
        workDirectory: manifestPath,
        dependencies,
      }),
    /work directory overlaps the private manifest/u,
  );
  const baseline = await runPrivateQueryPlannedRecallBaseline({
    corpus: loaded,
    baseConfig,
    workDirectory: join(privateDirectory, 'baseline-work'),
    dependencies,
  });
  const normalArm = baseline.cases[0]?.normal;
  const workMatchedArm = baseline.cases[0]?.retrievalWorkMatched;
  assert.equal(baseline.executedSearchRequests, 2);
  assert.equal(baseline.indexedSnapshotCount, 1);
  assert.deepEqual(baseline.indexedSnapshotSha256, [createSha256(snapshotContent)]);
  assert.deepEqual(controls.snapshotSha256, baseline.indexedSnapshotSha256);
  assert.equal(normalArm?.outcome, QueryPlannedRecallBaselineOutcome.SUCCESS);
  assert.equal(workMatchedArm?.outcome, QueryPlannedRecallBaselineOutcome.SUCCESS);
  assert.deepEqual(normalArm?.listLimits, { dense: 8, lexical: 8, identifier: 8 });
  assert.deepEqual(workMatchedArm?.listLimits, { dense: 34, lexical: 33, identifier: 33 });
  assert.ok((normalArm?.totalCandidatesExamined ?? 0) > 0);
  assert.ok((workMatchedArm?.uniqueCandidatesAdmitted ?? 0) > 0);
  assert.equal(normalArm?.provenancePassed, true);
  const publishedBaseline = JSON.stringify(baseline);
  assert.equal(publishedBaseline.includes(query), false);
  assert.equal(publishedBaseline.includes('Private mechanism phrase'), false);
  assert.equal(publishedBaseline.includes('/private/project'), false);
  assert.equal(publishedBaseline.includes(snapshotPath), false);

  const evidence = createPublishableQueryPlannedRecallBaselineEvidence(controls, baseline, {
    recordedAgainstCommit: '38aab6722a6fc97dd212e704faee4373af8b363e',
    embeddingProfile: {
      requestModel: 'test-embedding',
      servedModelId: 'test-embedding-served',
      artifact: 'test-embedding.fp32',
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
    },
  });
  const report = formatPublishableQueryPlannedRecallBaselineReport(evidence);
  const publishedEvidence = JSON.stringify(evidence);
  assert.equal(evidence.controlsSha256, createSha256(JSON.stringify(controls)));
  assert.match(report, /candidate union miss/u);
  assert.match(report, /retrieval-work-matched original query/u);
  assert.equal(publishedEvidence.includes(query), false);
  assert.equal(publishedEvidence.includes('/private/project'), false);
  assert.equal(report.includes('Private mechanism phrase'), false);
  assert.equal(report.includes(snapshotPath), false);

  const controlCase = controls.cases[0];
  const baselineCase = baseline.cases[0];
  if (!controlCase || !baselineCase) {
    throw new Error('Query-planned recall baseline fixture requires one measured case');
  }
  const evidenceEnvironment = {
    recordedAgainstCommit: '38aab6722a6fc97dd212e704faee4373af8b363e',
    embeddingProfile: {
      requestModel: 'test-embedding',
      servedModelId: 'test-embedding-served',
      artifact: 'test-embedding.fp32',
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
    },
  };
  assert.throws(
    () =>
      createPublishableQueryPlannedRecallBaselineEvidence(
        controls,
        { ...baseline, indexedSnapshotSha256: ['0'.repeat(64)] },
        evidenceEnvironment,
      ),
    /indexed snapshots must exactly match manifest hashes/u,
  );
  for (const invalidInput of [
    {
      invalidControls: { ...controls, cases: [controlCase, controlCase] },
      invalidBaseline: baseline,
    },
    {
      invalidControls: controls,
      invalidBaseline: { ...baseline, cases: [baselineCase, baselineCase] },
    },
    {
      invalidControls: controls,
      invalidBaseline: { ...baseline, cases: [] },
    },
    {
      invalidControls: controls,
      invalidBaseline: {
        ...baseline,
        cases: [{ ...baselineCase, caseId: 'case-999' }],
      },
    },
  ]) {
    assert.throws(
      () =>
        createPublishableQueryPlannedRecallBaselineEvidence(
          invalidInput.invalidControls,
          invalidInput.invalidBaseline,
          evidenceEnvironment,
        ),
      /Evaluation case coverage invalid/u,
    );
  }

  await writeFile(snapshotPath, snapshotContent, { mode: 0o600 });
  await chmod(snapshotPath, 0o644);
  await assert.rejects(
    () => loadPrivateQueryPlannedRecallCorpus(manifestPath),
    /Private query-planned recall artifact permissions invalid/u,
  );
  await chmod(snapshotPath, 0o600);
  await writeFile(snapshotPath, `${snapshotContent} `, { mode: 0o600 });
  await assert.rejects(
    () => loadPrivateQueryPlannedRecallCorpus(manifestPath),
    /Private query-planned recall snapshot checksum mismatch/u,
  );
});
