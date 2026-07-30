import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { isUnknownRecord } from './is-unknown-record.js';
import {
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

const CURRENT_EVALUATION_IDENTITY = {
  defaultScope: 'project',
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
  fusedPoolLimit: 24,
  rerankPoolLimit: 24,
  finalResultCount: 5,
};

async function createPassingQualityEvidence(
  evaluationIdentity: typeof CURRENT_EVALUATION_IDENTITY,
): Promise<unknown> {
  const evidence: unknown = JSON.parse(await readFile(RECALL_QUALITY_RESULTS_PATH, 'utf8'));
  if (!isUnknownRecord(evidence)) {
    throw new Error('Recall quality gate test fixture invalid: expected evidence object');
  }
  const specification = Reflect.get(evidence, 'specification');
  const result = Reflect.get(evidence, 'result');
  if (!isUnknownRecord(specification) || !isUnknownRecord(result)) {
    throw new Error('Recall quality gate test fixture invalid: expected specification and result');
  }
  const boundedWork = Reflect.get(result, 'boundedWork');
  if (!isUnknownRecord(boundedWork)) {
    throw new Error('Recall quality gate test fixture invalid: expected bounded work');
  }
  const indexRuns = Reflect.get(result, 'indexRuns');
  if (!Array.isArray(indexRuns)) {
    throw new Error('Recall quality gate test fixture invalid: expected index runs');
  }
  Reflect.set(specification, 'projectLineages', {});
  Reflect.set(specification, 'projectIdentityFixtures', []);
  Reflect.set(result, 'version', 6);
  Reflect.set(result, 'storageIdentity', {
    generationFormatVersion: 1,
    generationStoreFormatVersion: 1,
    validationReceiptVersion: 1,
    incrementalEligibilityPolicyVersion: 1,
  });
  Reflect.set(result, 'evaluationIdentity', evaluationIdentity);
  for (const indexRun of indexRuns) {
    if (!isUnknownRecord(indexRun)) {
      throw new Error('Recall quality gate test fixture invalid: expected index run object');
    }
    const indexSummary = Reflect.get(indexRun, 'indexSummary');
    const totalChunks = Reflect.get(indexRun, 'totalChunks');
    if (!isUnknownRecord(indexSummary) || typeof totalChunks !== 'number') {
      throw new Error('Recall quality gate test fixture invalid: expected index run counts');
    }
    const newlyEmbeddedChunks = Reflect.get(indexSummary, 'newlyEmbeddedChunks');
    if (typeof newlyEmbeddedChunks !== 'number') {
      throw new Error('Recall quality gate test fixture invalid: expected dense count');
    }
    Reflect.set(indexSummary, 'cacheHits', 0);
    Reflect.set(indexRun, 'generationId', 'generation_quality_active');
    Reflect.set(indexRun, 'manifestFingerprint', 'a'.repeat(64));
    Reflect.set(indexRun, 'startingSnapshotFingerprint', 'b'.repeat(64));
    Reflect.set(indexRun, 'storeCounts', {
      lexicalSource: totalChunks + 100,
      dense: newlyEmbeddedChunks,
      sessionProjection: 30,
    });
  }
  Reflect.set(boundedWork, 'repositoryIdentityResolutions', 0);
  return evidence;
}

void test('committed target-generation quality evidence approves its measured policy', async () => {
  const decision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);

  assert.equal(decision.automatedGatePassed, true);
  assert.deepEqual(decision.selectedPolicy, {
    chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
    candidateCount: 8,
    finalCount: 5,
  });
  assert.deepEqual(decision.blockers, []);
});

void test('clean target-generation quality evidence approves its measured policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-target-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify(await createPassingQualityEvidence(CURRENT_EVALUATION_IDENTITY)),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, true);
  assert.deepEqual(decision.selectedPolicy, {
    chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
    candidateCount: 8,
    finalCount: 5,
  });
  assert.deepEqual(decision.blockers, []);
});

void test('quality evidence without measurements or bounded work cannot approve rollout', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-truncated-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  const evidence = await createPassingQualityEvidence(CURRENT_EVALUATION_IDENTITY);
  if (!isUnknownRecord(evidence)) {
    throw new Error('Recall quality gate test fixture invalid: expected evidence object');
  }
  const result = Reflect.get(evidence, 'result');
  if (!isUnknownRecord(result)) {
    throw new Error('Recall quality gate test fixture invalid: expected result object');
  }
  Reflect.deleteProperty(result, 'configurations');
  Reflect.deleteProperty(result, 'boundedWork');
  await writeFile(resultsPath, JSON.stringify(evidence));

  await assert.rejects(
    () => readRecallQualityGateDecision(resultsPath),
    /Recall quality gate evidence invalid/,
  );
});

void test('legacy passing recall quality evidence approves no policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-legacy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify({
      version: 1,
      environment: { gitDirty: false },
      result: {
        version: 1,
        selection: {
          passed: true,
          selected: {
            chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
            candidateCount: 4,
            finalCount: 3,
            gatePassed: true,
          },
          blockers: [],
        },
      },
    }),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, false);
  assert.equal(decision.selectedPolicy, null);
  assert.match(decision.blockers.join('; '), /evidence version 1.*rerun/i);
});

void test('pre-scope passing quality evidence approves no production policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-pre-scope-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify({
      version: 1,
      environment: { gitDirty: false },
      result: {
        version: 3,
        rankingIdentity: {
          rankingMode: 'hybrid',
          rankFusionVersion: 1,
          reciprocalRankConstant: 60,
          activeBranchPrior: 0.01,
        },
        selection: {
          passed: true,
          selected: {
            chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
            candidateCount: 8,
            finalCount: 5,
            gatePassed: true,
          },
          blockers: [],
          combinations: [
            {
              chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
              candidateCount: 8,
              finalCount: 5,
              gatePassed: true,
            },
          ],
        },
      },
    }),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, false);
  assert.equal(decision.selectedPolicy, null);
  assert.match(decision.blockers.join('; '), /predates project-scoped measurement/i);
});

void test('quality evidence rejects a selected final count outside the tool contract', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-final-count-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify({
      version: 1,
      environment: { gitDirty: false },
      result: {
        version: 3,
        selection: {
          passed: true,
          selected: {
            chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
            candidateCount: 16,
            finalCount: 11,
            gatePassed: true,
          },
          blockers: [],
        },
      },
    }),
  );

  await assert.rejects(
    () => readRecallQualityGateDecision(resultsPath),
    /Recall quality gate evidence invalid/,
  );
});

void test('quality evidence rejects a selected policy absent from measured combinations', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-unmeasured-policy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  const evidence = await createPassingQualityEvidence(CURRENT_EVALUATION_IDENTITY);
  if (!isUnknownRecord(evidence)) {
    throw new Error('Recall quality gate test fixture invalid: expected evidence object');
  }
  const result = Reflect.get(evidence, 'result');
  const selection = isUnknownRecord(result) ? Reflect.get(result, 'selection') : null;
  const selected = isUnknownRecord(selection) ? Reflect.get(selection, 'selected') : null;
  if (!isUnknownRecord(selection) || !isUnknownRecord(selected)) {
    throw new Error('Recall quality gate test fixture invalid: expected selected policy');
  }
  Reflect.set(selection, 'combinations', [{ ...selected, candidateCount: 16 }]);
  await writeFile(resultsPath, JSON.stringify(evidence));

  await assert.rejects(
    () => readRecallQualityGateDecision(resultsPath),
    /selection was not reproduced from complete measurements/,
  );
});

void test('quality evidence rejects a stale project identity policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-project-policy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify(
      await createPassingQualityEvidence({
        ...CURRENT_EVALUATION_IDENTITY,
        projectIdentityPolicyVersion: 3,
      }),
    ),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, false);
  assert.match(decision.blockers.join('; '), /project identity does not match/i);
});

void test('quality evidence rejects a stale incremental storage identity', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-storage-policy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  const evidence = await createPassingQualityEvidence(CURRENT_EVALUATION_IDENTITY);
  if (!isUnknownRecord(evidence)) {
    throw new Error('Recall quality gate test fixture invalid: expected evidence object');
  }
  const result = Reflect.get(evidence, 'result');
  const storageIdentity = isUnknownRecord(result) ? Reflect.get(result, 'storageIdentity') : null;
  if (!isUnknownRecord(storageIdentity)) {
    throw new Error('Recall quality gate test fixture invalid: expected storage identity');
  }
  Reflect.set(storageIdentity, 'incrementalEligibilityPolicyVersion', 0);
  await writeFile(resultsPath, JSON.stringify(evidence));

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, false);
  assert.match(decision.blockers.join('; '), /target generation storage identity does not match/i);
});

void test('quality evidence rejects a stale rank-fusion identity', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-ranking-policy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify(
      await createPassingQualityEvidence({
        ...CURRENT_EVALUATION_IDENTITY,
        rankFusionVersion: 1,
      }),
    ),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, false);
  assert.match(decision.blockers.join('; '), /ranking identity does not match/i);
});

void test('quality evidence rejects a stale fused-pool limit', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-fused-limit-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify(
      await createPassingQualityEvidence({
        ...CURRENT_EVALUATION_IDENTITY,
        fusedPoolLimit: 23,
      }),
    ),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, false);
  assert.match(decision.blockers.join('; '), /ranking identity does not match/i);
});

void test('quality evidence rejects a stale lineage digest', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-lineage-policy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify(
      await createPassingQualityEvidence({
        ...CURRENT_EVALUATION_IDENTITY,
        lineageDigest: 'a'.repeat(64),
      }),
    ),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, false);
  assert.match(decision.blockers.join('; '), /project identity does not match/i);
});

void test('clean passing recall quality evidence returns its measured policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify(await createPassingQualityEvidence(CURRENT_EVALUATION_IDENTITY)),
  );

  const decision = await readRecallQualityGateDecision(resultsPath);

  assert.equal(decision.automatedGatePassed, true);
  assert.deepEqual(decision.selectedPolicy, {
    chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
    candidateCount: 8,
    finalCount: 5,
  });
  assert.deepEqual(decision.blockers, []);
});
