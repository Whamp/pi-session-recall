import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

const currentEvaluationIdentity = {
  defaultScope: 'project',
  projectScopePolicyVersion: 1,
  repositoryIdentityPolicyVersion: 3,
  projectIdentityMetadataSchemaVersion: 3,
  lineagePolicyVersion: 1,
  lineageDigest: '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
  rankingMode: 'hybrid',
  rankFusionVersion: 1,
  reciprocalRankConstant: 60,
  activeBranchPrior: 0.01,
  candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  finalResultCount: 5,
};

function createPassingQualityEvidence(evaluationIdentity: typeof currentEvaluationIdentity) {
  return {
    version: 2,
    environment: { gitDirty: false },
    specification: { version: 3, projectLineages: {} },
    result: {
      version: 4,
      evaluationIdentity,
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
  };
}

void test('committed pre-scope evidence approves no production policy', async () => {
  const decision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);

  assert.equal(decision.automatedGatePassed, false);
  assert.equal(decision.selectedPolicy, null);
  assert.match(decision.blockers.join('; '), /predates project-scoped measurement/i);
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
  await writeFile(
    resultsPath,
    JSON.stringify({
      version: 2,
      environment: { gitDirty: false },
      specification: { version: 3, projectLineages: {} },
      result: {
        version: 4,
        evaluationIdentity: currentEvaluationIdentity,
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
              candidateCount: 16,
              finalCount: 5,
              gatePassed: true,
            },
          ],
        },
      },
    }),
  );

  await assert.rejects(
    () => readRecallQualityGateDecision(resultsPath),
    /selected policy was not a passing measured combination/,
  );
});

void test('quality evidence rejects a stale repository identity policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-project-policy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify({
      version: 2,
      environment: { gitDirty: false },
      specification: { version: 3, projectLineages: {} },
      result: {
        version: 4,
        evaluationIdentity: {
          ...currentEvaluationIdentity,
          repositoryIdentityPolicyVersion: 2,
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
  assert.match(decision.blockers.join('; '), /project identity does not match/i);
});

void test('quality evidence rejects a stale rank-fusion identity', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-ranking-policy-gate-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const resultsPath = join(directory, 'results.json');
  await writeFile(
    resultsPath,
    JSON.stringify(
      createPassingQualityEvidence({
        ...currentEvaluationIdentity,
        rankFusionVersion: 2,
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
      createPassingQualityEvidence({
        ...currentEvaluationIdentity,
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
    JSON.stringify({
      version: 2,
      environment: { gitDirty: false },
      specification: { version: 3, projectLineages: {} },
      result: {
        version: 4,
        evaluationIdentity: currentEvaluationIdentity,
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

  assert.equal(decision.automatedGatePassed, true);
  assert.deepEqual(decision.selectedPolicy, {
    chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
    candidateCount: 8,
    finalCount: 5,
  });
  assert.deepEqual(decision.blockers, []);
});
