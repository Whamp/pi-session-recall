import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

void test('committed passing hybrid quality evidence approves its measured backfill policy', async () => {
  const decision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);

  assert.equal(decision.automatedGatePassed, true);
  assert.deepEqual(decision.selectedPolicy, {
    chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
    candidateCount: 8,
    finalCount: 5,
  });
  assert.deepEqual(decision.blockers, []);
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

void test('clean passing recall quality evidence returns its measured policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-gate-'));
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

  assert.equal(decision.automatedGatePassed, true);
  assert.deepEqual(decision.selectedPolicy, {
    chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
    candidateCount: 4,
    finalCount: 3,
  });
  assert.deepEqual(decision.blockers, []);
});
