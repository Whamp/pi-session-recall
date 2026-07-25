import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { isUnknownRecord } from './is-unknown-record.js';
import { loadRecallQualityCorpus } from './recall-quality-corpus.js';

void test('fixed recall quality corpus resolves every independently declared source', async () => {
  const loaded = await loadRecallQualityCorpus(
    join(process.cwd(), 'evaluation', 'recall-quality-cases.json'),
  );

  assert.equal(loaded.specification.corpus.id, 'recall-quality-project-scoped-bounded-v3');
  assert.equal(loaded.sessionFiles.length, 15);
  assert.equal(loaded.specification.cases.length, 17);
  assert.deepEqual(
    Array.from(new Set(loaded.specification.cases.map(({ category }) => category))).sort(),
    [
      'branch',
      'context_dependent_reply',
      'duplicate_content',
      'exact_identifier',
      'project_scope',
      'semantic_paraphrase',
      'summary',
      'tool_evidence',
    ],
  );
  assert.ok(loaded.sessionFiles.every(({ sha256 }) => sha256.length === 64));
  assert.equal(loaded.specificationSha256.length, 64);
});

void test('project-scoped recall quality corpus accepts explicit scope and identity fixtures', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-project-scope-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const evaluationDirectory = join(directory, 'evaluation');
  await cp(join(process.cwd(), 'evaluation'), evaluationDirectory, { recursive: true });
  const specificationPath = join(evaluationDirectory, 'recall-quality-cases.json');
  const parsed: unknown = JSON.parse(await readFile(specificationPath, 'utf8'));
  assert.ok(isUnknownRecord(parsed));
  const specification = structuredClone(parsed);
  specification['version'] = 3;
  specification['projectIdentityFixtures'] = [
    {
      workingDirectory: '/evaluation/fulfillment',
      projectIdentity: 'git-origin:github.com/whamp/quality-fixture',
      identitySource: 'git_origin',
    },
  ];
  specification['projectLineages'] = {
    'git-origin:github.com/whamp/quality-fixture': ['/evaluation/prototype'],
  };
  specification['candidateCounts'] = [8];
  specification['finalCounts'] = [5];
  const bounds = specification['bounds'];
  assert.ok(isUnknownRecord(bounds));
  bounds['maximumCandidateCounts'] = 1;
  bounds['maximumSearchRequests'] = 20;
  const cases = specification['cases'];
  assert.ok(Array.isArray(cases));
  for (const evaluationCase of cases) {
    assert.ok(isUnknownRecord(evaluationCase));
    evaluationCase['scope'] = 'global';
    delete evaluationCase['invocationDirectory'];
    delete evaluationCase['expectedInvocationProjectIdentity'];
    delete evaluationCase['preLimitChannelProof'];
    evaluationCase['excludedSessionFiles'] = [];
    const expectedSources = evaluationCase['expectedSources'];
    assert.ok(Array.isArray(expectedSources));
    for (const expectedSource of expectedSources) {
      assert.ok(isUnknownRecord(expectedSource));
      const sessionFile = expectedSource['sessionFile'];
      const entryId = expectedSource['entryId'];
      assert.ok(typeof sessionFile === 'string');
      assert.ok(typeof entryId === 'string');
      const source = await readFile(join(evaluationDirectory, 'corpus', sessionFile), 'utf8');
      const header: unknown = JSON.parse(source.split('\n')[0] ?? 'null');
      assert.ok(isUnknownRecord(header));
      expectedSource['expectedSessionOrigin'] = header['cwd'];
      expectedSource['expectedEvidenceRelation'] = 'unrestricted_global_evidence';
      expectedSource['requiredContributingEntryIds'] = [entryId];
    }
  }
  await writeFile(specificationPath, `${JSON.stringify(specification, null, 2)}\n`);

  const loaded = await loadRecallQualityCorpus(specificationPath);

  assert.equal(loaded.specification.version, 3);
  assert.equal(loaded.specification.candidateCounts[0], 8);
  assert.equal(loaded.specification.finalCounts[0], 5);
  assert.equal(loaded.specification.projectIdentityFixtures.length, 1);
});

void test('fixed recall quality corpus rejects an incorrect evidence kind', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-evidence-kind-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const evaluationDirectory = join(directory, 'evaluation');
  await cp(join(process.cwd(), 'evaluation'), evaluationDirectory, { recursive: true });
  const specificationPath = join(evaluationDirectory, 'recall-quality-cases.json');
  const specification = await readFile(specificationPath, 'utf8');
  const changed = specification.replace(
    '"expectedEvidenceKind": "branch_summary"',
    '"expectedEvidenceKind": "compaction_summary"',
  );
  assert.notEqual(changed, specification);
  await writeFile(specificationPath, changed);

  await assert.rejects(
    () => loadRecallQualityCorpus(specificationPath),
    /evidence kind mismatch.*branches\.jsonl#hosted-branch-summary.*compaction_summary/,
  );
});

void test('fixed recall quality corpus rejects an incorrect branch expectation', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-quality-corpus-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const evaluationDirectory = join(directory, 'evaluation');
  await cp(join(process.cwd(), 'evaluation'), evaluationDirectory, { recursive: true });
  const specificationPath = join(evaluationDirectory, 'recall-quality-cases.json');
  const specification = await readFile(specificationPath, 'utf8');
  const changed = specification.replace(
    '"expectedBranch": "abandoned"',
    '"expectedBranch": "active"',
  );
  assert.notEqual(changed, specification);
  await writeFile(specificationPath, changed);

  await assert.rejects(
    () => loadRecallQualityCorpus(specificationPath),
    /branch expectation mismatch.*branches\.jsonl#redis-abandoned.*expected active, received abandoned/,
  );
});
