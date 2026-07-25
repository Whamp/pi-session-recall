import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { loadRecallQualityCorpus } from './recall-quality-corpus.js';

void test('fixed recall quality corpus resolves every independently declared source', async () => {
  const loaded = await loadRecallQualityCorpus(
    join(process.cwd(), 'evaluation', 'recall-quality-cases.json'),
  );

  assert.equal(loaded.specification.corpus.id, 'recall-quality-bounded-v1');
  assert.equal(loaded.sessionFiles.length, 8);
  assert.equal(loaded.specification.cases.length, 10);
  assert.deepEqual(
    Array.from(new Set(loaded.specification.cases.map(({ category }) => category))).sort(),
    [
      'branch',
      'context_dependent_reply',
      'duplicate_content',
      'exact_identifier',
      'semantic_paraphrase',
      'summary',
      'tool_evidence',
    ],
  );
  assert.ok(loaded.sessionFiles.every(({ sha256 }) => sha256.length === 64));
  assert.equal(loaded.specificationSha256.length, 64);
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
