import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import test from 'node:test';

import { isUnknownRecord } from './is-unknown-record.js';

const repositoryRoot = join(dirname(new URL(import.meta.url).pathname), '..');

void test('current Recall storage has no Zvec dependency or legacy storage modules', async () => {
  const packageJson: unknown = JSON.parse(
    await readFile(join(repositoryRoot, 'package.json'), 'utf8'),
  );
  if (!isUnknownRecord(packageJson)) {
    throw new Error('Recall package.json must contain an object');
  }
  for (const dependencyGroup of ['dependencies', 'devDependencies', 'optionalDependencies']) {
    const dependencies: unknown = packageJson[dependencyGroup];
    assert.equal(
      isUnknownRecord(dependencies) ? Reflect.has(dependencies, '@zvec/zvec') : false,
      false,
    );
  }

  const sourceFileNames = await readdir(join(repositoryRoot, 'src'));
  assert.deepEqual(
    sourceFileNames.filter(
      (fileName) =>
        fileName.startsWith('legacy-v6-') ||
        fileName.startsWith('legacy-v7-') ||
        fileName.startsWith('zvec-conversation-store'),
    ),
    [],
  );
});
