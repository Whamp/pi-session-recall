import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';

import { RecallInstallationMode } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  assertRecallInstallationConfigured,
  resolveRecallInstallationMode,
} from './resolveRecallInstallationMode.js';

void test('old storage never configures a fresh target-generation installation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-installation-mode-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  assert.equal(await resolveRecallInstallationMode(config), RecallInstallationMode.UNCONFIGURED);
  assert.throws(
    () => assertRecallInstallationConfigured(RecallInstallationMode.UNCONFIGURED),
    /Recall inference is unconfigured.*pi-session-recall setup/u,
  );

  await mkdir(dirname(config.manifestPath), { recursive: true });
  await writeFile(config.manifestPath, '{"manifestVersion":5}\n', 'utf8');

  assert.equal(await resolveRecallInstallationMode(config), RecallInstallationMode.UNCONFIGURED);
});
