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
  resolveRecallLegacyOctenMarkerPath,
} from './resolveRecallInstallationMode.js';

void test('fresh installations stay unconfigured while proven legacy Octen installations retain compatibility', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-installation-mode-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  const markerPath = resolveRecallLegacyOctenMarkerPath(config);

  assert.equal(await resolveRecallInstallationMode(config), RecallInstallationMode.UNCONFIGURED);
  assert.throws(
    () => assertRecallInstallationConfigured(RecallInstallationMode.UNCONFIGURED),
    /Recall inference is unconfigured.*setup:recall/u,
  );

  await mkdir(dirname(config.manifestPath), { recursive: true });
  await writeFile(config.manifestPath, '{}\n', 'utf8');

  assert.equal(await resolveRecallInstallationMode(config), RecallInstallationMode.LEGACY_OCTEN);
  await rm(config.manifestPath);
  assert.equal(await resolveRecallInstallationMode(config), RecallInstallationMode.LEGACY_OCTEN);
  assert.equal(markerPath, join(dirname(config.manifestPath), 'legacy-octen-installation.json'));
});
