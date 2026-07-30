import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured activation snapshots pending and quarantined marker names without reading their contents', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-replay-markers-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  await mkdir(config.markerSpoolDirectory, { recursive: true });
  await mkdir(join(config.markerQuarantineDirectory, 'corrupt'), { recursive: true });
  await writeFile(join(config.markerSpoolDirectory, 'marker_pending.json'), 'not parsed');
  await writeFile(
    join(config.markerQuarantineDirectory, 'corrupt', 'marker_quarantined.json.bad'),
    'not parsed',
  );
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const generationId = 'generation_replay_markers_owner';
  await service.createEmptyRecallGeneration({ generationId });
  const activated = await service.activateValidatedRecallGeneration(generationId);
  assert.equal(activated.replayPendingMarkerCount, 1);
  assert.equal(activated.replayQuarantinedMarkerCount, 1);
  assert.equal(await service.completeRecallGenerationReplay(), false);
});
