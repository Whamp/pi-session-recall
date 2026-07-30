import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import type {
  AgentSettledEvent,
  SessionCompactEvent,
  SessionShutdownEvent,
  SessionStartEvent,
  SessionTreeEvent,
} from '@earendil-works/pi-coding-agent';

import { RecallWorkMarkerTrigger } from './enums.js';
import {
  registerRecallLifecycleMarkers,
  type RecallLifecycleMarkerContext,
  type RecallLifecycleRegistrationApi,
} from './register-recall-lifecycle-markers.js';
import { resolveRecallPhysicalSourceIdentity } from './recall-source-identity.js';
import type { RecallWorkMarker } from './recall-work-marker.js';

interface RecallLifecycleTestEvents {
  agent_settled: [AgentSettledEvent, RecallLifecycleMarkerContext];
  session_compact: [SessionCompactEvent, RecallLifecycleMarkerContext];
  session_tree: [SessionTreeEvent, RecallLifecycleMarkerContext];
  session_shutdown: [SessionShutdownEvent, RecallLifecycleMarkerContext];
  session_start: [SessionStartEvent, RecallLifecycleMarkerContext];
}

interface RecallLifecycleTestHarness {
  emit: {
    agentSettled(...eventArguments: RecallLifecycleTestEvents['agent_settled']): Promise<void>;
    sessionCompact(...eventArguments: RecallLifecycleTestEvents['session_compact']): Promise<void>;
    sessionTree(...eventArguments: RecallLifecycleTestEvents['session_tree']): Promise<void>;
    sessionShutdown(
      ...eventArguments: RecallLifecycleTestEvents['session_shutdown']
    ): Promise<void>;
    sessionStart(...eventArguments: RecallLifecycleTestEvents['session_start']): Promise<void>;
  };
  pi: RecallLifecycleRegistrationApi;
}

function createLifecycleTestHarness(): RecallLifecycleTestHarness {
  const events = new EventEmitter();
  async function finishEmittedHandlers(): Promise<void> {
    await Promise.resolve();
    await Promise.resolve();
  }
  return {
    emit: {
      async agentSettled(...eventArguments) {
        events.emit('agent_settled', ...eventArguments);
        await finishEmittedHandlers();
      },
      async sessionCompact(...eventArguments) {
        events.emit('session_compact', ...eventArguments);
        await finishEmittedHandlers();
      },
      async sessionTree(...eventArguments) {
        events.emit('session_tree', ...eventArguments);
        await finishEmittedHandlers();
      },
      async sessionShutdown(...eventArguments) {
        events.emit('session_shutdown', ...eventArguments);
        await finishEmittedHandlers();
      },
      async sessionStart(...eventArguments) {
        events.emit('session_start', ...eventArguments);
        await finishEmittedHandlers();
      },
    },
    pi: {
      on(eventName, handler) {
        events.on(eventName, handler);
      },
    },
  };
}

function createScalarOnlyContext(
  options: {
    physicalSessionPath?: string;
    physicalSessionId?: string;
    leafEntryId?: string | null;
  } = {},
): RecallLifecycleMarkerContext {
  const physicalSessionPath = options.physicalSessionPath;
  const physicalSessionId = options.physicalSessionId ?? 'physical-session-1';
  const leafEntryId = options.leafEntryId ?? 'shutdown-leaf-1';
  const forbidden = (): never => {
    throw new Error('Recall lifecycle marker hook requested forbidden session body work');
  };
  const sessionManager = new Proxy(
    {
      getSessionFile: () => physicalSessionPath,
      getSessionId: () => physicalSessionId,
      getLeafId: () => leafEntryId,
    },
    {
      get(target, property) {
        if (property === 'getSessionFile') {
          return target.getSessionFile;
        }
        if (property === 'getSessionId') {
          return target.getSessionId;
        }
        if (property === 'getLeafId') {
          return target.getLeafId;
        }
        return forbidden;
      },
    },
  );
  return { sessionManager };
}

void test('recall lifecycle markers map every documented Pi trigger with one physical source identity', async () => {
  const harness = createLifecycleTestHarness();
  const markers: RecallWorkMarker[] = [];
  const sessionsDirectory = '/trusted/sessions';
  const physicalSessionPath = `${sessionsDirectory}/session-1.jsonl`;
  const physicalSourceIdentity = resolveRecallPhysicalSourceIdentity(
    sessionsDirectory,
    physicalSessionPath,
  ).physicalSourceIdentity;
  let clock = 1_753_315_200_000;
  registerRecallLifecycleMarkers(
    harness.pi,
    {
      async publishRecallWorkMarker(marker) {
        markers.push(marker);
      },
    },
    {
      createRuntimeInstanceId() {
        return 'runtime-instance-1';
      },
      nowEpochMilliseconds() {
        return clock++;
      },
    },
    sessionsDirectory,
  );
  const context = createScalarOnlyContext({ physicalSessionPath });

  await harness.emit.sessionStart({ type: 'session_start', reason: 'startup' }, context);
  await harness.emit.agentSettled({ type: 'agent_settled' }, context);
  await harness.emit.sessionCompact(
    {
      type: 'session_compact',
      compactionEntry: {
        type: 'compaction',
        id: 'compaction-entry-1',
        parentId: 'entry-1',
        timestamp: '2026-07-27T10:00:00.000Z',
        summary: 'durable summary',
        firstKeptEntryId: 'entry-1',
        tokensBefore: 42,
      },
      fromExtension: false,
      reason: 'manual',
      willRetry: false,
    },
    context,
  );
  await harness.emit.sessionTree(
    {
      type: 'session_tree',
      oldLeafId: 'old-leaf-1',
      newLeafId: 'new-leaf-1',
      summaryEntry: {
        type: 'branch_summary',
        id: 'summary-entry-1',
        parentId: 'new-leaf-1',
        timestamp: '2026-07-27T10:01:00.000Z',
        summary: 'branch summary',
        fromId: 'old-leaf-1',
      },
    },
    context,
  );
  await harness.emit.sessionTree(
    { type: 'session_tree', oldLeafId: 'old-leaf-1', newLeafId: null },
    context,
  );
  await harness.emit.sessionTree(
    { type: 'session_tree', oldLeafId: null, newLeafId: null },
    context,
  );
  await harness.emit.sessionShutdown({ type: 'session_shutdown', reason: 'quit' }, context);

  assert.deepEqual(
    markers.map((marker) => ({
      physicalSessionId: marker.physicalSessionId,
      physicalSessionPath: marker.physicalSessionPath,
      runtimeInstanceId: marker.runtimeInstanceId,
      runtimeSequence: marker.runtimeSequence,
      createdAtEpochMilliseconds: marker.createdAtEpochMilliseconds,
      trigger: marker.trigger,
    })),
    [
      {
        physicalSessionId: physicalSourceIdentity,
        physicalSessionPath,
        runtimeInstanceId: 'runtime-instance-1',
        runtimeSequence: 1,
        createdAtEpochMilliseconds: 1_753_315_200_000,
        trigger: { kind: RecallWorkMarkerTrigger.ARRIVAL },
      },
      {
        physicalSessionId: physicalSourceIdentity,
        physicalSessionPath,
        runtimeInstanceId: 'runtime-instance-1',
        runtimeSequence: 2,
        createdAtEpochMilliseconds: 1_753_315_200_001,
        trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
      },
      {
        physicalSessionId: physicalSourceIdentity,
        physicalSessionPath,
        runtimeInstanceId: 'runtime-instance-1',
        runtimeSequence: 3,
        createdAtEpochMilliseconds: 1_753_315_200_002,
        trigger: {
          kind: RecallWorkMarkerTrigger.COMPACTION,
          logicalSessionId: 'physical-session-1',
          compactionEntryId: 'compaction-entry-1',
        },
      },
      {
        physicalSessionId: physicalSourceIdentity,
        physicalSessionPath,
        runtimeInstanceId: 'runtime-instance-1',
        runtimeSequence: 4,
        createdAtEpochMilliseconds: 1_753_315_200_003,
        trigger: {
          kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
          logicalSessionId: 'physical-session-1',
          oldLeafEntryId: 'old-leaf-1',
          newLeafEntryId: 'new-leaf-1',
          summaryEntryId: 'summary-entry-1',
        },
      },
      {
        physicalSessionId: physicalSourceIdentity,
        physicalSessionPath,
        runtimeInstanceId: 'runtime-instance-1',
        runtimeSequence: 5,
        createdAtEpochMilliseconds: 1_753_315_200_004,
        trigger: {
          kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
          logicalSessionId: 'physical-session-1',
          oldLeafEntryId: 'old-leaf-1',
          newLeafEntryId: null,
        },
      },
      {
        physicalSessionId: physicalSourceIdentity,
        physicalSessionPath,
        runtimeInstanceId: 'runtime-instance-1',
        runtimeSequence: 6,
        createdAtEpochMilliseconds: 1_753_315_200_005,
        trigger: {
          kind: RecallWorkMarkerTrigger.DEPARTURE,
          logicalSessionId: 'physical-session-1',
          leafEntryId: 'shutdown-leaf-1',
        },
      },
    ],
  );
});

void test('reload emits only arrival while replacement and quit reasons preserve runtime ownership', async () => {
  const oldHarness = createLifecycleTestHarness();
  const newHarness = createLifecycleTestHarness();
  const markers: RecallWorkMarker[] = [];
  const publisher = {
    async publishRecallWorkMarker(marker: RecallWorkMarker) {
      markers.push(marker);
    },
  };
  const sessionsDirectory = '/trusted/sessions';
  const oldPhysicalSourceIdentity = resolveRecallPhysicalSourceIdentity(
    sessionsDirectory,
    `${sessionsDirectory}/old.jsonl`,
  ).physicalSourceIdentity;
  const newPhysicalSourceIdentity = resolveRecallPhysicalSourceIdentity(
    sessionsDirectory,
    `${sessionsDirectory}/new.jsonl`,
  ).physicalSourceIdentity;
  registerRecallLifecycleMarkers(
    oldHarness.pi,
    publisher,
    {
      createRuntimeInstanceId: () => 'runtime-old',
      nowEpochMilliseconds: () => 10,
    },
    sessionsDirectory,
  );
  registerRecallLifecycleMarkers(
    newHarness.pi,
    publisher,
    {
      createRuntimeInstanceId: () => 'runtime-new',
      nowEpochMilliseconds: () => 20,
    },
    sessionsDirectory,
  );
  const oldContext = createScalarOnlyContext({
    physicalSessionPath: '/trusted/sessions/old.jsonl',
    physicalSessionId: 'old-session',
  });
  const newContext = createScalarOnlyContext({
    physicalSessionPath: '/trusted/sessions/new.jsonl',
    physicalSessionId: 'new-session',
  });

  await oldHarness.emit.sessionShutdown({ type: 'session_shutdown', reason: 'reload' }, oldContext);
  await newHarness.emit.sessionStart({ type: 'session_start', reason: 'reload' }, oldContext);
  for (const reason of ['new', 'resume', 'fork'] as const) {
    await oldHarness.emit.sessionShutdown(
      { type: 'session_shutdown', reason, targetSessionFile: '/trusted/sessions/new.jsonl' },
      oldContext,
    );
    await newHarness.emit.sessionStart(
      {
        type: 'session_start',
        reason,
        previousSessionFile: '/trusted/sessions/old.jsonl',
      },
      newContext,
    );
  }

  assert.deepEqual(
    markers.map((marker) => [
      marker.runtimeInstanceId,
      marker.physicalSessionId,
      marker.runtimeSequence,
      marker.trigger.kind,
    ]),
    [
      ['runtime-new', oldPhysicalSourceIdentity, 1, RecallWorkMarkerTrigger.ARRIVAL],
      ['runtime-old', oldPhysicalSourceIdentity, 1, RecallWorkMarkerTrigger.DEPARTURE],
      ['runtime-new', newPhysicalSourceIdentity, 2, RecallWorkMarkerTrigger.ARRIVAL],
      ['runtime-old', oldPhysicalSourceIdentity, 2, RecallWorkMarkerTrigger.DEPARTURE],
      ['runtime-new', newPhysicalSourceIdentity, 3, RecallWorkMarkerTrigger.ARRIVAL],
      ['runtime-old', oldPhysicalSourceIdentity, 3, RecallWorkMarkerTrigger.DEPARTURE],
      ['runtime-new', newPhysicalSourceIdentity, 4, RecallWorkMarkerTrigger.ARRIVAL],
    ],
  );
});

void test('recall hook modules import no session parsing, service, tokenizer, embedding, or zvec work', async () => {
  const sources = await Promise.all([
    readFile(new URL('./register-recall-lifecycle-markers.ts', import.meta.url), 'utf8'),
    readFile(new URL('./publish-recall-work-marker.ts', import.meta.url), 'utf8'),
  ]);
  for (const source of sources) {
    assert.doesNotMatch(
      source,
      /create-recall-live-session-ingestion|recall-conversation-service|session-conversation-index|tokenizer|embedding|zvec|getEntries|getBranch|buildContextEntries|openStore/iu,
    );
  }
});

void test('ephemeral sessions emit no marker and request no session body or heavy service work', async () => {
  const harness = createLifecycleTestHarness();
  let publicationCount = 0;
  registerRecallLifecycleMarkers(
    harness.pi,
    {
      async publishRecallWorkMarker() {
        publicationCount += 1;
      },
    },
    {
      createRuntimeInstanceId: () => 'ephemeral-runtime',
      nowEpochMilliseconds: () => 30,
    },
    '/trusted/sessions',
  );
  const context = createScalarOnlyContext();

  await harness.emit.sessionStart({ type: 'session_start', reason: 'startup' }, context);
  await harness.emit.agentSettled({ type: 'agent_settled' }, context);
  await harness.emit.sessionShutdown({ type: 'session_shutdown', reason: 'quit' }, context);

  assert.equal(publicationCount, 0);
});
