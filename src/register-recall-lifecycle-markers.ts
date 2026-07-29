import type {
  AgentSettledEvent,
  SessionCompactEvent,
  SessionShutdownEvent,
  SessionStartEvent,
  SessionTreeEvent,
} from '@earendil-works/pi-coding-agent';

import { RecallWorkMarkerTrigger } from './enums.js';
import {
  createRecallWorkMarkerId,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarker,
  type RecallWorkMarkerIdentity,
  type RecallWorkMarkerTriggerPayload,
} from './recall-work-marker.js';

/** Scalar-only Pi session context available to recall lifecycle marker handlers. */
export interface RecallLifecycleMarkerContext {
  sessionManager: {
    getSessionFile(): string | undefined;
    getSessionId(): string;
    getLeafId(): string | null;
  };
}

/** Five-event Pi registration capability accepted by recall lifecycle marker handlers. */
export interface RecallLifecycleRegistrationApi {
  on(
    eventName: 'agent_settled',
    handler: (
      event: AgentSettledEvent,
      context: RecallLifecycleMarkerContext,
    ) => Promise<void> | void,
  ): void;
  on(
    eventName: 'session_compact',
    handler: (
      event: SessionCompactEvent,
      context: RecallLifecycleMarkerContext,
    ) => Promise<void> | void,
  ): void;
  on(
    eventName: 'session_tree',
    handler: (
      event: SessionTreeEvent,
      context: RecallLifecycleMarkerContext,
    ) => Promise<void> | void,
  ): void;
  on(
    eventName: 'session_shutdown',
    handler: (
      event: SessionShutdownEvent,
      context: RecallLifecycleMarkerContext,
    ) => Promise<void> | void,
  ): void;
  on(
    eventName: 'session_start',
    handler: (
      event: SessionStartEvent,
      context: RecallLifecycleMarkerContext,
    ) => Promise<void> | void,
  ): void;
}

/** Durable publication capability used by recall lifecycle handlers. */
export interface RecallLifecycleMarkerPublisher {
  publishRecallWorkMarker(marker: RecallWorkMarker): Promise<void>;
}

/** Runtime identity and diagnostic clock factory injected once per extension instance. */
export interface RecallLifecycleRuntimeFactory {
  createRuntimeInstanceId(): string;
  nowEpochMilliseconds(): number;
}

/** Registers the five documented Pi lifecycle events as scalar-only recall marker publishers. */
export function registerRecallLifecycleMarkers(
  pi: RecallLifecycleRegistrationApi,
  publisher: RecallLifecycleMarkerPublisher,
  runtimeFactory: RecallLifecycleRuntimeFactory,
): void {
  const runtimeInstanceId = runtimeFactory.createRuntimeInstanceId();
  let runtimeSequence = 0;

  async function publishLifecycleMarker(
    context: RecallLifecycleMarkerContext,
    trigger: RecallWorkMarkerTriggerPayload,
  ): Promise<void> {
    const physicalSessionPath = context.sessionManager.getSessionFile();
    if (physicalSessionPath === undefined) {
      return;
    }
    runtimeSequence += 1;
    const identity: RecallWorkMarkerIdentity = {
      version: RECALL_WORK_MARKER_VERSION,
      physicalSessionId: context.sessionManager.getSessionId(),
      physicalSessionPath,
      runtimeInstanceId,
      runtimeSequence,
      createdAtEpochMilliseconds: runtimeFactory.nowEpochMilliseconds(),
      trigger,
    };
    await publisher.publishRecallWorkMarker({
      ...identity,
      markerId: createRecallWorkMarkerId(identity),
    });
  }

  pi.on('agent_settled', async (event, context) => {
    await publishLifecycleMarker(context, { kind: RecallWorkMarkerTrigger.ACTIVITY });
  });
  pi.on('session_compact', async (event, context) => {
    await publishLifecycleMarker(context, {
      kind: RecallWorkMarkerTrigger.COMPACTION,
      logicalSessionId: context.sessionManager.getSessionId(),
      compactionEntryId: event.compactionEntry.id,
    });
  });
  pi.on('session_tree', async (event, context) => {
    if (event.oldLeafId === null && event.newLeafId === null) {
      return;
    }
    await publishLifecycleMarker(context, {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: context.sessionManager.getSessionId(),
      oldLeafEntryId: event.oldLeafId,
      newLeafEntryId: event.newLeafId,
      ...(event.summaryEntry === undefined ? {} : { summaryEntryId: event.summaryEntry.id }),
    });
  });
  pi.on('session_shutdown', async (event, context) => {
    if (event.reason === 'reload') {
      return;
    }
    const leafEntryId = context.sessionManager.getLeafId();
    if (leafEntryId === null) {
      return;
    }
    await publishLifecycleMarker(context, {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: context.sessionManager.getSessionId(),
      leafEntryId,
    });
  });
  pi.on('session_start', async (event, context) => {
    await publishLifecycleMarker(context, { kind: RecallWorkMarkerTrigger.ARRIVAL });
  });
}
