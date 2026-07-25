import type { RecallConversationService } from './recall-conversation-service.js';

const BACKGROUND_RECALL_LOCK_WAIT_MILLISECONDS = 250;

/** Lifecycle operations that keep the durable recall index current with Pi session files. */
export interface RecallLiveSessionIngestion {
  catchUpSessions(): Promise<void>;
  reconcileActiveSession(sessionPath?: string): Promise<void>;
  shutdownActiveSession(sessionPath?: string): Promise<void>;
}

/** Creates serialized, retryable background ingestion for active and previously missed sessions. */
export function createRecallLiveSessionIngestion(
  service: Pick<RecallConversationService, 'index' | 'reconcileSession'>,
  notifyWarning: (message: string) => void,
): RecallLiveSessionIngestion {
  let pendingIngestion = Promise.resolve();

  function scheduleIngestion(operation: () => Promise<unknown>): Promise<void> {
    pendingIngestion = pendingIngestion
      .then(operation)
      .then(() => undefined)
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        if (message === 'Recall conversation operation cancelled') {
          return;
        }
        notifyWarning(`Recall automatic session ingestion failed: ${message}`);
      });
    return pendingIngestion;
  }

  function reconcileActiveSession(sessionPath?: string): Promise<void> {
    if (!sessionPath) {
      return pendingIngestion;
    }
    return scheduleIngestion(() =>
      service.reconcileSession(sessionPath, {
        lockWaitMilliseconds: BACKGROUND_RECALL_LOCK_WAIT_MILLISECONDS,
      }),
    );
  }

  return {
    catchUpSessions() {
      return scheduleIngestion(() =>
        service.index({
          lockWaitMilliseconds: BACKGROUND_RECALL_LOCK_WAIT_MILLISECONDS,
          requireExistingGeneration: true,
        }),
      );
    },
    reconcileActiveSession,
    shutdownActiveSession: reconcileActiveSession,
  };
}
