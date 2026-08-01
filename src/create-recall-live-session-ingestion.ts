import { RecallLifecycleTrigger } from './enums.js';
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
  let catchUpAbortController: AbortController | undefined;

  function scheduleIngestion(operation: () => Promise<unknown>): Promise<void> {
    pendingIngestion = pendingIngestion
      .then(operation)
      .then(() => undefined)
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        if (message === 'Recall conversation operation cancelled') {
          return;
        }
        notifyWarning(
          `Recall automatic session update skipped without changing the existing index: ${message}`,
        );
      });
    return pendingIngestion;
  }

  function scheduleActiveSessionReconciliation(
    lifecycleTrigger: RecallLifecycleTrigger,
    sessionPath?: string,
  ): Promise<void> {
    if (!sessionPath) {
      return pendingIngestion;
    }
    return scheduleIngestion(() =>
      service.reconcileSession(sessionPath, {
        lifecycleTrigger,
        lockWaitMilliseconds: BACKGROUND_RECALL_LOCK_WAIT_MILLISECONDS,
      }),
    );
  }

  function reconcileActiveSession(sessionPath?: string): Promise<void> {
    return scheduleActiveSessionReconciliation(RecallLifecycleTrigger.AGENT_SETTLED, sessionPath);
  }

  return {
    catchUpSessions() {
      catchUpAbortController?.abort();
      const abortController = new AbortController();
      catchUpAbortController = abortController;
      return scheduleIngestion(async () => {
        try {
          await service.index({
            signal: abortController.signal,
            lockWaitMilliseconds: BACKGROUND_RECALL_LOCK_WAIT_MILLISECONDS,
            requireExistingGeneration: true,
          });
        } catch (error) {
          if (!abortController.signal.aborted) {
            throw error;
          }
        } finally {
          if (catchUpAbortController === abortController) {
            catchUpAbortController = undefined;
          }
        }
      });
    },
    reconcileActiveSession,
    shutdownActiveSession(sessionPath) {
      catchUpAbortController?.abort();
      return scheduleActiveSessionReconciliation(
        RecallLifecycleTrigger.SESSION_SHUTDOWN,
        sessionPath,
      );
    },
  };
}
