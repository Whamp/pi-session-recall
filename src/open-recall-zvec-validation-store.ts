import { setTimeout as sleep } from 'node:timers/promises';

const RECALL_ZVEC_VALIDATION_OPEN_ATTEMPTS = 50;
const RECALL_ZVEC_VALIDATION_OPEN_RETRY_DELAY_MILLISECONDS = 20;

function isRecallZvecValidationLockUnavailable(error: unknown): boolean {
  return error instanceof Error && error.message.startsWith("Can't lock read-only collection:");
}

/** Opens a read-only zvec validation store after writer close, retrying only transient lock contention. */
export async function openRecallZvecValidationStore<Store>(
  openStore: () => Store,
  signal?: AbortSignal,
): Promise<Store> {
  for (let attempt = 1; attempt <= RECALL_ZVEC_VALIDATION_OPEN_ATTEMPTS; attempt += 1) {
    try {
      return openStore();
    } catch (error) {
      const shouldRetry =
        isRecallZvecValidationLockUnavailable(error) &&
        attempt < RECALL_ZVEC_VALIDATION_OPEN_ATTEMPTS;
      if (!shouldRetry) {
        throw error;
      }
      await sleep(
        RECALL_ZVEC_VALIDATION_OPEN_RETRY_DELAY_MILLISECONDS,
        undefined,
        signal ? { signal } : undefined,
      );
    }
  }
  throw new Error('Recall zvec validation store retry loop exhausted unexpectedly');
}
