import { setTimeout as sleep } from 'node:timers/promises';

const RECALL_ZVEC_VALIDATION_OPEN_ATTEMPTS = 50;
const RECALL_ZVEC_VALIDATION_OPEN_RETRY_DELAY_MILLISECONDS = 20;

function isRecallZvecPostCloseLockUnavailable(
  error: unknown,
  accessMode: 'read-only' | 'read-write',
): boolean {
  return error instanceof Error && error.message.startsWith(`Can't lock ${accessMode} collection:`);
}

async function openRecallZvecStoreAfterClose<Store>(
  openStore: () => Store,
  accessMode: 'read-only' | 'read-write',
  signal?: AbortSignal,
): Promise<Store> {
  for (let attempt = 1; attempt <= RECALL_ZVEC_VALIDATION_OPEN_ATTEMPTS; attempt += 1) {
    try {
      return openStore();
    } catch (error) {
      const shouldRetry =
        isRecallZvecPostCloseLockUnavailable(error, accessMode) &&
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
  throw new Error('Recall zvec post-close store retry loop exhausted unexpectedly');
}

/** Opens a read-only zvec validation store after writer close, retrying only its transient lock. */
export async function openRecallZvecValidationStore<Store>(
  openStore: () => Store,
  signal?: AbortSignal,
): Promise<Store> {
  return openRecallZvecStoreAfterClose(openStore, 'read-only', signal);
}

/** Opens a writable zvec store after a coarse rebuild close, retrying only its transient lock. */
export async function openRecallZvecWritableStoreAfterClose<Store>(
  openStore: () => Store,
  signal?: AbortSignal,
): Promise<Store> {
  return openRecallZvecStoreAfterClose(openStore, 'read-write', signal);
}
