import { existsSync, writeFileSync } from 'node:fs';
import { access, appendFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationService,
} from './recall-conversation-service.js';
import { RECALL_EMBEDDING_CANARY_TEXT } from './recall-index-manifest.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';

function isUnknownFunction(value: unknown): value is (...argumentsList: unknown[]) => unknown {
  return typeof value === 'function';
}

async function waitForFixtureRelease(releasePath: string, signal?: AbortSignal): Promise<void> {
  while (true) {
    try {
      await access(releasePath);
      return;
    } catch {
      await sleep(20, undefined, signal ? { signal } : undefined);
    }
  }
}

/** Creates the deterministic child-process service used by background-build integration tests. */
export function createRecallBackgroundIndexWorkerFixtureService(
  config: RecallConversationConfig,
): RecallConversationService {
  const dataDirectory = dirname(config.manifestPath);
  const startedPath = join(dataDirectory, 'fixture-embedding-started');
  const releasePath = join(dataDirectory, 'fixture-embedding-release');
  const projectResolutionLogPath = join(dataDirectory, 'fixture-project-resolutions.jsonl');
  const embeddingLogPath = join(dataDirectory, 'fixture-embeddings.jsonl');
  function interruptAtFixturePhase(
    phase: 'parsing' | 'embedding' | 'store-write' | 'optimization' | 'pre-activation',
  ): void {
    const triggerPath = join(dataDirectory, `fixture-interrupt-${phase}`);
    const markerPath = join(dataDirectory, `fixture-interrupted-${phase}`);
    if (!existsSync(triggerPath) || existsSync(markerPath)) {
      return;
    }
    writeFileSync(markerPath, `${process.pid}\n`, 'utf8');
    process.kill(process.pid, 'SIGKILL');
  }
  const fixtureTokenizer = {
    encodeConversationText(text: string) {
      interruptAtFixturePhase('parsing');
      return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
    },
  };
  return createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents, signal) {
        if (documents.some((document) => document.includes('background replacement evidence'))) {
          await writeFile(startedPath, 'started\n', 'utf8');
          await waitForFixtureRelease(releasePath, signal);
        }
        if (documents.some((document) => document !== RECALL_EMBEDDING_CANARY_TEXT)) {
          await appendFile(embeddingLogPath, `${JSON.stringify(documents)}\n`, 'utf8');
          interruptAtFixturePhase('embedding');
        }
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return fixtureTokenizer;
    },
    async resolveProjectIdentity(sessionOrigin) {
      await appendFile(projectResolutionLogPath, `${JSON.stringify(sessionOrigin)}\n`, 'utf8');
      return null;
    },
    openStore(mode, databasePath = config.databasePath) {
      const store = openZvecConversationStore({
        databasePath,
        dimensions: config.embeddingDimensions,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      });
      if (mode !== 'write') {
        return store;
      }
      return new Proxy(store, {
        get(target, property): unknown {
          if (property === 'upsertChunks') {
            return (...argumentsList: Parameters<typeof target.upsertChunks>) => {
              const result = target.upsertChunks(...argumentsList);
              interruptAtFixturePhase('store-write');
              return result;
            };
          }
          if (property === 'optimize') {
            return async () => {
              await target.optimize();
              interruptAtFixturePhase('optimization');
            };
          }
          if (property === 'fetchVectors') {
            return (...argumentsList: Parameters<typeof target.fetchVectors>) => {
              const result = target.fetchVectors(...argumentsList);
              interruptAtFixturePhase('pre-activation');
              return result;
            };
          }
          const value: unknown = Reflect.get(target, property, target);
          return isUnknownFunction(value)
            ? (...argumentsList: unknown[]) => value.apply(target, argumentsList)
            : value;
        },
      });
    },
  });
}
