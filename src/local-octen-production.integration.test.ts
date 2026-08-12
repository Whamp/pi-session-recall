import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallEmbeddingProfile, RecallSearchScope } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createLocalOctenEmbeddingProvider } from './local-octen-embedding-provider.js';
import {
  LOCAL_OCTEN_ARTIFACT_IDENTITY,
  resolveLocalOctenModelDirectory,
} from './local-octen-model-manager.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';

const modelRootDirectory = process.env.PI_LOCAL_OCTEN_MODEL_ROOT_DIRECTORY;

function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  assert.equal(left.length, right.length);
  let dotProduct = 0;
  let leftSquaredNorm = 0;
  let rightSquaredNorm = 0;
  for (const [index, leftValue] of left.entries()) {
    const rightValue = right[index]!;
    dotProduct += leftValue * rightValue;
    leftSquaredNorm += leftValue * leftValue;
    rightSquaredNorm += rightValue * rightValue;
  }
  return dotProduct / Math.sqrt(leftSquaredNorm * rightSquaredNorm);
}

void test(
  'downloaded local Octen profile conforms, indexes offline, closes, and reopens for search',
  { skip: modelRootDirectory ? false : 'PI_LOCAL_OCTEN_MODEL_ROOT_DIRECTORY is not set' },
  async (t) => {
    assert.ok(modelRootDirectory);
    const root = await mkdtemp(join(tmpdir(), 'local-octen-production-'));
    t.after(() => rm(root, { recursive: true, force: true }));
    const modelDirectory = resolveLocalOctenModelDirectory(modelRootDirectory);
    const referencePath = new URL(
      '../docs/evaluation/local-octen-0.6b-safetensors-reference-vectors.json',
      import.meta.url,
    );
    const reference: unknown = JSON.parse(await readFile(referencePath, 'utf8'));
    assert.ok(isUnknownRecord(reference));
    assert.ok(Array.isArray(reference.texts));
    assert.ok(Array.isArray(reference.vectors));
    const texts = reference.texts.slice(0, 2);
    const vectors = reference.vectors.slice(0, 2);
    assert.ok(texts.every((value) => typeof value === 'string'));
    assert.ok(
      vectors.every(
        (value) => Array.isArray(value) && value.every((item) => typeof item === 'number'),
      ),
    );

    const conformanceProvider = createLocalOctenEmbeddingProvider({
      modelDirectory,
      nativeDimensions: LOCAL_OCTEN_ARTIFACT_IDENTITY.nativeDimensions,
      parallelism: 2,
    });
    try {
      const queryEmbedding = await conformanceProvider.embedQuery(texts[0]!);
      const documentEmbedding = (await conformanceProvider.embedDocuments([texts[1]!]))[0]!;
      const queryCosine = cosineSimilarity(queryEmbedding, vectors[0]!);
      const documentCosine = cosineSimilarity(documentEmbedding, vectors[1]!);
      assert.ok(queryCosine >= 0.94, `Query conformance cosine ${queryCosine} is below 0.94`);
      assert.ok(
        documentCosine >= 0.94,
        `Document conformance cosine ${documentCosine} is below 0.94`,
      );
    } finally {
      await conformanceProvider.close();
    }

    const sessionsDirectory = join(root, 'sessions');
    const projectDirectory = join(root, 'project');
    const dataDirectory = join(root, 'recall');
    const configPath = join(root, 'recall.json');
    await Promise.all([
      mkdir(sessionsDirectory, { recursive: true }),
      mkdir(projectDirectory, { recursive: true }),
    ]);
    const session = [
      {
        type: 'session',
        version: 3,
        id: 'local-octen-end-to-end',
        timestamp: '2026-08-11T20:00:00.000Z',
        cwd: projectDirectory,
      },
      {
        type: 'message',
        id: 'local-octen-answer',
        parentId: null,
        timestamp: '2026-08-11T20:01:00.000Z',
        message: {
          role: 'assistant',
          content: 'The offline recall beacon phrase is cobalt lighthouse.',
        },
      },
    ];
    await writeFile(
      join(sessionsDirectory, 'local-octen.jsonl'),
      `${session.map((entry) => JSON.stringify(entry)).join('\n')}\n`,
      'utf8',
    );
    await writeFile(
      configPath,
      `${JSON.stringify({
        sessionsDirectory,
        dataDirectory,
        embeddingProfile: RecallEmbeddingProfile.LOCAL_OCTEN,
        localModelRootDirectory: modelRootDirectory,
        localEmbeddingParallelism: 2,
        localEmbeddingIntraOperationThreads: 1,
      })}\n`,
      'utf8',
    );
    const config = await loadRecallConversationConfig({
      configPath,
      homeDirectory: root,
      environment: {},
    });
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      throw new Error('Offline local Octen integration must not use the network');
    };
    try {
      const indexingService = createRecallConversationService(config);
      try {
        const indexed = await indexingService.index({ rebuild: true });
        assert.equal(indexed.indexSummary.failedSessions.length, 0);
        assert.ok(indexed.documentCounts.dense > 0);
      } finally {
        await indexingService.close?.();
      }

      const restartedService = createRecallConversationService(config);
      try {
        const search = await restartedService.search('offline beacon phrase', 5, {
          scope: RecallSearchScope.GLOBAL,
        });
        assert.ok(search.results.some((result) => result.content.includes('cobalt lighthouse')));
      } finally {
        await restartedService.close?.();
      }
    } finally {
      globalThis.fetch = originalFetch;
    }
  },
);
