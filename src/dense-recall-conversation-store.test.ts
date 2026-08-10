import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import test from 'node:test';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
} from '@zvec/zvec';

import { RecallProjectIdentitySource } from './enums.js';
import { createTestSessionConversationChunk } from './recall-test-utils.js';
import { parseRepositoryIdentity } from './resolve-project-identity.js';
import {
  DENSE_RECALL_EMBEDDING_DIMENSIONS,
  openDenseRecallConversationStore,
  type DenseRecallDocument,
} from './dense-recall-conversation-store.js';

function createDenseEmbedding(...components: Array<[index: number, value: number]>): number[] {
  const embedding = Array.from({ length: DENSE_RECALL_EMBEDDING_DIMENSIONS }, () => 0);
  for (const [index, value] of components) {
    embedding[index] = value;
  }
  return embedding;
}

function createDenseRecallDocument(
  id: string,
  embedding: number[],
  options: Omit<Parameters<typeof createTestSessionConversationChunk>[0], 'id'> = {},
): DenseRecallDocument {
  return {
    ...createTestSessionConversationChunk({
      ...options,
      id,
      checksum: `${id}-checksum`,
      content: `${id} content`,
    }),
    embedding,
  };
}

void test('dense recall conversation store creates a flat FP32 schema and searches real vectors', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'dense-recall-zvec-'));
  const databasePath = join(directory, 'collection');
  const store = openDenseRecallConversationStore({ databasePath });
  let storeIsOpen = true;
  t.after(async () => {
    if (storeIsOpen) {
      store.close();
    }
    await rm(directory, { recursive: true, force: true });
  });

  const first = createDenseRecallDocument('dense-a', createDenseEmbedding([1, 1]), {
    siblingIds: ['dense-b'],
    nextSiblingId: 'dense-b',
  });
  const second = createDenseRecallDocument('dense-b', createDenseEmbedding([0, 1]), {
    siblingIds: ['dense-a'],
    previousSiblingId: 'dense-a',
  });
  store.upsertDocuments([first, second]);

  assert.equal(store.countDocuments(), 2);
  assert.equal(store.searchDenseCandidates(createDenseEmbedding([0, 1]), 1)[0]?.id, 'dense-b');
  assert.deepEqual(store.fetchVectors(['dense-a']), new Map([['dense-a', first.embedding]]));
  const winner = store.fetchDocuments(['dense-a']).get('dense-a');
  assert.ok(winner);
  assert.deepEqual([...store.fetchDocuments(winner.siblingIds).keys()], ['dense-b']);

  store.close();
  storeIsOpen = false;
  const rawCollection = ZVecOpen(databasePath, { readOnly: true });
  const vectorSchema = rawCollection.schema.vector('embedding');
  assert.equal(vectorSchema.dataType, ZVecDataType.VECTOR_FP32);
  assert.equal(vectorSchema.dimension, DENSE_RECALL_EMBEDDING_DIMENSIONS);
  assert.equal(vectorSchema.indexParams?.indexType, ZVecIndexType.FLAT);
  assert.equal(vectorSchema.indexParams?.metricType, ZVecMetricType.IP);
  assert.equal(
    rawCollection.schema.fields().some((field) => field.indexParams !== undefined),
    false,
  );
  assert.equal(
    rawCollection.schema.fields().some((field) => field.name === 'identifierContent'),
    false,
  );
  rawCollection.closeSync();
});

void test('dense recall conversation store accepts conversation, summary, branch-summary, and turn-context documents', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'dense-recall-document-kinds-'));
  const store = openDenseRecallConversationStore({ databasePath: join(directory, 'collection') });
  t.after(async () => {
    store.close();
    await rm(directory, { recursive: true, force: true });
  });
  store.upsertDocuments([
    createDenseRecallDocument('conversation', createDenseEmbedding([0, 1])),
    createDenseRecallDocument('compaction-summary', createDenseEmbedding([1, 1]), {
      documentKind: 'summary',
      summaryKind: 'compaction',
      evidenceKind: 'compaction_summary',
      role: 'summary',
    }),
    createDenseRecallDocument('branch-summary', createDenseEmbedding([2, 1]), {
      documentKind: 'summary',
      summaryKind: 'branch',
      evidenceKind: 'branch_summary',
      role: 'summary',
    }),
    createDenseRecallDocument('turn-context', createDenseEmbedding([3, 1]), {
      documentKind: 'turn_context',
      evidenceKind: 'turn_context',
      role: 'turn',
    }),
  ]);

  assert.deepEqual(
    [
      ...store
        .fetchDocuments(['conversation', 'compaction-summary', 'branch-summary', 'turn-context'])
        .values(),
    ]
      .map((document) => [document.documentKind, document.summaryKind])
      .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right))),
    [
      ['conversation', null],
      ['summary', 'branch'],
      ['summary', 'compaction'],
      ['turn_context', null],
    ],
  );
});

void test('dense recall conversation store rejects invalid document kinds and vectors', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'dense-recall-validation-'));
  const store = openDenseRecallConversationStore({ databasePath: join(directory, 'collection') });
  t.after(async () => {
    store.close();
    await rm(directory, { recursive: true, force: true });
  });

  const toolDocument = createDenseRecallDocument('tool-document', createDenseEmbedding([0, 1]));
  Reflect.set(toolDocument, 'documentKind', 'tool');
  assert.throws(
    () => store.upsertDocuments([toolDocument]),
    /only conversation, summary, branch-summary, and turn-context documents are allowed/,
  );
  assert.throws(
    () =>
      store.upsertDocuments([
        createDenseRecallDocument(
          'wrong-width',
          Array.from({ length: DENSE_RECALL_EMBEDDING_DIMENSIONS - 1 }, () => 1),
        ),
      ]),
    /expected 1024 dimensions, received 1023/,
  );
  assert.throws(
    () =>
      store.upsertDocuments([
        createDenseRecallDocument(
          'zero-vector',
          Array.from({ length: DENSE_RECALL_EMBEDDING_DIMENSIONS }, () => 0),
        ),
      ]),
    /zero vectors are not allowed/,
  );
});

void test('dense recall conversation store supports project and global search, reuse, and deletion', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'dense-recall-scope-'));
  const store = openDenseRecallConversationStore({ databasePath: join(directory, 'collection') });
  t.after(async () => {
    store.close();
    await rm(directory, { recursive: true, force: true });
  });
  const projectA = parseRepositoryIdentity('git-origin:github.com/Whamp/project-a');
  const projectB = parseRepositoryIdentity('git-origin:github.com/Whamp/project-b');
  const projectADocument = createDenseRecallDocument(
    'project-a-document',
    createDenseEmbedding([0, 0.8], [1, 0.2]),
    {
      projectAttribution: {
        projectIdentity: projectA,
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
      },
    },
  );
  const projectBDocument = createDenseRecallDocument(
    'project-b-document',
    createDenseEmbedding([0, 1]),
    {
      projectAttribution: {
        projectIdentity: projectB,
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
      },
    },
  );
  store.upsertDocuments([projectADocument, projectBDocument]);

  const query = createDenseEmbedding([0, 1]);
  assert.equal(store.searchDenseCandidates(query, 1)[0]?.id, 'project-b-document');
  assert.equal(store.searchDenseCandidates(query, 1, projectA)[0]?.id, 'project-a-document');
  assert.deepEqual(
    store.searchDenseCandidates(
      query,
      1,
      parseRepositoryIdentity('git-origin:github.com/Whamp/none'),
    ),
    [],
  );

  const replacement = {
    ...projectADocument,
    checksum: 'replacement-checksum',
    content: 'replacement content',
    embedding: createDenseEmbedding([1, 1]),
  };
  store.upsertDocuments([replacement]);
  assert.equal(
    store.fetchDocuments([replacement.id]).get(replacement.id)?.content,
    replacement.content,
  );
  assert.deepEqual(
    store.fetchVectors([replacement.id]),
    new Map([[replacement.id, replacement.embedding]]),
  );

  store.deleteDocuments([replacement.id]);
  assert.equal(store.fetchDocuments([replacement.id]).size, 0);
  assert.equal(store.countDocuments(), 1);
});

void test('dense recall conversation store rejects incompatible existing zvec data', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dense-recall-incompatible-'));
  const databasePath = join(directory, 'collection');
  try {
    const incompatible = ZVecCreateAndOpen(
      databasePath,
      new ZVecCollectionSchema({
        name: 'legacy_hnsw',
        vectors: {
          name: 'embedding',
          dataType: ZVecDataType.VECTOR_FP32,
          dimension: DENSE_RECALL_EMBEDDING_DIMENSIONS,
          indexParams: {
            indexType: ZVecIndexType.HNSW,
            metricType: ZVecMetricType.IP,
          },
        },
      }),
    );
    incompatible.closeSync();

    assert.throws(
      () => openDenseRecallConversationStore({ databasePath }),
      /Dense recall zvec schema incompatible[\s\S]*psr index --rebuild/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

const PROTOTYPE_DENSE_QUERY_CASES = [
  {
    query: 'Why have recent pi-session-recall optimization attempts failed?',
    dimension: 0,
    expectedTopId: 'query-0-rank-0',
    hnswControlTopEight: [
      'query-0-rank-0',
      'query-0-rank-1',
      'query-0-rank-2',
      'query-0-rank-3',
      'query-0-rank-4',
      'query-0-rank-5',
      'query-0-rank-6',
      'prototype-control-only-0',
    ],
    expectedOverlap: 7,
  },
  {
    query: 'How is automatic recall indexing scheduled?',
    dimension: 1,
    expectedTopId: 'query-1-rank-0',
    hnswControlTopEight: [
      'query-1-rank-0',
      'query-1-rank-1',
      'query-1-rank-2',
      'query-1-rank-3',
      'query-1-rank-4',
      'query-1-rank-5',
      'query-1-rank-6',
      'prototype-control-only-1',
    ],
    expectedOverlap: 7,
  },
  {
    query: 'Which corrupted February session files are ignored?',
    dimension: 2,
    expectedTopId: 'query-2-rank-0',
    hnswControlTopEight: [
      'query-2-rank-0',
      'query-2-rank-1',
      'query-2-rank-2',
      'query-2-rank-3',
      'query-2-rank-4',
      'query-2-rank-5',
      'query-2-rank-6',
      'query-2-rank-7',
    ],
    expectedOverlap: 8,
  },
  {
    query: 'How large is the recall database?',
    dimension: 3,
    expectedTopId: 'query-3-rank-0',
    hnswControlTopEight: [
      'query-3-rank-0',
      'query-3-rank-1',
      'query-3-rank-2',
      'query-3-rank-3',
      'query-3-rank-4',
      'query-3-rank-5',
      'query-3-rank-6',
      'query-3-rank-7',
    ],
    expectedOverlap: 8,
  },
  {
    query: 'Why would an agent use pi-session-recall instead of searching raw JSONL?',
    dimension: 4,
    expectedTopId: 'query-4-rank-0',
    hnswControlTopEight: [
      'query-4-rank-0',
      'query-4-rank-1',
      'query-4-rank-2',
      'query-4-rank-3',
      'query-4-rank-4',
      'query-4-rank-5',
      'query-4-rank-6',
      'query-4-rank-7',
    ],
    expectedOverlap: 8,
  },
] as const;

void test('fixed prototype queries preserve flat-search top results and top-eight overlap', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'dense-recall-prototype-certification-'));
  const store = openDenseRecallConversationStore({ databasePath: join(directory, 'collection') });
  t.after(async () => {
    store.close();
    await rm(directory, { recursive: true, force: true });
  });

  const documents = PROTOTYPE_DENSE_QUERY_CASES.flatMap(({ dimension }) =>
    Array.from({ length: 8 }, (_, rankIndex) =>
      createDenseRecallDocument(
        `query-${dimension}-rank-${rankIndex}`,
        createDenseEmbedding([dimension, 1 - rankIndex * 0.05]),
      ),
    ),
  );
  store.upsertDocuments(documents);

  const flatSearchMilliseconds: number[] = [];
  for (const queryCase of PROTOTYPE_DENSE_QUERY_CASES) {
    const startedAt = performance.now();
    const candidateIds = store
      .searchDenseCandidates(createDenseEmbedding([queryCase.dimension, 1]), 8)
      .map((candidate) => candidate.id);
    flatSearchMilliseconds.push(performance.now() - startedAt);

    assert.equal(candidateIds[0], queryCase.expectedTopId, queryCase.query);
    assert.equal(
      candidateIds.filter((id) =>
        queryCase.hnswControlTopEight.some((controlId) => controlId === id),
      ).length,
      queryCase.expectedOverlap,
      queryCase.query,
    );
  }

  const performanceRecord = {
    documentCount: store.countDocuments(),
    flatSearchMilliseconds,
  };
  t.diagnostic(`dense flat-search performance ${JSON.stringify(performanceRecord)}`);
  assert.equal(performanceRecord.documentCount, 40);
  assert.equal(performanceRecord.flatSearchMilliseconds.length, PROTOTYPE_DENSE_QUERY_CASES.length);
  assert.equal(performanceRecord.flatSearchMilliseconds.every(Number.isFinite), true);
});
