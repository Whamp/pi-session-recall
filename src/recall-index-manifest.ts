import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  EMBEDDING_TEXT_NORMALIZATION_VERSION,
  EMBEDDING_VECTOR_CACHE_VERSION,
} from './embedding-vector-cache.js';
import {
  OCTEN_TOKENIZER_IDENTITY,
  type ConversationTokenizerAssetIdentity,
} from './octen-conversation-tokenizer.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { SESSION_CONVERSATION_SCHEMA_VERSION } from './session-conversation-index.js';
import {
  ZVEC_CONVERSATION_SCHEMA_VERSION,
  ZVEC_FTS_CONFIGURATION_VERSION,
  ZVEC_HNSW_EF_CONSTRUCTION,
  ZVEC_HNSW_EF_SEARCH,
  ZVEC_HNSW_M,
} from './zvec-conversation-store.js';

/** Version of the index-manifest file format, independent from document and zvec schemas. */
export const RECALL_INDEX_MANIFEST_VERSION = 1;

/** Fixed probe whose FP32 embedding fingerprint detects served-model drift across restarts. */
export const RECALL_EMBEDDING_CANARY_TEXT =
  'pi-session-recall embedding identity canary v1: durable source provenance';

/** Token ceiling and sibling overlap that define one recall index's chunk geometry. */
export interface RecallChunkPolicy {
  maxTokens: number;
  overlapTokens: number;
}

/** Production chunk geometry used unless a bounded evaluation supplies another policy. */
export const DEFAULT_RECALL_CHUNK_POLICY: Readonly<RecallChunkPolicy> = {
  maxTokens: 1_024,
  overlapTokens: 128,
};

/** Full configured identity of the served embedding model, beyond its request alias. */
export interface RecallEmbeddingModelIdentity {
  requestModel: string;
  servedModelId: string;
  artifact: string;
  dimensions: number;
  quantization: string;
  pooling: string;
}

/** Versioned identity required before one zvec index generation can be read or updated. */
export interface RecallIndexManifest {
  manifestVersion: 1;
  embedding: {
    requestModel: string;
    servedModelId: string;
    artifact: string;
    dimensions: number;
    quantization: string;
    pooling: string;
    canaryProbe: string;
    canaryFingerprint: string;
  };
  tokenizer: {
    model: string;
    revision: string;
    library: { name: string; version: string };
    encodeOptions: { addSpecialTokens: boolean; returnTokenTypeIds: boolean };
    assets: Array<{ fileName: string; sha256: string }>;
  };
  chunkPolicy: {
    version: number;
    maxTokens: number;
    overlapTokens: number;
    boundaryAlgorithm: string;
    normalization: string;
  };
  conversationSchemaVersion: number;
  provenanceSchemaVersion: number;
  embeddingCacheVersion: number;
  zvec: {
    schemaVersion: number;
    ftsConfigurationVersion: number;
    vectorQuantization: string;
    hnswM: number;
    hnswEfConstruction: number;
    hnswEfSearch: number;
  };
}

const manifestAssetSchema = Type.Object(
  {
    fileName: Type.String({ minLength: 1 }),
    sha256: Type.String({ pattern: '^[a-f0-9]{64}$' }),
  },
  { additionalProperties: false },
);

const recallIndexManifestSchema = Type.Object(
  {
    manifestVersion: Type.Literal(1),
    embedding: Type.Object(
      {
        requestModel: Type.String({ minLength: 1 }),
        servedModelId: Type.String({ minLength: 1 }),
        artifact: Type.String({ minLength: 1 }),
        dimensions: Type.Integer({ minimum: 1 }),
        quantization: Type.String({ minLength: 1 }),
        pooling: Type.String({ minLength: 1 }),
        canaryProbe: Type.String({ minLength: 1 }),
        canaryFingerprint: Type.String({ pattern: '^[a-f0-9]{64}$' }),
      },
      { additionalProperties: false },
    ),
    tokenizer: Type.Object(
      {
        model: Type.String({ minLength: 1 }),
        revision: Type.String({ minLength: 1 }),
        library: Type.Object(
          {
            name: Type.String({ minLength: 1 }),
            version: Type.String({ minLength: 1 }),
          },
          { additionalProperties: false },
        ),
        encodeOptions: Type.Object(
          {
            addSpecialTokens: Type.Boolean(),
            returnTokenTypeIds: Type.Boolean(),
          },
          { additionalProperties: false },
        ),
        assets: Type.Array(manifestAssetSchema, { minItems: 2, maxItems: 2 }),
      },
      { additionalProperties: false },
    ),
    chunkPolicy: Type.Object(
      {
        version: Type.Integer({ minimum: 1 }),
        maxTokens: Type.Integer({ minimum: 1 }),
        overlapTokens: Type.Integer({ minimum: 0 }),
        boundaryAlgorithm: Type.String({ minLength: 1 }),
        normalization: Type.String({ minLength: 1 }),
      },
      { additionalProperties: false },
    ),
    conversationSchemaVersion: Type.Integer({ minimum: 1 }),
    provenanceSchemaVersion: Type.Integer({ minimum: 1 }),
    embeddingCacheVersion: Type.Integer({ minimum: 1 }),
    zvec: Type.Object(
      {
        schemaVersion: Type.Integer({ minimum: 1 }),
        ftsConfigurationVersion: Type.Integer({ minimum: 1 }),
        vectorQuantization: Type.String({ minLength: 1 }),
        hnswM: Type.Integer({ minimum: 1 }),
        hnswEfConstruction: Type.Integer({ minimum: 1 }),
        hnswEfSearch: Type.Integer({ minimum: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

function createTokenizerManifestIdentity(
  identity: ConversationTokenizerAssetIdentity,
): RecallIndexManifest['tokenizer'] {
  return {
    model: identity.model,
    revision: identity.revision,
    library: { ...identity.library },
    encodeOptions: { ...identity.encodeOptions },
    assets: [
      {
        fileName: identity.tokenizerJson.fileName,
        sha256: identity.tokenizerJson.sha256,
      },
      {
        fileName: identity.tokenizerConfigJson.fileName,
        sha256: identity.tokenizerConfigJson.sha256,
      },
    ],
  };
}

/** Hashes one exact FP32 canary vector after validating dimensions and finite values. */
export function createRecallEmbeddingCanaryFingerprint(
  embedding: number[],
  dimensions: number,
): string {
  if (embedding.length !== dimensions) {
    throw new Error(
      `Recall embedding canary dimension mismatch: expected ${dimensions}, received ${embedding.length}`,
    );
  }
  const bytes = Buffer.alloc(dimensions * Float32Array.BYTES_PER_ELEMENT);
  for (const [index, value] of embedding.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`Recall embedding canary invalid: value ${index} is not finite`);
    }
    bytes.writeFloatLE(value, index * Float32Array.BYTES_PER_ELEMENT);
  }
  return createHash('sha256').update(bytes).digest('hex');
}

function assertRecallChunkPolicy(chunkPolicy: RecallChunkPolicy): void {
  if (
    !Number.isInteger(chunkPolicy.maxTokens) ||
    chunkPolicy.maxTokens < 1 ||
    chunkPolicy.maxTokens > 1_024 ||
    !Number.isInteger(chunkPolicy.overlapTokens) ||
    chunkPolicy.overlapTokens < 0 ||
    chunkPolicy.overlapTokens > 128 ||
    chunkPolicy.overlapTokens >= chunkPolicy.maxTokens
  ) {
    throw new Error(
      'Recall chunk policy invalid: maxTokens must be 1..1024 and overlapTokens must be 0..128 and smaller than maxTokens',
    );
  }
}

/** Creates the expected manifest for the current model, tokenizer, policy, and store code. */
export function createRecallIndexManifest(options: {
  embeddingIdentity: RecallEmbeddingModelIdentity;
  canaryFingerprint: string;
  chunkPolicy?: RecallChunkPolicy;
}): RecallIndexManifest {
  if (!/^[a-f0-9]{64}$/u.test(options.canaryFingerprint)) {
    throw new Error('Recall embedding canary fingerprint invalid: expected lowercase SHA-256');
  }
  const chunkPolicy = options.chunkPolicy ?? DEFAULT_RECALL_CHUNK_POLICY;
  assertRecallChunkPolicy(chunkPolicy);
  return {
    manifestVersion: RECALL_INDEX_MANIFEST_VERSION,
    embedding: {
      ...options.embeddingIdentity,
      canaryProbe: RECALL_EMBEDDING_CANARY_TEXT,
      canaryFingerprint: options.canaryFingerprint,
    },
    tokenizer: createTokenizerManifestIdentity(OCTEN_TOKENIZER_IDENTITY),
    chunkPolicy: {
      version: 2,
      maxTokens: chunkPolicy.maxTokens,
      overlapTokens: chunkPolicy.overlapTokens,
      boundaryAlgorithm: 'markdown-structure-v1',
      normalization: EMBEDDING_TEXT_NORMALIZATION_VERSION,
    },
    conversationSchemaVersion: SESSION_CONVERSATION_SCHEMA_VERSION,
    provenanceSchemaVersion: SESSION_CONVERSATION_SCHEMA_VERSION,
    embeddingCacheVersion: EMBEDDING_VECTOR_CACHE_VERSION,
    zvec: {
      schemaVersion: ZVEC_CONVERSATION_SCHEMA_VERSION,
      ftsConfigurationVersion: ZVEC_FTS_CONFIGURATION_VERSION,
      vectorQuantization: 'fp32',
      hnswM: ZVEC_HNSW_M,
      hnswEfConstruction: ZVEC_HNSW_EF_CONSTRUCTION,
      hnswEfSearch: ZVEC_HNSW_EF_SEARCH,
    },
  };
}

function isUnknownRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function formatManifestValue(value: unknown): string {
  const formatted = JSON.stringify(value);
  return formatted === undefined ? String(value) : formatted;
}

function collectManifestMismatches(
  actual: unknown,
  expected: unknown,
  path: string,
  mismatches: string[],
): void {
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) {
      mismatches.push(`${path}: expected array, received ${formatManifestValue(actual)}`);
      return;
    }
    if (actual.length !== expected.length) {
      mismatches.push(`${path}.length: expected ${expected.length}, received ${actual.length}`);
    }
    for (const [index, expectedItem] of expected.entries()) {
      collectManifestMismatches(actual[index], expectedItem, `${path}[${index}]`, mismatches);
    }
    return;
  }
  if (isUnknownRecord(expected)) {
    if (!isUnknownRecord(actual)) {
      mismatches.push(`${path}: expected object, received ${formatManifestValue(actual)}`);
      return;
    }
    for (const [key, expectedValue] of Object.entries(expected)) {
      const childPath = path ? `${path}.${key}` : key;
      collectManifestMismatches(actual[key], expectedValue, childPath, mismatches);
    }
    return;
  }
  if (!Object.is(actual, expected)) {
    mismatches.push(
      `${path}: expected ${formatManifestValue(expected)}, received ${formatManifestValue(actual)}`,
    );
  }
}

/** Rejects a missing or incompatible manifest and reports every identity mismatch together. */
export function assertRecallIndexManifestCompatible(
  actual: RecallIndexManifest | null,
  expected: RecallIndexManifest,
  manifestPath: string,
): asserts actual is RecallIndexManifest {
  if (!actual) {
    throw new Error(
      `Recall index manifest missing at ${manifestPath}; reindex with /pi-session-recall-index --rebuild`,
    );
  }
  const mismatches: string[] = [];
  collectManifestMismatches(actual, expected, '', mismatches);
  if (mismatches.length > 0) {
    throw new Error(
      `Recall index manifest incompatible at ${manifestPath}:\n- ${mismatches.join('\n- ')}\nReindex with /pi-session-recall-index --rebuild.`,
    );
  }
}

/** Reads and validates an index manifest, returning null only when the file is absent. */
export async function readRecallIndexManifest(
  manifestPath: string,
): Promise<RecallIndexManifest | null> {
  let content: string;
  try {
    content = await readFile(manifestPath, 'utf8');
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
  try {
    const parsed: unknown = JSON.parse(content);
    return Value.Parse(recallIndexManifestSchema, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall index manifest invalid at ${manifestPath}: ${message}; reindex with /pi-session-recall-index --rebuild`,
      { cause: error },
    );
  }
}

/** Persists a complete index manifest through a unique temporary file and atomic rename. */
export async function writeRecallIndexManifest(
  manifestPath: string,
  manifest: RecallIndexManifest,
): Promise<void> {
  Value.Parse(recallIndexManifestSchema, manifest);
  await mkdir(dirname(manifestPath), { recursive: true });
  const temporaryPath = `${manifestPath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(manifest, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
    await rename(temporaryPath, manifestPath);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}
