import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  EMBEDDING_TEXT_NORMALIZATION_VERSION,
  EMBEDDING_VECTOR_CACHE_VERSION,
} from './embedding-vector-cache.js';
import { assertRecallChunkPolicy, type RecallChunkPolicy } from './recall-chunk-policy.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  OCTEN_TOKENIZER_IDENTITY,
  type ConversationTokenizerAssetIdentity,
} from './octen-conversation-tokenizer.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  createLineageDigest,
  normalizeRecallProjectLineages,
  PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
  PROJECT_IDENTITY_POLICY_VERSION,
  PROJECT_LINEAGE_POLICY_VERSION,
  type RecallProjectLineages,
} from './resolve-project-identity.js';
import { SESSION_CONVERSATION_SCHEMA_VERSION } from './session-conversation-index.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import {
  ZVEC_CONVERSATION_SCHEMA_VERSION,
  ZVEC_FTS_CONFIGURATION_VERSION,
  ZVEC_HNSW_EF_CONSTRUCTION,
  ZVEC_HNSW_EF_SEARCH,
  ZVEC_HNSW_M,
} from './zvec-conversation-store.js';

/** Version of the index-manifest file format, independent from document and zvec schemas. */
export const RECALL_INDEX_MANIFEST_VERSION = 5;

/** Lowest accepted cosine similarity across parallel slots serving the same embedding model. */
export const RECALL_EMBEDDING_CANARY_MINIMUM_COSINE_SIMILARITY = 0.9995;

/** Fixed probe whose FP32 embedding fingerprint detects served-model drift across restarts. */
export const RECALL_EMBEDDING_CANARY_TEXT =
  'pi-session-recall embedding identity canary v1: durable source provenance';

export type { RecallChunkPolicy } from './recall-chunk-policy.js';

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
  artifactRepository?: string;
  artifactRevision?: string;
  artifactSha256?: string;
  dimensions: number;
  quantization: string;
  pooling: string;
  normalization?: 'l2';
}

/** Tokenizer implementation and immutable assets that determine conversation chunk geometry. */
export interface RecallTokenizerManifestIdentity {
  model: string;
  revision: string;
  library: { name: string; version: string };
  encodeOptions: { addSpecialTokens: boolean; returnTokenTypeIds: boolean };
  assets: Array<{ fileName: string; sha256: string }>;
}

/** Profile-owned canary operation used to verify one embedding model generation. */
export interface RecallIndexEmbeddingCanaryIdentity {
  operation: 'query' | 'document';
  query: string;
  minimumRepeatCosineSimilarity: number;
}

/** Versioned identity required before one zvec index generation can be read or updated. */
export interface RecallIndexManifest {
  manifestVersion: 5;
  importPolicy: {
    version: number;
  };
  embedding: {
    requestModel: string;
    servedModelId: string;
    artifact: string;
    artifactRepository?: string;
    artifactRevision?: string;
    artifactSha256?: string;
    dimensions: number;
    quantization: string;
    pooling: string;
    normalization?: 'l2';
    canaryOperation?: 'query' | 'document';
    canaryProbe: string;
    canaryFingerprint: string;
    canaryVector: number[];
    canaryMinimumCosineSimilarity: number;
  };
  tokenizer: RecallTokenizerManifestIdentity;
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
  projectIdentity: {
    policyVersion: number;
    metadataSchemaVersion: number;
    lineagePolicyVersion: number;
    lineageDigest: string;
  };
  zvec: {
    schemaVersion: number;
    ftsConfigurationVersion: number;
    vectorQuantization: string;
    hnswM: number;
    hnswEfConstruction: number;
    hnswEfSearch: number;
  };
}

interface RecoveredRecallEmbeddingCanary {
  dimensions: number;
  canaryVector: number[];
  canaryMinimumCosineSimilarity: number;
}

const recoverableRecallEmbeddingCanarySchema = Type.Object(
  {
    embedding: Type.Object(
      {
        dimensions: Type.Integer({ minimum: 1 }),
        canaryFingerprint: Type.String({ pattern: '^[a-f0-9]{64}$' }),
        canaryVector: Type.Array(Type.Number(), { minItems: 1 }),
        canaryMinimumCosineSimilarity: Type.Literal(
          RECALL_EMBEDDING_CANARY_MINIMUM_COSINE_SIMILARITY,
        ),
      },
      { additionalProperties: true },
    ),
  },
  { additionalProperties: true },
);

const manifestAssetSchema = Type.Object(
  {
    fileName: Type.String({ minLength: 1 }),
    sha256: Type.String({ pattern: '^[a-f0-9]{64}$' }),
  },
  { additionalProperties: false },
);

const recallIndexManifestSchema = Type.Object(
  {
    manifestVersion: Type.Literal(5),
    importPolicy: Type.Object(
      {
        version: Type.Literal(SESSION_IMPORT_POLICY_VERSION),
      },
      { additionalProperties: false },
    ),
    embedding: Type.Object(
      {
        requestModel: Type.String({ minLength: 1 }),
        servedModelId: Type.String({ minLength: 1 }),
        artifact: Type.String({ minLength: 1 }),
        artifactRepository: Type.Optional(Type.String({ minLength: 1 })),
        artifactRevision: Type.Optional(Type.String({ minLength: 1 })),
        artifactSha256: Type.Optional(Type.String({ pattern: '^[a-f0-9]{64}$' })),
        dimensions: Type.Integer({ minimum: 1 }),
        quantization: Type.String({ minLength: 1 }),
        pooling: Type.String({ minLength: 1 }),
        normalization: Type.Optional(Type.Literal('l2')),
        canaryOperation: Type.Optional(
          Type.Union([Type.Literal('query'), Type.Literal('document')]),
        ),
        canaryProbe: Type.String({ minLength: 1 }),
        canaryFingerprint: Type.String({ pattern: '^[a-f0-9]{64}$' }),
        canaryVector: Type.Array(Type.Number(), { minItems: 1 }),
        canaryMinimumCosineSimilarity: Type.Literal(
          RECALL_EMBEDDING_CANARY_MINIMUM_COSINE_SIMILARITY,
        ),
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
        assets: Type.Array(manifestAssetSchema, { minItems: 1, maxItems: 2 }),
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
    projectIdentity: Type.Object(
      {
        policyVersion: Type.Literal(PROJECT_IDENTITY_POLICY_VERSION),
        metadataSchemaVersion: Type.Literal(PROJECT_IDENTITY_METADATA_SCHEMA_VERSION),
        lineagePolicyVersion: Type.Literal(PROJECT_LINEAGE_POLICY_VERSION),
        lineageDigest: Type.String({ pattern: '^[a-f0-9]{64}$' }),
      },
      { additionalProperties: false },
    ),
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
): RecallTokenizerManifestIdentity {
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

function createCanonicalRecallEmbeddingCanary(
  embedding: readonly number[],
  dimensions: number,
): number[] {
  if (embedding.length !== dimensions) {
    throw new Error(
      `Recall embedding canary dimension mismatch: expected ${dimensions}, received ${embedding.length}`,
    );
  }
  const canonical: number[] = [];
  let squaredNorm = 0;
  for (const [index, value] of embedding.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`Recall embedding canary invalid: value ${index} is not finite`);
    }
    const float32Value = Math.fround(value);
    canonical.push(float32Value);
    squaredNorm += float32Value * float32Value;
  }
  if (squaredNorm === 0) {
    throw new Error('Recall embedding canary invalid: vector norm must be positive');
  }
  return canonical;
}

/** Hashes one canonical FP32 canary vector after validating dimensions and finite values. */
export function createRecallEmbeddingCanaryFingerprint(
  embedding: readonly number[],
  dimensions: number,
): string {
  const canonical = createCanonicalRecallEmbeddingCanary(embedding, dimensions);
  const bytes = Buffer.alloc(dimensions * Float32Array.BYTES_PER_ELEMENT);
  for (const [index, value] of canonical.entries()) {
    bytes.writeFloatLE(value, index * Float32Array.BYTES_PER_ELEMENT);
  }
  return createHash('sha256').update(bytes).digest('hex');
}

/** Recovers a validated model-check vector from an otherwise incompatible index manifest. */
export async function recoverRecallEmbeddingCanaryFromManifest(
  manifestPath: string,
  expectedDimensions: number,
): Promise<RecoveredRecallEmbeddingCanary | null> {
  let source: string;
  try {
    source = await readFile(manifestPath, 'utf8');
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }

  try {
    const recovered = Value.Parse(recoverableRecallEmbeddingCanarySchema, JSON.parse(source));
    if (recovered.embedding.dimensions !== expectedDimensions) {
      return null;
    }
    const actualFingerprint = createRecallEmbeddingCanaryFingerprint(
      recovered.embedding.canaryVector,
      expectedDimensions,
    );
    if (actualFingerprint !== recovered.embedding.canaryFingerprint) {
      return null;
    }
    return {
      dimensions: recovered.embedding.dimensions,
      canaryVector: [...recovered.embedding.canaryVector],
      canaryMinimumCosineSimilarity: recovered.embedding.canaryMinimumCosineSimilarity,
    };
  } catch {
    return null;
  }
}

/** Creates the expected manifest for the current model, tokenizer, policy, and store code. */
export function createRecallIndexManifest(options: {
  embeddingIdentity: RecallEmbeddingModelIdentity;
  canaryEmbedding: readonly number[];
  embeddingCanary?: RecallIndexEmbeddingCanaryIdentity;
  tokenizerIdentity?: RecallTokenizerManifestIdentity;
  chunkPolicy?: RecallChunkPolicy;
  projectLineages?: RecallProjectLineages;
}): RecallIndexManifest {
  const canaryVector = createCanonicalRecallEmbeddingCanary(
    options.canaryEmbedding,
    options.embeddingIdentity.dimensions,
  );
  const chunkPolicy = options.chunkPolicy ?? DEFAULT_RECALL_CHUNK_POLICY;
  assertRecallChunkPolicy(chunkPolicy);
  return {
    manifestVersion: RECALL_INDEX_MANIFEST_VERSION,
    importPolicy: {
      version: SESSION_IMPORT_POLICY_VERSION,
    },
    embedding: {
      ...options.embeddingIdentity,
      ...(options.embeddingCanary ? { canaryOperation: options.embeddingCanary.operation } : {}),
      canaryProbe: options.embeddingCanary?.query ?? RECALL_EMBEDDING_CANARY_TEXT,
      canaryFingerprint: createRecallEmbeddingCanaryFingerprint(
        canaryVector,
        options.embeddingIdentity.dimensions,
      ),
      canaryVector,
      canaryMinimumCosineSimilarity:
        options.embeddingCanary?.minimumRepeatCosineSimilarity ??
        RECALL_EMBEDDING_CANARY_MINIMUM_COSINE_SIMILARITY,
    },
    tokenizer: structuredClone(
      options.tokenizerIdentity ?? createTokenizerManifestIdentity(OCTEN_TOKENIZER_IDENTITY),
    ),
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
    projectIdentity: {
      policyVersion: PROJECT_IDENTITY_POLICY_VERSION,
      metadataSchemaVersion: PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
      lineagePolicyVersion: PROJECT_LINEAGE_POLICY_VERSION,
      lineageDigest: createLineageDigest(
        options.projectLineages ?? normalizeRecallProjectLineages({}),
      ),
    },
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

/** Computes cosine similarity between two validated canonical FP32 canary vectors. */
export function calculateRecallEmbeddingCanaryCosineSimilarity(
  actual: readonly number[],
  expected: readonly number[],
  dimensions: number,
): number {
  const actualCanonical = createCanonicalRecallEmbeddingCanary(actual, dimensions);
  const expectedCanonical = createCanonicalRecallEmbeddingCanary(expected, dimensions);
  let dotProduct = 0;
  let actualSquaredNorm = 0;
  let expectedSquaredNorm = 0;
  for (let index = 0; index < dimensions; index += 1) {
    const actualValue = actualCanonical[index];
    const expectedValue = expectedCanonical[index];
    if (actualValue === undefined || expectedValue === undefined) {
      throw new Error(`Recall embedding canary invalid: missing value ${index}`);
    }
    dotProduct += actualValue * expectedValue;
    actualSquaredNorm += actualValue * actualValue;
    expectedSquaredNorm += expectedValue * expectedValue;
  }
  return dotProduct / Math.sqrt(actualSquaredNorm * expectedSquaredNorm);
}

function assertRecallIndexManifestCanaryIntegrity(manifest: RecallIndexManifest): void {
  const actualFingerprint = createRecallEmbeddingCanaryFingerprint(
    manifest.embedding.canaryVector,
    manifest.embedding.dimensions,
  );
  if (actualFingerprint !== manifest.embedding.canaryFingerprint) {
    throw new Error(
      `Recall embedding canary fingerprint mismatch: expected ${manifest.embedding.canaryFingerprint}, received ${actualFingerprint}`,
    );
  }
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
  if (path === 'embedding.canaryFingerprint' || path === 'embedding.canaryVector') {
    return;
  }
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
  try {
    const canaryCosineSimilarity = calculateRecallEmbeddingCanaryCosineSimilarity(
      actual.embedding.canaryVector,
      expected.embedding.canaryVector,
      expected.embedding.dimensions,
    );
    if (canaryCosineSimilarity < expected.embedding.canaryMinimumCosineSimilarity) {
      mismatches.push(
        `embedding.canaryCosineSimilarity: expected at least ${expected.embedding.canaryMinimumCosineSimilarity}, received ${canaryCosineSimilarity}`,
      );
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    mismatches.push(`embedding.canaryVector: ${message}`);
  }
  if (mismatches.length > 0) {
    throw new Error(
      `Recall index manifest incompatible at ${manifestPath}:\n- ${mismatches.join('\n- ')}\nReindex with /pi-session-recall-index --rebuild.`,
    );
  }
}

function readRecallIndexManifestVersion(value: unknown): number | undefined {
  if (
    !isUnknownRecord(value) ||
    typeof value.manifestVersion !== 'number' ||
    !Number.isInteger(value.manifestVersion)
  ) {
    return undefined;
  }
  return value.manifestVersion;
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
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall index manifest unreadable at ${manifestPath}: ${message}. Reload Pi first. If this persists, update pi-session-recall and inspect or restore the manifest. Do not rebuild automatically.`,
      { cause: error },
    );
  }
  const manifestVersion = readRecallIndexManifestVersion(parsed);
  if (manifestVersion !== undefined && manifestVersion > RECALL_INDEX_MANIFEST_VERSION) {
    throw new Error(
      `Recall index manifest version ${manifestVersion} at ${manifestPath} is newer than this pi-session-recall extension supports (${RECALL_INDEX_MANIFEST_VERSION}). Reload Pi to load the installed extension. If this persists, update pi-session-recall and reload Pi again. Do not rebuild the index.`,
    );
  }
  try {
    const manifest = Value.Parse(recallIndexManifestSchema, parsed);
    assertRecallIndexManifestCanaryIntegrity(manifest);
    return manifest;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall index manifest invalid at ${manifestPath}: ${message}. Reload Pi first. If this persists, update pi-session-recall and inspect or restore the manifest. Do not rebuild automatically.`,
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
  assertRecallIndexManifestCanaryIntegrity(manifest);
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
