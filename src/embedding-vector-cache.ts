import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { LocalEmbeddingClient } from './local-embedding-client.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type { RecallDiagnosticsClock } from './recall-operation-diagnostics.js';

/** Version of the durable embedding-vector cache file format. */
export const EMBEDDING_VECTOR_CACHE_VERSION = 1;

/** Versioned normalization applied before embedding text and hashing cache content. */
export const EMBEDDING_TEXT_NORMALIZATION_VERSION = 'unicode-nfc-v1';

const EMBEDDING_CACHE_MAGIC = Buffer.from('PIEVC001', 'ascii');
const EMBEDDING_CACHE_CHECKSUM_BYTES = 32;
const EMBEDDING_CACHE_HEADER_LENGTH_BYTES = 4;

const embeddingCacheIdentitySchema = Type.Object(
  {
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
        assets: Type.Array(
          Type.Object(
            {
              fileName: Type.String({ minLength: 1 }),
              sha256: Type.String({ pattern: '^[a-f0-9]{64}$' }),
            },
            { additionalProperties: false },
          ),
          { minItems: 1, maxItems: 2 },
        ),
      },
      { additionalProperties: false },
    ),
    chunkPolicy: Type.Object(
      {
        version: Type.Integer({ minimum: 1 }),
        normalization: Type.String({ minLength: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

const embeddingCacheHeaderSchema = Type.Object(
  {
    cacheVersion: Type.Literal(EMBEDDING_VECTOR_CACHE_VERSION),
    checksumAlgorithm: Type.Literal('sha256'),
    normalizedTextSha256: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    vectorDimensions: Type.Integer({ minimum: 1 }),
    payloadByteLength: Type.Integer({ minimum: 1 }),
    identity: embeddingCacheIdentitySchema,
  },
  { additionalProperties: false },
);

/** Complete geometry identity that prevents vectors from crossing model or chunk-policy changes. */
export interface EmbeddingVectorCacheIdentity {
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
    normalization: string;
  };
}

/** Cache outcome for chunk vectors restored to the caller's original document order. */
export interface EmbeddingVectorCacheResult {
  vectors: number[][];
  cacheHits: number;
  newlyEmbeddedChunks: number;
  embeddingRequestCount: number;
  embeddingCacheResolutionMilliseconds: number;
  embeddingServerRequestMilliseconds: number;
}

/** Optional scalar observer for embedding requests that may fail before a cache result exists. */
export interface EmbeddingVectorCacheDiagnostics {
  recordEmbeddingServerRequest(milliseconds: number): void;
}

/** Durable content-addressed FP32 cache used by the conversation indexing service. */
export interface EmbeddingVectorCache {
  resolveEmbeddingVectors(
    texts: readonly string[],
    signal?: AbortSignal,
    diagnostics?: EmbeddingVectorCacheDiagnostics,
  ): Promise<EmbeddingVectorCacheResult>;
}

/** Durable cache location, exact geometry identity, batching, and embedding boundary. */
export interface EmbeddingVectorCacheOptions {
  cacheDirectory: string;
  identity: EmbeddingVectorCacheIdentity;
  embeddingRequestBatchSize: number;
  embeddings: LocalEmbeddingClient;
  diagnosticsClock?: RecallDiagnosticsClock;
}

interface PendingCacheEntry {
  cachePath: string;
  normalizedText: string;
  normalizedTextSha256: string;
  indexes: number[];
}

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function serializeCacheIdentity(identity: EmbeddingVectorCacheIdentity): string {
  return JSON.stringify(Value.Parse(embeddingCacheIdentitySchema, identity));
}

/** Normalizes embedding input with the version pinned by the index manifest and cache identity. */
export function normalizeConversationTextForEmbedding(text: string): string {
  return text.normalize('NFC');
}

/** Derives vector-cache identity only from fields already protected by manifest validation. */
export function createEmbeddingVectorCacheIdentity(
  manifest: EmbeddingVectorCacheIdentity,
): EmbeddingVectorCacheIdentity {
  return {
    embedding: {
      requestModel: manifest.embedding.requestModel,
      servedModelId: manifest.embedding.servedModelId,
      artifact: manifest.embedding.artifact,
      ...(manifest.embedding.artifactRepository
        ? { artifactRepository: manifest.embedding.artifactRepository }
        : {}),
      ...(manifest.embedding.artifactRevision
        ? { artifactRevision: manifest.embedding.artifactRevision }
        : {}),
      ...(manifest.embedding.artifactSha256
        ? { artifactSha256: manifest.embedding.artifactSha256 }
        : {}),
      dimensions: manifest.embedding.dimensions,
      quantization: manifest.embedding.quantization,
      pooling: manifest.embedding.pooling,
      ...(manifest.embedding.normalization
        ? { normalization: manifest.embedding.normalization }
        : {}),
      ...(manifest.embedding.canaryOperation
        ? { canaryOperation: manifest.embedding.canaryOperation }
        : {}),
      canaryProbe: manifest.embedding.canaryProbe,
      canaryFingerprint: manifest.embedding.canaryFingerprint,
    },
    tokenizer: {
      ...manifest.tokenizer,
      library: { ...manifest.tokenizer.library },
      encodeOptions: { ...manifest.tokenizer.encodeOptions },
      assets: manifest.tokenizer.assets.map((asset) => ({ ...asset })),
    },
    chunkPolicy: {
      version: manifest.chunkPolicy.version,
      normalization: manifest.chunkPolicy.normalization,
    },
  };
}

function createFp32Payload(vector: readonly number[], dimensions: number): Buffer {
  if (vector.length !== dimensions) {
    throw new Error(
      `Recall embedding cache dimension mismatch: expected ${dimensions}, received ${vector.length}`,
    );
  }
  const payload = Buffer.alloc(dimensions * Float32Array.BYTES_PER_ELEMENT);
  for (const [index, value] of vector.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`Recall embedding cache vector invalid: value ${index} is not finite`);
    }
    payload.writeFloatLE(value, index * Float32Array.BYTES_PER_ELEMENT);
  }
  return payload;
}

function readFp32Payload(payload: Buffer, dimensions: number): number[] {
  const expectedByteLength = dimensions * Float32Array.BYTES_PER_ELEMENT;
  if (payload.byteLength !== expectedByteLength) {
    throw new Error(
      `payload byte length mismatch: expected ${expectedByteLength}, received ${payload.byteLength}`,
    );
  }
  const vector: number[] = [];
  for (let index = 0; index < dimensions; index += 1) {
    const value = payload.readFloatLE(index * Float32Array.BYTES_PER_ELEMENT);
    if (!Number.isFinite(value)) {
      throw new Error(`vector value ${index} is not finite`);
    }
    vector.push(value);
  }
  return vector;
}

function createCacheFile(
  identity: EmbeddingVectorCacheIdentity,
  normalizedTextSha256: string,
  vector: readonly number[],
): { file: Buffer; fp32Vector: number[] } {
  const payload = createFp32Payload(vector, identity.embedding.dimensions);
  const header = {
    cacheVersion: EMBEDDING_VECTOR_CACHE_VERSION,
    checksumAlgorithm: 'sha256',
    normalizedTextSha256,
    vectorDimensions: identity.embedding.dimensions,
    payloadByteLength: payload.byteLength,
    identity,
  };
  const headerBytes = Buffer.from(JSON.stringify(header), 'utf8');
  const headerLength = Buffer.alloc(EMBEDDING_CACHE_HEADER_LENGTH_BYTES);
  headerLength.writeUInt32LE(headerBytes.byteLength);
  const checksummedBytes = Buffer.concat([
    EMBEDDING_CACHE_MAGIC,
    headerLength,
    headerBytes,
    payload,
  ]);
  const checksum = createHash('sha256').update(checksummedBytes).digest();
  return {
    file: Buffer.concat([checksummedBytes, checksum]),
    fp32Vector: readFp32Payload(payload, identity.embedding.dimensions),
  };
}

function parseCacheFile(
  file: Buffer,
  expectedIdentity: EmbeddingVectorCacheIdentity,
  expectedNormalizedTextSha256: string,
): number[] {
  const minimumBytes =
    EMBEDDING_CACHE_MAGIC.byteLength +
    EMBEDDING_CACHE_HEADER_LENGTH_BYTES +
    EMBEDDING_CACHE_CHECKSUM_BYTES;
  if (file.byteLength < minimumBytes) {
    throw new Error(`file is truncated: expected at least ${minimumBytes} bytes`);
  }
  if (!file.subarray(0, EMBEDDING_CACHE_MAGIC.byteLength).equals(EMBEDDING_CACHE_MAGIC)) {
    throw new Error('file header magic is invalid');
  }
  const headerLengthOffset = EMBEDDING_CACHE_MAGIC.byteLength;
  const headerLength = file.readUInt32LE(headerLengthOffset);
  const headerStart = headerLengthOffset + EMBEDDING_CACHE_HEADER_LENGTH_BYTES;
  const headerEnd = headerStart + headerLength;
  const checksumStart = file.byteLength - EMBEDDING_CACHE_CHECKSUM_BYTES;
  if (headerEnd > checksumStart) {
    throw new Error('header length exceeds the checksummed file body');
  }
  const expectedChecksum = file.subarray(checksumStart);
  const actualChecksum = createHash('sha256').update(file.subarray(0, checksumStart)).digest();
  if (!actualChecksum.equals(expectedChecksum)) {
    throw new Error('checksum mismatch');
  }

  let parsedHeader: unknown;
  try {
    parsedHeader = JSON.parse(file.subarray(headerStart, headerEnd).toString('utf8'));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`header JSON is invalid: ${message}`, { cause: error });
  }
  const header = Value.Parse(embeddingCacheHeaderSchema, parsedHeader);
  if (header.normalizedTextSha256 !== expectedNormalizedTextSha256) {
    throw new Error(
      `normalized text identity mismatch: expected ${expectedNormalizedTextSha256}, received ${header.normalizedTextSha256}`,
    );
  }
  if (serializeCacheIdentity(header.identity) !== serializeCacheIdentity(expectedIdentity)) {
    throw new Error('model, tokenizer, or chunk-policy identity mismatch');
  }
  if (header.vectorDimensions !== expectedIdentity.embedding.dimensions) {
    throw new Error(
      `vector dimensions mismatch: expected ${expectedIdentity.embedding.dimensions}, received ${header.vectorDimensions}`,
    );
  }
  const payload = file.subarray(headerEnd, checksumStart);
  if (payload.byteLength !== header.payloadByteLength) {
    throw new Error(
      `payload byte length mismatch: header declares ${header.payloadByteLength}, received ${payload.byteLength}`,
    );
  }
  return readFp32Payload(payload, header.vectorDimensions);
}

async function readCachedVector(
  cachePath: string,
  identity: EmbeddingVectorCacheIdentity,
  normalizedTextSha256: string,
): Promise<number[] | null> {
  let file: Buffer;
  try {
    file = await readFile(cachePath);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
  try {
    return parseCacheFile(file, identity, normalizedTextSha256);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall embedding cache invalid at ${cachePath}: ${message}`, { cause: error });
  }
}

async function writeCachedVectorAtomically(
  cachePath: string,
  identity: EmbeddingVectorCacheIdentity,
  normalizedTextSha256: string,
  vector: readonly number[],
): Promise<number[]> {
  const { file, fp32Vector } = createCacheFile(identity, normalizedTextSha256, vector);
  await mkdir(dirname(cachePath), { recursive: true });
  const temporaryPath = `${cachePath}.${process.pid}.${randomUUID()}.tmp`;
  let handle;
  try {
    handle = await open(temporaryPath, 'wx');
    await handle.writeFile(file);
    await handle.sync();
    await handle.close();
    handle = undefined;
    await rename(temporaryPath, cachePath);
    return fp32Vector;
  } catch (error) {
    await handle?.close();
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

/** Creates a durable cache that embeds only content-address misses and restores original order. */
export function createEmbeddingVectorCache(
  options: EmbeddingVectorCacheOptions,
): EmbeddingVectorCache {
  if (
    !Number.isInteger(options.embeddingRequestBatchSize) ||
    options.embeddingRequestBatchSize < 1
  ) {
    throw new Error('Recall embedding cache batch size invalid: expected a positive integer');
  }
  const identity = Value.Parse(embeddingCacheIdentitySchema, options.identity);
  if (identity.chunkPolicy.normalization !== EMBEDDING_TEXT_NORMALIZATION_VERSION) {
    throw new Error(
      `Recall embedding cache normalization mismatch: expected ${EMBEDDING_TEXT_NORMALIZATION_VERSION}, received ${identity.chunkPolicy.normalization}`,
    );
  }
  const identityFingerprint = sha256(serializeCacheIdentity(identity));
  const monotonicMilliseconds = () =>
    options.diagnosticsClock?.monotonicMilliseconds() ?? performance.now();

  return {
    async resolveEmbeddingVectors(texts, signal, diagnostics) {
      const resolutionStartedAtMilliseconds = monotonicMilliseconds();
      let embeddingServerRequestMilliseconds = 0;
      async function requestEmbeddings(batch: PendingCacheEntry[]): Promise<number[][]> {
        const requestStartedAtMilliseconds = monotonicMilliseconds();
        try {
          return await options.embeddings.embedTexts(
            batch.map((entry) => entry.normalizedText),
            signal,
          );
        } finally {
          const requestMilliseconds = Math.max(
            monotonicMilliseconds() - requestStartedAtMilliseconds,
            0,
          );
          embeddingServerRequestMilliseconds += requestMilliseconds;
          try {
            diagnostics?.recordEmbeddingServerRequest(requestMilliseconds);
          } catch (diagnosticError) {
            void diagnosticError;
          }
        }
      }

      const vectors = new Array<number[] | undefined>(texts.length);
      const entriesByCachePath = new Map<string, PendingCacheEntry>();
      for (const [index, text] of texts.entries()) {
        const normalizedText = normalizeConversationTextForEmbedding(text);
        const normalizedTextSha256 = sha256(normalizedText);
        const cachePath = join(
          options.cacheDirectory,
          `v${EMBEDDING_VECTOR_CACHE_VERSION}`,
          identityFingerprint,
          normalizedTextSha256.slice(0, 2),
          `${normalizedTextSha256}.fp32`,
        );
        const pending = entriesByCachePath.get(cachePath);
        if (pending) {
          pending.indexes.push(index);
        } else {
          entriesByCachePath.set(cachePath, {
            cachePath,
            normalizedText,
            normalizedTextSha256,
            indexes: [index],
          });
        }
      }

      let cacheHits = 0;
      const misses: PendingCacheEntry[] = [];
      for (const entry of entriesByCachePath.values()) {
        const cached = await readCachedVector(
          entry.cachePath,
          identity,
          entry.normalizedTextSha256,
        );
        if (cached) {
          for (const index of entry.indexes) {
            vectors[index] = cached;
            cacheHits += 1;
          }
        } else {
          misses.push(entry);
        }
      }

      let embeddingRequestCount = 0;
      for (let start = 0; start < misses.length; start += options.embeddingRequestBatchSize) {
        const batch = misses.slice(start, start + options.embeddingRequestBatchSize);
        const embedded = await requestEmbeddings(batch);
        embeddingRequestCount += 1;
        if (embedded.length !== batch.length) {
          throw new Error(
            `Recall embedding cache response count mismatch: expected ${batch.length}, received ${embedded.length}`,
          );
        }
        for (const [batchIndex, entry] of batch.entries()) {
          const vector = embedded[batchIndex];
          if (!vector) {
            throw new Error(`Recall embedding cache response missing vector ${batchIndex}`);
          }
          const fp32Vector = await writeCachedVectorAtomically(
            entry.cachePath,
            identity,
            entry.normalizedTextSha256,
            vector,
          );
          for (const index of entry.indexes) {
            vectors[index] = fp32Vector;
          }
        }
      }

      const resolvedVectors = vectors.map((vector, index) => {
        if (!vector) {
          throw new Error(`Recall embedding cache failed to resolve vector ${index}`);
        }
        return vector;
      });
      const totalResolutionMilliseconds = Math.max(
        monotonicMilliseconds() - resolutionStartedAtMilliseconds,
        0,
      );
      return {
        vectors: resolvedVectors,
        cacheHits,
        newlyEmbeddedChunks: texts.length - cacheHits,
        embeddingRequestCount,
        embeddingCacheResolutionMilliseconds: Math.max(
          totalResolutionMilliseconds - embeddingServerRequestMilliseconds,
          0,
        ),
        embeddingServerRequestMilliseconds,
      };
    },
  };
}
