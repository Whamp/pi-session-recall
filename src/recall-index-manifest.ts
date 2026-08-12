import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { assertRecallChunkPolicy, type RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  SQLITE_RECALL_DATABASE_MANIFEST_IDENTITY,
  type SqliteRecallDatabaseManifestIdentity,
} from './sqlite-recall-database.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
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
/** Version of the unified SQLite Recall database manifest. */
export const RECALL_INDEX_MANIFEST_VERSION = 8;

/** Frozen chunk geometry selected by the accepted recall-quality evaluation. */
export const DEFAULT_RECALL_CHUNK_POLICY: Readonly<RecallChunkPolicy> = Object.freeze({
  maxTokens: 512,
  overlapTokens: 64,
});

/** Octen model and stored-prefix semantics that determine vector compatibility. */
export interface RecallEmbeddingModelIdentity {
  requestModel: string;
  servedModelId: string;
  nativeDimensions: number;
  storedDimensions: number;
  executionBackend?: string;
  transformation: 'vendor-prefix-then-l2-v1' | 'tokenizer-final-token-then-l2-v1';
}

/** Tokenizer implementation and immutable assets that determine chunk geometry. */
export interface RecallTokenizerManifestIdentity {
  model: string;
  revision: string;
  library: { name: string; version: string };
  encodeOptions: { addSpecialTokens: boolean; returnTokenTypeIds: boolean };
  assets: Array<{ fileName: string; sha256: string }>;
}

/** Complete compatibility identity for one explicitly maintained Recall database. */
export interface RecallIndexManifest {
  manifestVersion: 8;
  importPolicy: { version: number };
  embedding: RecallEmbeddingModelIdentity;
  tokenizer: RecallTokenizerManifestIdentity;
  chunkPolicy: {
    version: 3;
    maxTokens: number;
    overlapTokens: number;
    boundaryAlgorithm: 'markdown-structure-v1';
  };
  conversationSchemaVersion: number;
  provenanceSchemaVersion: number;
  projectIdentity: {
    policyVersion: number;
    metadataSchemaVersion: number;
    lineagePolicyVersion: number;
    lineageDigest: string;
  };
  sqliteRecallDatabase: SqliteRecallDatabaseManifestIdentity;
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
    manifestVersion: Type.Literal(8),
    importPolicy: Type.Object(
      { version: Type.Literal(SESSION_IMPORT_POLICY_VERSION) },
      { additionalProperties: false },
    ),
    embedding: Type.Object(
      {
        requestModel: Type.String({ minLength: 1 }),
        servedModelId: Type.String({ minLength: 1 }),
        nativeDimensions: Type.Integer({ minimum: 1 }),
        storedDimensions: Type.Integer({ minimum: 1 }),
        executionBackend: Type.Optional(Type.String({ minLength: 1 })),
        transformation: Type.Union([
          Type.Literal('vendor-prefix-then-l2-v1'),
          Type.Literal('tokenizer-final-token-then-l2-v1'),
        ]),
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
        assets: Type.Array(manifestAssetSchema, { minItems: 1 }),
      },
      { additionalProperties: false },
    ),
    chunkPolicy: Type.Object(
      {
        version: Type.Literal(3),
        maxTokens: Type.Integer({ minimum: 1 }),
        overlapTokens: Type.Integer({ minimum: 0 }),
        boundaryAlgorithm: Type.Literal('markdown-structure-v1'),
      },
      { additionalProperties: false },
    ),
    conversationSchemaVersion: Type.Integer({ minimum: 1 }),
    provenanceSchemaVersion: Type.Integer({ minimum: 1 }),
    projectIdentity: Type.Object(
      {
        policyVersion: Type.Integer({ minimum: 1 }),
        metadataSchemaVersion: Type.Integer({ minimum: 1 }),
        lineagePolicyVersion: Type.Integer({ minimum: 1 }),
        lineageDigest: Type.String({ pattern: '^[a-f0-9]{64}$' }),
      },
      { additionalProperties: false },
    ),
    sqliteRecallDatabase: Type.Object(
      {
        schemaVersion: Type.Literal(3),
        storageLayout: Type.Literal('unified-sqlite-vec'),
        sqliteVecVersion: Type.Literal('0.1.9'),
        embedding: Type.Object(
          {
            dimensions: Type.Literal(1_024),
            encoding: Type.Literal('fp32'),
            distanceMetric: Type.Literal('cosine'),
          },
          { additionalProperties: false },
        ),
        routing: Type.Object(
          {
            table: Type.Literal('bucketed'),
            bucketCount: Type.Literal(16),
            bucketFunction: Type.Literal('project-key-modulo-16'),
            projectExactKey: Type.Literal(true),
            global: Type.Literal('all-buckets'),
          },
          { additionalProperties: false },
        ),
        fullTextSearch: Type.Object(
          {
            engine: Type.Literal('fts5'),
            tokenizer: Type.Literal('unicode61'),
          },
          { additionalProperties: false },
        ),
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
      { fileName: identity.tokenizerJson.fileName, sha256: identity.tokenizerJson.sha256 },
      {
        fileName: identity.tokenizerConfigJson.fileName,
        sha256: identity.tokenizerConfigJson.sha256,
      },
    ],
  };
}

/** Creates the manifest expected by the configured Octen profile and current store code. */
export function createRecallIndexManifest(options: {
  embeddingIdentity: RecallEmbeddingModelIdentity;
  tokenizerIdentity?: RecallTokenizerManifestIdentity;
  chunkPolicy?: RecallChunkPolicy;
  projectLineages?: RecallProjectLineages;
}): RecallIndexManifest {
  if (options.embeddingIdentity.storedDimensions > options.embeddingIdentity.nativeDimensions) {
    throw new Error(
      `Recall embedding profile invalid: stored dimensions ${options.embeddingIdentity.storedDimensions} exceed native dimensions ${options.embeddingIdentity.nativeDimensions}`,
    );
  }
  const chunkPolicy = options.chunkPolicy ?? DEFAULT_RECALL_CHUNK_POLICY;
  assertRecallChunkPolicy(chunkPolicy);
  return {
    manifestVersion: RECALL_INDEX_MANIFEST_VERSION,
    importPolicy: { version: SESSION_IMPORT_POLICY_VERSION },
    embedding: { ...options.embeddingIdentity },
    tokenizer: structuredClone(
      options.tokenizerIdentity ?? createTokenizerManifestIdentity(OCTEN_TOKENIZER_IDENTITY),
    ),
    chunkPolicy: {
      version: 3,
      maxTokens: chunkPolicy.maxTokens,
      overlapTokens: chunkPolicy.overlapTokens,
      boundaryAlgorithm: 'markdown-structure-v1',
    },
    conversationSchemaVersion: SESSION_CONVERSATION_SCHEMA_VERSION,
    provenanceSchemaVersion: SESSION_CONVERSATION_SCHEMA_VERSION,
    projectIdentity: {
      policyVersion: PROJECT_IDENTITY_POLICY_VERSION,
      metadataSchemaVersion: PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
      lineagePolicyVersion: PROJECT_LINEAGE_POLICY_VERSION,
      lineageDigest: createLineageDigest(
        options.projectLineages ?? normalizeRecallProjectLineages({}),
      ),
    },
    sqliteRecallDatabase: structuredClone(SQLITE_RECALL_DATABASE_MANIFEST_IDENTITY),
  };
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
      collectManifestMismatches(
        actual[key],
        expectedValue,
        path ? `${path}.${key}` : key,
        mismatches,
      );
    }
    return;
  }
  if (!Object.is(actual, expected)) {
    mismatches.push(
      `${path}: expected ${formatManifestValue(expected)}, received ${formatManifestValue(actual)}`,
    );
  }
}

/** Rejects any stored identity that requires an explicit `psr index --rebuild`. */
export function assertRecallIndexManifestCompatible(
  actual: unknown,
  expected: RecallIndexManifest,
  manifestPath: string,
): asserts actual is RecallIndexManifest {
  if (!actual) {
    throw new Error(
      `Recall index manifest missing at ${manifestPath}; rebuild with psr index --rebuild`,
    );
  }
  const mismatches: string[] = [];
  collectManifestMismatches(actual, expected, '', mismatches);
  if (mismatches.length > 0) {
    throw new Error(
      `Recall index manifest incompatible at ${manifestPath}:\n- ${mismatches.join('\n- ')}\nRebuild with psr index --rebuild.`,
    );
  }
}

/** Validates that a stored manifest uses the current unified SQLite layout. */
export async function assertCurrentRecallIndexManifestLayout(manifestPath: string): Promise<void> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(manifestPath, 'utf8'));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall index manifest invalid at ${manifestPath}: ${message}. Rebuild with psr index --rebuild.`,
      { cause: error },
    );
  }
  if (!isUnknownRecord(parsed) || typeof parsed.manifestVersion !== 'number') {
    throw new Error(
      `Recall index manifest invalid at ${manifestPath}: manifestVersion is missing. Rebuild with psr index --rebuild.`,
    );
  }
  if (parsed.manifestVersion === RECALL_INDEX_MANIFEST_VERSION) {
    await readRecallIndexManifest(manifestPath);
    return;
  }
  throw new Error(
    `Recall index manifest version ${parsed.manifestVersion} at ${manifestPath} is incompatible; rebuild with psr index --rebuild.`,
  );
}

/** Reads and strictly validates the version 8 index manifest. */
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
      `Recall index manifest invalid at ${manifestPath}: ${message}. Rebuild with psr index --rebuild.`,
      { cause: error },
    );
  }
}

/** Writes the complete manifest through a unique temporary file and atomic rename. */
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
