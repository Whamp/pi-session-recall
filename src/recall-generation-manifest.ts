import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  DEFAULT_RECALL_CHUNK_POLICY,
  type RecallEmbeddingModelIdentity,
} from './recall-index-manifest.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import type { RecallEmbeddingModelProfile } from './recall-model-profiles.js';
import {
  createLineageDigest,
  PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
  PROJECT_IDENTITY_POLICY_VERSION,
  PROJECT_LINEAGE_POLICY_VERSION,
  type RecallProjectLineages,
} from './resolve-project-identity.js';
import { SESSION_CONVERSATION_SCHEMA_VERSION } from './session-conversation-index.js';
import {
  createRecallGenerationStoreContracts,
  RECALL_GENERATION_STORE_FORMAT_VERSION,
  type RecallGenerationScalarFieldContract,
  type RecallGenerationStoreContract,
  type RecallGenerationVectorFieldContract,
} from './recall-generation-stores.js';

/** Current fixed contract version for one coherent three-store recall generation. */
export const RECALL_GENERATION_FORMAT_VERSION = 1;

/** Fixed import, storage, source, profile, and validation identity written before store content. */
export interface RecallGenerationManifest {
  generationFormatVersion: 1;
  generationId: string;
  stores: Readonly<{
    lexicalSource: Readonly<RecallGenerationStoreContract>;
    dense: Readonly<RecallGenerationStoreContract>;
    sessionProjection: Readonly<RecallGenerationStoreContract>;
  }>;
  importPolicy: Readonly<{ version: number; framing: 'lf-byte-v1'; validation: 'strict-graph-v1' }>;
  chunkPolicy: Readonly<{
    version: 2;
    maximumTokens: number;
    overlapTokens: number;
    boundaryAlgorithm: 'markdown-structure-v1';
  }>;
  provenancePolicy: Readonly<{
    version: number;
    physicalSourceIdentity: 'sessions-root-relative-path-v1';
    logicalSessionOccurrenceIdentity: 'physical-source-and-complete-header-position-v1';
    evidenceOccurrenceIdentity: 'exact-source-location-v1';
    branchPathMembership: 'all-endpoint-paths-v1';
  }>;
  projectIdentityPolicy: Readonly<{
    policyVersion: number;
    metadataSchemaVersion: number;
    lineagePolicyVersion: number;
    lineageDigest: string;
  }>;
  sourceAnchorPolicy: Readonly<{
    version: 1;
    owningStore: 'lexical-source';
    parentLinks: 'exact-entry-parent-v1';
    forwardPath: 'explicit-leaf-at-fork-v1';
  }>;
  embeddingProfile: Readonly<{
    profileId: string;
    modelIdentity: Readonly<RecallEmbeddingModelIdentity>;
    nativeDimensions: number;
    storedDimensions: number;
    reduction: 'none';
    normalization: 'l2' | 'none';
    queryInputPrefix: string;
    documentInputPrefix: string;
  }>;
  validationPolicy: Readonly<{
    version: 1;
    membership: 'exact-v1';
    closeAndReopenRequired: true;
    unresolvedRecovery: 'reject';
    canaries: readonly ['store-schema-v1', 'empty-membership-v1'];
  }>;
}

const generationIdSchema = Type.String({ pattern: '^[A-Za-z0-9_-]+$' });
const checksumSchema = Type.String({ pattern: '^[a-f0-9]{64}$' });
const scalarFieldSchema = Type.Object(
  {
    name: Type.String({ minLength: 1 }),
    type: Type.Union([
      Type.Literal('string'),
      Type.Literal('boolean'),
      Type.Literal('int32'),
      Type.Literal('int64'),
      Type.Literal('array-string'),
    ]),
    index: Type.Optional(
      Type.Object(
        {
          kind: Type.Literal('full-text'),
          tokenizer: Type.Literal('standard'),
          filters: Type.Array(Type.String()),
        },
        { additionalProperties: false },
      ),
    ),
  },
  { additionalProperties: false },
);
const vectorFieldSchema = Type.Object(
  {
    name: Type.String({ minLength: 1 }),
    type: Type.Literal('vector-fp32'),
    dimensions: Type.Integer({ minimum: 1 }),
    index: Type.Object(
      {
        kind: Type.Literal('hnsw'),
        metric: Type.Literal('cosine'),
        m: Type.Integer({ minimum: 1 }),
        efConstruction: Type.Integer({ minimum: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);
const storeContractSchema = Type.Object(
  {
    formatVersion: Type.Literal(RECALL_GENERATION_STORE_FORMAT_VERSION),
    directory: Type.String({ minLength: 1 }),
    collectionName: Type.String({ minLength: 1 }),
    responsibility: Type.Union([
      Type.Literal('lexical-source'),
      Type.Literal('dense-evidence'),
      Type.Literal('session-projection'),
    ]),
    scalarFields: Type.Array(scalarFieldSchema),
    vectorFields: Type.Array(vectorFieldSchema),
  },
  { additionalProperties: false },
);
const embeddingModelIdentitySchema = Type.Object(
  {
    requestModel: Type.String({ minLength: 1 }),
    servedModelId: Type.String({ minLength: 1 }),
    artifact: Type.String({ minLength: 1 }),
    artifactRepository: Type.Optional(Type.String({ minLength: 1 })),
    artifactRevision: Type.Optional(Type.String({ minLength: 1 })),
    artifactSha256: Type.Optional(checksumSchema),
    dimensions: Type.Integer({ minimum: 1 }),
    quantization: Type.String({ minLength: 1 }),
    pooling: Type.String({ minLength: 1 }),
    normalization: Type.Optional(Type.Literal('l2')),
  },
  { additionalProperties: false },
);
const recallGenerationManifestSchema = Type.Object(
  {
    generationFormatVersion: Type.Literal(RECALL_GENERATION_FORMAT_VERSION),
    generationId: generationIdSchema,
    stores: Type.Object(
      {
        lexicalSource: storeContractSchema,
        dense: storeContractSchema,
        sessionProjection: storeContractSchema,
      },
      { additionalProperties: false },
    ),
    importPolicy: Type.Object(
      {
        version: Type.Literal(SESSION_IMPORT_POLICY_VERSION),
        framing: Type.Literal('lf-byte-v1'),
        validation: Type.Literal('strict-graph-v1'),
      },
      { additionalProperties: false },
    ),
    chunkPolicy: Type.Object(
      {
        version: Type.Literal(2),
        maximumTokens: Type.Integer({ minimum: 1 }),
        overlapTokens: Type.Integer({ minimum: 0 }),
        boundaryAlgorithm: Type.Literal('markdown-structure-v1'),
      },
      { additionalProperties: false },
    ),
    provenancePolicy: Type.Object(
      {
        version: Type.Literal(SESSION_CONVERSATION_SCHEMA_VERSION),
        physicalSourceIdentity: Type.Literal('sessions-root-relative-path-v1'),
        logicalSessionOccurrenceIdentity: Type.Literal(
          'physical-source-and-complete-header-position-v1',
        ),
        evidenceOccurrenceIdentity: Type.Literal('exact-source-location-v1'),
        branchPathMembership: Type.Literal('all-endpoint-paths-v1'),
      },
      { additionalProperties: false },
    ),
    projectIdentityPolicy: Type.Object(
      {
        policyVersion: Type.Literal(PROJECT_IDENTITY_POLICY_VERSION),
        metadataSchemaVersion: Type.Literal(PROJECT_IDENTITY_METADATA_SCHEMA_VERSION),
        lineagePolicyVersion: Type.Literal(PROJECT_LINEAGE_POLICY_VERSION),
        lineageDigest: checksumSchema,
      },
      { additionalProperties: false },
    ),
    sourceAnchorPolicy: Type.Object(
      {
        version: Type.Literal(1),
        owningStore: Type.Literal('lexical-source'),
        parentLinks: Type.Literal('exact-entry-parent-v1'),
        forwardPath: Type.Literal('explicit-leaf-at-fork-v1'),
      },
      { additionalProperties: false },
    ),
    embeddingProfile: Type.Object(
      {
        profileId: Type.String({ minLength: 1 }),
        modelIdentity: embeddingModelIdentitySchema,
        nativeDimensions: Type.Integer({ minimum: 1 }),
        storedDimensions: Type.Integer({ minimum: 1 }),
        reduction: Type.Literal('none'),
        normalization: Type.Union([Type.Literal('l2'), Type.Literal('none')]),
        queryInputPrefix: Type.String(),
        documentInputPrefix: Type.String(),
      },
      { additionalProperties: false },
    ),
    validationPolicy: Type.Object(
      {
        version: Type.Literal(1),
        membership: Type.Literal('exact-v1'),
        closeAndReopenRequired: Type.Literal(true),
        unresolvedRecovery: Type.Literal('reject'),
        canaries: Type.Tuple([
          Type.Literal('store-schema-v1'),
          Type.Literal('empty-membership-v1'),
        ]),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

function cloneScalarField(
  field: Readonly<RecallGenerationScalarFieldContract>,
): RecallGenerationScalarFieldContract {
  return {
    name: field.name,
    type: field.type,
    ...(field.index
      ? {
          index: {
            kind: field.index.kind,
            tokenizer: field.index.tokenizer,
            filters: [...field.index.filters],
          },
        }
      : {}),
  };
}

function cloneVectorField(
  field: Readonly<RecallGenerationVectorFieldContract>,
): RecallGenerationVectorFieldContract {
  return {
    name: field.name,
    type: field.type,
    dimensions: field.dimensions,
    index: { ...field.index },
  };
}

function cloneStoreContract(
  contract: Readonly<RecallGenerationStoreContract>,
): RecallGenerationStoreContract {
  return {
    formatVersion: RECALL_GENERATION_STORE_FORMAT_VERSION,
    directory: contract.directory,
    collectionName: contract.collectionName,
    responsibility: contract.responsibility,
    scalarFields: contract.scalarFields.map(cloneScalarField),
    vectorFields: contract.vectorFields.map(cloneVectorField),
  };
}

/** Builds the immutable generation manifest from configured semantics without reading session data. */
export function createRecallGenerationManifest(options: {
  generationId: string;
  embeddingProfileId: string;
  embeddingProfile: Readonly<RecallEmbeddingModelProfile>;
  projectLineages: RecallProjectLineages;
  chunkPolicy?: Readonly<RecallChunkPolicy>;
}): RecallGenerationManifest {
  const chunkPolicy = options.chunkPolicy ?? DEFAULT_RECALL_CHUNK_POLICY;
  if (chunkPolicy.overlapTokens >= chunkPolicy.maxTokens) {
    throw new Error(
      'Recall coherent generation chunk policy invalid: overlap must be smaller than maximum tokens',
    );
  }
  const dimensions = options.embeddingProfile.identity.dimensions;
  const stores = createRecallGenerationStoreContracts(options.generationId, dimensions);
  const manifest: RecallGenerationManifest = {
    generationFormatVersion: RECALL_GENERATION_FORMAT_VERSION,
    generationId: options.generationId,
    stores: {
      lexicalSource: cloneStoreContract(stores.lexicalSource),
      dense: cloneStoreContract(stores.dense),
      sessionProjection: cloneStoreContract(stores.sessionProjection),
    },
    importPolicy: {
      version: SESSION_IMPORT_POLICY_VERSION,
      framing: 'lf-byte-v1',
      validation: 'strict-graph-v1',
    },
    chunkPolicy: {
      version: 2,
      maximumTokens: chunkPolicy.maxTokens,
      overlapTokens: chunkPolicy.overlapTokens,
      boundaryAlgorithm: 'markdown-structure-v1',
    },
    provenancePolicy: {
      version: SESSION_CONVERSATION_SCHEMA_VERSION,
      physicalSourceIdentity: 'sessions-root-relative-path-v1',
      logicalSessionOccurrenceIdentity: 'physical-source-and-complete-header-position-v1',
      evidenceOccurrenceIdentity: 'exact-source-location-v1',
      branchPathMembership: 'all-endpoint-paths-v1',
    },
    projectIdentityPolicy: {
      policyVersion: PROJECT_IDENTITY_POLICY_VERSION,
      metadataSchemaVersion: PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
      lineagePolicyVersion: PROJECT_LINEAGE_POLICY_VERSION,
      lineageDigest: createLineageDigest(options.projectLineages),
    },
    sourceAnchorPolicy: {
      version: 1,
      owningStore: 'lexical-source',
      parentLinks: 'exact-entry-parent-v1',
      forwardPath: 'explicit-leaf-at-fork-v1',
    },
    embeddingProfile: {
      profileId: options.embeddingProfileId,
      modelIdentity: structuredClone(options.embeddingProfile.identity),
      nativeDimensions: dimensions,
      storedDimensions: dimensions,
      reduction: 'none',
      normalization: options.embeddingProfile.identity.normalization ?? 'none',
      queryInputPrefix: options.embeddingProfile.queryInputPrefix,
      documentInputPrefix: options.embeddingProfile.documentInputPrefix,
    },
    validationPolicy: {
      version: 1,
      membership: 'exact-v1',
      closeAndReopenRequired: true,
      unresolvedRecovery: 'reject',
      canaries: ['store-schema-v1', 'empty-membership-v1'],
    },
  };
  return Value.Parse(recallGenerationManifestSchema, manifest);
}

/** Encodes one strict fixed manifest with stable formatting for receipt fingerprinting. */
export function encodeRecallGenerationManifest(manifest: RecallGenerationManifest): string {
  const validated = Value.Parse(recallGenerationManifestSchema, manifest);
  return `${JSON.stringify(validated, null, 2)}\n`;
}

/** Calculates the SHA-256 fingerprint of the exact immutable manifest bytes. */
export function calculateRecallGenerationManifestFingerprint(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}

/** Writes the fixed manifest exactly once before any generation store content. */
export async function writeRecallGenerationManifest(
  manifestPath: string,
  manifest: RecallGenerationManifest,
): Promise<string> {
  const content = encodeRecallGenerationManifest(manifest);
  await writeFile(manifestPath, content, { encoding: 'utf8', flag: 'wx' });
  return calculateRecallGenerationManifestFingerprint(content);
}

/** Reads one generation manifest through its strict, unknown-field-rejecting contract. */
export async function readRecallGenerationManifest(
  manifestPath: string,
): Promise<{ manifest: RecallGenerationManifest; content: string; fingerprint: string }> {
  let content: string;
  try {
    content = await readFile(manifestPath, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation manifest unreadable at ${manifestPath}: ${message}`,
      {
        cause: error,
      },
    );
  }
  try {
    const parsed: unknown = JSON.parse(content);
    const manifest = Value.Parse(recallGenerationManifestSchema, parsed);
    return {
      manifest,
      content,
      fingerprint: calculateRecallGenerationManifestFingerprint(content),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall coherent generation manifest invalid at ${manifestPath}: ${message}`, {
      cause: error,
    });
  }
}

/** Rejects a strict but configured-incompatible generation manifest before stores are opened. */
export function assertRecallGenerationManifestCompatible(
  actual: RecallGenerationManifest,
  expected: RecallGenerationManifest,
  manifestPath: string,
): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`Recall coherent generation manifest incompatible at ${manifestPath}`);
  }
}
