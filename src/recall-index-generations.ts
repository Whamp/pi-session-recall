import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { readNodeErrorCode } from './read-node-error-code.js';

const RECALL_INDEX_GENERATION_SELECTION_VERSION = 1;
const RECALL_INDEX_GENERATION_ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,127}$/u;

const ACTIVE_RECALL_INDEX_GENERATION_SCHEMA = Type.Object(
  {
    version: Type.Literal(RECALL_INDEX_GENERATION_SELECTION_VERSION),
    generationId: Type.String({ pattern: '^[a-z0-9][a-z0-9-]{0,127}$' }),
    embeddingProfileId: Type.String({ minLength: 1 }),
    activatedAt: Type.String({ format: 'date-time' }),
  },
  { additionalProperties: false },
);

const STAGING_RECALL_INDEX_GENERATION_SCHEMA = Type.Object(
  {
    version: Type.Literal(RECALL_INDEX_GENERATION_SELECTION_VERSION),
    generationId: Type.String({ pattern: '^[a-z0-9][a-z0-9-]{0,127}$' }),
    embeddingProfileId: Type.String({ minLength: 1 }),
    createdAt: Type.String({ format: 'date-time' }),
    status: Type.Union([Type.Literal('building'), Type.Literal('resumable')]),
  },
  { additionalProperties: false },
);

/** The vector store, session state, manifest, and writer lock owned by one index generation. */
export interface RecallIndexGenerationPaths {
  databasePath: string;
  statePath: string;
  manifestPath: string;
  lockPath: string;
}

/** A selected legacy or managed index generation exposed by conversation-service status. */
export interface ActiveRecallIndexGenerationStatus {
  kind: 'legacy' | 'managed';
  generationId: string;
  embeddingProfileId: string | null;
  manifestPath: string;
}

/** Resumable staging generation exposed by conversation-service status. */
export interface StagingRecallIndexGenerationStatus {
  generationId: string;
  embeddingProfileId: string;
  manifestPath: string;
  status: 'building' | 'resumable';
}

/** Active and staging selection state without reading or opening either vector store. */
export interface RecallIndexGenerationStatus {
  active: ActiveRecallIndexGenerationStatus | null;
  staging: StagingRecallIndexGenerationStatus | null;
}

/** Configured legacy paths and managed-generation selection locations. */
export interface RecallIndexGenerationCoordinatorConfig {
  legacyPaths: RecallIndexGenerationPaths;
  generationsDirectory: string;
  activeGenerationPath: string;
  stagingGenerationPath: string;
}

interface ActiveRecallIndexGenerationSelection {
  version: 1;
  generationId: string;
  embeddingProfileId: string;
  activatedAt: string;
}

interface StagingRecallIndexGenerationSelection {
  version: 1;
  generationId: string;
  embeddingProfileId: string;
  createdAt: string;
  status: 'building' | 'resumable';
}

/** One staging generation selected for a build or resumed after interruption. */
export interface WritableStagingRecallIndexGeneration {
  generationId: string;
  paths: RecallIndexGenerationPaths;
  resumed: boolean;
}

function createManagedRecallIndexGenerationPaths(
  generationsDirectory: string,
  generationId: string,
): RecallIndexGenerationPaths {
  if (!RECALL_INDEX_GENERATION_ID_PATTERN.test(generationId)) {
    throw new Error(`Recall index generation ID invalid: ${generationId}`);
  }
  const generationDirectory = join(generationsDirectory, generationId);
  return {
    databasePath: join(generationDirectory, 'zvec'),
    statePath: join(generationDirectory, 'index-state.json'),
    manifestPath: join(generationDirectory, 'index-manifest.json'),
    lockPath: join(generationDirectory, 'operation.lock'),
  };
}

async function readRecallIndexGenerationSelection<T>(
  selectionPath: string,
  selectionName: string,
  parseSelection: (selection: unknown) => T,
): Promise<T | null> {
  try {
    const selection: unknown = JSON.parse(await readFile(selectionPath, 'utf8'));
    return parseSelection(selection);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall ${selectionName} generation selection invalid at ${selectionPath}: ${message}`,
      { cause: error },
    );
  }
}

async function writeRecallIndexGenerationSelection(
  selectionPath: string,
  selection: ActiveRecallIndexGenerationSelection | StagingRecallIndexGenerationSelection,
): Promise<void> {
  await mkdir(dirname(selectionPath), { recursive: true });
  const temporaryPath = `${selectionPath}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(selection)}\n`, 'utf8');
  await rename(temporaryPath, selectionPath);
}

async function readActiveRecallIndexGenerationSelection(
  config: RecallIndexGenerationCoordinatorConfig,
): Promise<ActiveRecallIndexGenerationSelection | null> {
  return readRecallIndexGenerationSelection(config.activeGenerationPath, 'active', (selection) =>
    Value.Parse(ACTIVE_RECALL_INDEX_GENERATION_SCHEMA, selection),
  );
}

async function readStagingRecallIndexGenerationSelection(
  config: RecallIndexGenerationCoordinatorConfig,
): Promise<StagingRecallIndexGenerationSelection | null> {
  return readRecallIndexGenerationSelection(config.stagingGenerationPath, 'staging', (selection) =>
    Value.Parse(STAGING_RECALL_INDEX_GENERATION_SCHEMA, selection),
  );
}

/** Resolves the selected active generation, falling back to an intact pre-generation layout. */
export async function resolveActiveRecallIndexGeneration(
  config: RecallIndexGenerationCoordinatorConfig,
): Promise<{
  status: ActiveRecallIndexGenerationStatus;
  paths: RecallIndexGenerationPaths;
} | null> {
  const active = await readActiveRecallIndexGenerationSelection(config);
  if (active) {
    const paths = createManagedRecallIndexGenerationPaths(
      config.generationsDirectory,
      active.generationId,
    );
    return {
      status: {
        kind: 'managed',
        generationId: active.generationId,
        embeddingProfileId: active.embeddingProfileId,
        manifestPath: paths.manifestPath,
      },
      paths,
    };
  }
  if (existsSync(config.legacyPaths.manifestPath)) {
    return {
      status: {
        kind: 'legacy',
        generationId: 'legacy',
        embeddingProfileId: null,
        manifestPath: config.legacyPaths.manifestPath,
      },
      paths: config.legacyPaths,
    };
  }
  return null;
}

/** Reads active and resumable staging generation selection without opening zvec. */
export async function readRecallIndexGenerationStatus(
  config: RecallIndexGenerationCoordinatorConfig,
): Promise<RecallIndexGenerationStatus> {
  const [active, staging] = await Promise.all([
    resolveActiveRecallIndexGeneration(config),
    readStagingRecallIndexGenerationSelection(config),
  ]);
  return {
    active: active?.status ?? null,
    staging: staging
      ? {
          generationId: staging.generationId,
          embeddingProfileId: staging.embeddingProfileId,
          manifestPath: createManagedRecallIndexGenerationPaths(
            config.generationsDirectory,
            staging.generationId,
          ).manifestPath,
          status: staging.status,
        }
      : null,
  };
}

/** Selects one profile-bound staging generation, reusing it after interruption. */
export async function prepareStagingRecallIndexGeneration(
  config: RecallIndexGenerationCoordinatorConfig,
  embeddingProfileId: string,
): Promise<WritableStagingRecallIndexGeneration> {
  let existing = await readStagingRecallIndexGenerationSelection(config);
  const active = await readActiveRecallIndexGenerationSelection(config);
  if (existing && active?.generationId === existing.generationId) {
    await rm(config.stagingGenerationPath, { force: true });
    existing = null;
  }
  if (existing) {
    if (existing.embeddingProfileId !== embeddingProfileId) {
      throw new Error(
        `Recall staging generation uses embedding profile ${existing.embeddingProfileId}, not ${embeddingProfileId}; discard it explicitly before building another profile`,
      );
    }
    await writeRecallIndexGenerationSelection(config.stagingGenerationPath, {
      ...existing,
      status: 'building',
    });
    return {
      generationId: existing.generationId,
      paths: createManagedRecallIndexGenerationPaths(
        config.generationsDirectory,
        existing.generationId,
      ),
      resumed: true,
    };
  }

  const generationId = `generation-${randomUUID()}`;
  const selection: StagingRecallIndexGenerationSelection = {
    version: RECALL_INDEX_GENERATION_SELECTION_VERSION,
    generationId,
    embeddingProfileId,
    createdAt: new Date().toISOString(),
    status: 'building',
  };
  await mkdir(join(config.generationsDirectory, generationId), { recursive: true });
  await writeRecallIndexGenerationSelection(config.stagingGenerationPath, selection);
  return {
    generationId,
    paths: createManagedRecallIndexGenerationPaths(config.generationsDirectory, generationId),
    resumed: false,
  };
}

/** Marks an interrupted or failed staging generation as resumable without changing active recall. */
export async function preserveStagingRecallIndexGeneration(
  config: RecallIndexGenerationCoordinatorConfig,
  generationId: string,
): Promise<void> {
  const staging = await readStagingRecallIndexGenerationSelection(config);
  if (!staging || staging.generationId !== generationId) {
    return;
  }
  await writeRecallIndexGenerationSelection(config.stagingGenerationPath, {
    ...staging,
    status: 'resumable',
  });
}

/** Atomically selects a validated staging generation and clears only its staging marker. */
export async function activateStagingRecallIndexGeneration(
  config: RecallIndexGenerationCoordinatorConfig,
  generationId: string,
  embeddingProfileId: string,
): Promise<void> {
  const staging = await readStagingRecallIndexGenerationSelection(config);
  if (
    !staging ||
    staging.generationId !== generationId ||
    staging.embeddingProfileId !== embeddingProfileId
  ) {
    throw new Error(
      `Recall staging generation ${generationId} is no longer selected for activation`,
    );
  }
  await writeRecallIndexGenerationSelection(config.activeGenerationPath, {
    version: RECALL_INDEX_GENERATION_SELECTION_VERSION,
    generationId,
    embeddingProfileId,
    activatedAt: new Date().toISOString(),
  });
  await rm(config.stagingGenerationPath, { force: true });
}

/** Explicitly removes abandoned staging work without touching the selected active generation. */
export async function discardStagingRecallIndexGeneration(
  config: RecallIndexGenerationCoordinatorConfig,
): Promise<boolean> {
  const staging = await readStagingRecallIndexGenerationSelection(config);
  if (!staging) {
    return false;
  }
  const active = await readActiveRecallIndexGenerationSelection(config);
  if (active?.generationId === staging.generationId) {
    await rm(config.stagingGenerationPath, { force: true });
    return false;
  }
  await rm(join(config.generationsDirectory, staging.generationId), {
    recursive: true,
    force: true,
  });
  await rm(config.stagingGenerationPath, { force: true });
  return true;
}
