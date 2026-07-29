import { createHash, randomUUID } from 'node:crypto';
import { access, mkdir, open, readFile, rename, stat } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';
import { adoptLegacyRecallGenerationTransition } from './recall-generation-transitions.js';
import { readLegacyRecallIndexManifestV5 } from './recall-index-manifest.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { syncRecallDirectory } from './sync-recall-directory.js';

/** Exact pre-generation version-5 paths accepted only by explicit legacy adoption. */
export interface AdoptLegacyRecallGenerationOptions {
  dataDirectory: string;
  legacyDatabasePath: string;
  legacyStatePath: string;
  legacyManifestPath: string;
  generationRootDirectory: string;
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  backupEvidencePath: string;
  lockPath: string;
  signal?: AbortSignal;
  nowEpochMilliseconds?: () => number;
  validateLegacyDatabase(databasePath: string): Promise<void>;
}

/** Adopted read-only generation identity and durable backup evidence location. */
export interface AdoptLegacyRecallGenerationResult {
  generationId: string;
  generationDirectory: string;
  backupEvidencePath: string;
}

interface LegacyAdoptionJournal {
  version: 1;
  state: 'prepared' | 'completed';
  generationId: string;
  stagingDirectory: string;
  layoutDigest: string;
  indexManifestFingerprint: string;
  adoptedAtEpochMilliseconds: number;
  sourceEntries: ['zvec', 'index-state.json', 'index-manifest.json'];
}

const legacyIndexStateSchema = Type.Object(
  {
    version: Type.Literal(2),
    importPolicyVersion: Type.Literal(SESSION_IMPORT_POLICY_VERSION),
    sessions: Type.Record(
      Type.String(),
      Type.Object(
        {
          size: Type.Number({ minimum: 0 }),
          mtimeMs: Type.Number({ minimum: 0 }),
          chunks: Type.Array(Type.Object({ id: Type.String() }, { additionalProperties: false })),
        },
        { additionalProperties: false },
      ),
    ),
  },
  { additionalProperties: false },
);
const legacyAdoptionJournalSchema = Type.Object(
  {
    version: Type.Literal(1),
    state: Type.Union([Type.Literal('prepared'), Type.Literal('completed')]),
    generationId: Type.String({ pattern: '^legacy-[a-f0-9]{24}$' }),
    stagingDirectory: Type.String({ minLength: 1 }),
    layoutDigest: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    indexManifestFingerprint: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    adoptedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
    sourceEntries: Type.Tuple([
      Type.Literal('zvec'),
      Type.Literal('index-state.json'),
      Type.Literal('index-manifest.json'),
    ]),
  },
  { additionalProperties: false },
);

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function writeDurableLegacyAdoptionEvidence(
  evidencePath: string,
  evidence: LegacyAdoptionJournal,
): Promise<void> {
  Value.Parse(legacyAdoptionJournalSchema, evidence);
  await mkdir(dirname(evidencePath), { recursive: true });
  const temporaryPath = `${evidencePath}.${randomUUID()}.tmp`;
  const file = await open(temporaryPath, 'wx', 0o600);
  try {
    await file.writeFile(`${JSON.stringify(evidence)}\n`, 'utf8');
    await file.sync();
  } finally {
    await file.close();
  }
  await rename(temporaryPath, evidencePath);
  await syncRecallDirectory(dirname(evidencePath));
}

async function readLegacyAdoptionJournal(
  evidencePath: string,
): Promise<LegacyAdoptionJournal | null> {
  try {
    const source = await readFile(evidencePath, 'utf8');
    return Value.Parse(legacyAdoptionJournalSchema, JSON.parse(source));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall legacy adoption journal invalid at ${evidencePath}: ${message}`, {
      cause: error,
    });
  }
}

async function assertLegacyAdoptionFilesystem(
  options: AdoptLegacyRecallGenerationOptions,
): Promise<void> {
  await mkdir(options.generationRootDirectory, { recursive: true });
  const [dataStats, generationStats, databaseStats, stateStats, manifestStats] = await Promise.all([
    stat(options.dataDirectory),
    stat(options.generationRootDirectory),
    stat(options.legacyDatabasePath),
    stat(options.legacyStatePath),
    stat(options.legacyManifestPath),
  ]);
  if (!databaseStats.isDirectory() || !stateStats.isFile() || !manifestStats.isFile()) {
    throw new Error('Recall legacy adoption requires zvec directory, index state, and manifest');
  }
  const expectedDevice = dataStats.dev;
  if (
    generationStats.dev !== expectedDevice ||
    databaseStats.dev !== expectedDevice ||
    stateStats.dev !== expectedDevice ||
    manifestStats.dev !== expectedDevice
  ) {
    throw new Error('Recall legacy adoption requires source and generations on one filesystem');
  }
}

function assertLegacyAdoptionJournalPaths(
  options: AdoptLegacyRecallGenerationOptions,
  journal: LegacyAdoptionJournal,
): string {
  const generationRootDirectory = resolve(options.generationRootDirectory);
  const generationDirectory = join(generationRootDirectory, journal.generationId);
  const expectedStagingPrefix = `.${journal.generationId}.`;
  if (
    resolve(dirname(journal.stagingDirectory)) !== generationRootDirectory ||
    !basename(journal.stagingDirectory).startsWith(expectedStagingPrefix) ||
    !basename(journal.stagingDirectory).endsWith('.staging')
  ) {
    throw new Error('Recall legacy adoption journal staging path escapes generation root');
  }
  return generationDirectory;
}

async function relocateLegacyAdoptionEntries(
  options: AdoptLegacyRecallGenerationOptions,
  journal: LegacyAdoptionJournal,
  generationDirectory: string,
): Promise<void> {
  if (await pathExists(generationDirectory)) {
    return;
  }
  await mkdir(journal.stagingDirectory, { recursive: true });
  const relocations = [
    { source: options.legacyDatabasePath, destination: join(journal.stagingDirectory, 'zvec') },
    {
      source: options.legacyStatePath,
      destination: join(journal.stagingDirectory, 'index-state.json'),
    },
    {
      source: options.legacyManifestPath,
      destination: join(journal.stagingDirectory, 'index-manifest.json'),
    },
  ];
  for (const relocation of relocations) {
    const [sourceExists, destinationExists] = await Promise.all([
      pathExists(relocation.source),
      pathExists(relocation.destination),
    ]);
    if (sourceExists && !destinationExists) {
      await rename(relocation.source, relocation.destination);
      continue;
    }
    if (!sourceExists && destinationExists) {
      continue;
    }
    throw new Error('Recall legacy adoption journal found an ambiguous relocation state');
  }
  await syncRecallDirectory(options.dataDirectory);
  await syncRecallDirectory(journal.stagingDirectory);
  await rename(journal.stagingDirectory, generationDirectory);
  await syncRecallDirectory(options.generationRootDirectory);
}

async function completeLegacyAdoptionJournal(
  options: AdoptLegacyRecallGenerationOptions,
  journal: LegacyAdoptionJournal,
): Promise<AdoptLegacyRecallGenerationResult> {
  const generationDirectory = assertLegacyAdoptionJournalPaths(options, journal);
  await relocateLegacyAdoptionEntries(options, journal, generationDirectory);
  const manifestPath = join(generationDirectory, 'index-manifest.json');
  await readLegacyRecallIndexManifestV5(manifestPath);
  await options.validateLegacyDatabase(join(generationDirectory, 'zvec'));
  await adoptLegacyRecallGenerationTransition({
    activeGenerationPointerPath: options.activeGenerationPointerPath,
    generationRegistryPath: options.generationRegistryPath,
    backlogSummaryPath: options.backlogSummaryPath,
    generationId: journal.generationId,
    indexManifestFingerprint: journal.indexManifestFingerprint,
    adoptedAtEpochMilliseconds: journal.adoptedAtEpochMilliseconds,
  });
  await writeDurableLegacyAdoptionEvidence(options.backupEvidencePath, {
    ...journal,
    state: 'completed',
  });
  return {
    generationId: journal.generationId,
    generationDirectory,
    backupEvidencePath: options.backupEvidencePath,
  };
}

/** Explicitly validates or resumes exact version-5 root-layout adoption without reading sessions. */
export async function adoptLegacyRecallGeneration(
  options: AdoptLegacyRecallGenerationOptions,
): Promise<AdoptLegacyRecallGenerationResult> {
  return coordinateRecallWriteWindow(
    {
      lockPath: options.lockPath,
      allowRecovery: true,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async (writeWindow) => {
      let journal = await readLegacyAdoptionJournal(options.backupEvidencePath);
      let journalPrepared = journal !== null;
      try {
        if (journal !== null) {
          return await completeLegacyAdoptionJournal(options, journal);
        }
        const [existingPointer, existingRegistry] = await Promise.all([
          readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
          readRecallGenerationRegistry(options.generationRegistryPath),
        ]);
        if (existingPointer || existingRegistry) {
          throw new Error(
            'Recall legacy adoption refused because generation management is initialized',
          );
        }
        await assertLegacyAdoptionFilesystem(options);
        const [manifestSource, stateSource] = await Promise.all([
          readFile(options.legacyManifestPath, 'utf8'),
          readFile(options.legacyStatePath, 'utf8'),
        ]);
        await readLegacyRecallIndexManifestV5(options.legacyManifestPath);
        try {
          Value.Parse(legacyIndexStateSchema, JSON.parse(stateSource));
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(
            `Recall legacy index state invalid at ${options.legacyStatePath}: ${message}`,
            { cause: error },
          );
        }
        await options.validateLegacyDatabase(options.legacyDatabasePath);
        const layoutDigest = createHash('sha256')
          .update(manifestSource)
          .update('\0')
          .update(stateSource)
          .digest('hex');
        const generationId = `legacy-${layoutDigest.slice(0, 24)}`;
        journal = {
          version: 1,
          state: 'prepared',
          generationId,
          stagingDirectory: join(
            options.generationRootDirectory,
            `.${generationId}.${randomUUID()}.staging`,
          ),
          layoutDigest,
          indexManifestFingerprint: createHash('sha256').update(manifestSource).digest('hex'),
          adoptedAtEpochMilliseconds: options.nowEpochMilliseconds?.() ?? Date.now(),
          sourceEntries: ['zvec', 'index-state.json', 'index-manifest.json'],
        };
        await writeDurableLegacyAdoptionEvidence(options.backupEvidencePath, journal);
        journalPrepared = true;
        return await completeLegacyAdoptionJournal(options, journal);
      } catch (error) {
        if (journalPrepared || writeWindow.recovering) {
          writeWindow.retainRecoveryRequired();
        }
        throw error;
      }
    },
  );
}
