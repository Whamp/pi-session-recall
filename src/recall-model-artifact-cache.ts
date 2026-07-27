import { randomUUID } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { access, mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallModelArtifactSource } from './recall-model-profiles.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { validateRecallGgufModelArtifact } from './validate-recall-gguf-model-artifact.js';

const RECALL_MODEL_ARTIFACT_RECEIPT_SCHEMA_VERSION = 1;
const recallModelArtifactReceiptSchema = Type.Object(
  {
    schemaVersion: Type.Literal(RECALL_MODEL_ARTIFACT_RECEIPT_SCHEMA_VERSION),
    profileId: Type.String(),
    repository: Type.String(),
    revision: Type.String(),
    artifact: Type.String(),
    byteSize: Type.Integer({ minimum: 1 }),
    sha256: Type.String(),
  },
  { additionalProperties: false },
);

/** Downloadable model identity needed by the artifact cache, independent of inference execution. */
export interface RecallDownloadableModelProfile {
  profileId: string;
  source: Readonly<RecallModelArtifactSource>;
}

/** Fixed-source transport that writes one immutable model artifact to the requested path. */
export interface RecallModelArtifactTransport {
  downloadArtifact(sourceUrl: string, destinationPath: string, signal?: AbortSignal): Promise<void>;
}

/** Creates the built-in streaming HTTPS transport used only after model download approval. */
export function createRecallHttpsModelArtifactTransport(): RecallModelArtifactTransport {
  return {
    async downloadArtifact(sourceUrl, destinationPath, signal) {
      const response = await fetch(sourceUrl, {
        redirect: 'follow',
        ...(signal ? { signal } : {}),
      });
      if (!response.ok) {
        throw new Error(
          `Recall model artifact download failed: HTTP ${response.status} ${response.statusText}`,
        );
      }
      if (!response.body) {
        throw new Error('Recall model artifact download failed: response body is missing');
      }
      await pipeline(
        Readable.from(response.body),
        createWriteStream(destinationPath, { flags: 'wx', mode: 0o600 }),
        { signal },
      );
    },
  };
}

/** Explicit consent required before model artifact transport or removal can touch the cache. */
export interface RecallModelArtifactApproval {
  approved: boolean;
  signal?: AbortSignal;
}

/** Operator-visible state of one pinned model artifact and its actionable repair. */
export interface RecallModelArtifactStatus {
  state: 'missing' | 'partial' | 'corrupt' | 'valid' | 'incompatible';
  artifactPath: string;
  partialPaths: readonly string[];
  issue?: string;
  repair: string;
}

/** Profile metadata and current cache status returned by model inspection. */
export interface RecallModelArtifactInspection {
  profile: Readonly<RecallDownloadableModelProfile>;
  status: RecallModelArtifactStatus;
}

/** Health conclusion and repair action returned by the model doctor. */
export interface RecallModelArtifactDiagnosis {
  healthy: boolean;
  status: RecallModelArtifactStatus;
  action: string;
}

/** Public artifact operations used by deterministic model setup and doctor commands. */
export interface RecallModelArtifactCache {
  inspectArtifact(): Promise<RecallModelArtifactInspection>;
  verifyArtifact(): Promise<RecallModelArtifactStatus>;
  diagnoseArtifact(): Promise<RecallModelArtifactDiagnosis>;
  downloadArtifact(approval: RecallModelArtifactApproval): Promise<RecallModelArtifactStatus>;
  repairArtifact(approval: RecallModelArtifactApproval): Promise<RecallModelArtifactStatus>;
  removeArtifact(approval: RecallModelArtifactApproval): Promise<RecallModelArtifactStatus>;
}

/** Dependencies for one profile-specific model artifact cache. */
export interface RecallModelArtifactCacheOptions {
  cacheDirectory: string;
  profile: Readonly<RecallDownloadableModelProfile>;
  transport?: RecallModelArtifactTransport;
}

interface RecallModelArtifactReceipt {
  schemaVersion: number;
  profileId: string;
  repository: string;
  revision: string;
  artifact: string;
  byteSize: number;
  sha256: string;
}

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

function createExpectedArtifactReceipt(
  profile: Readonly<RecallDownloadableModelProfile>,
): RecallModelArtifactReceipt {
  return {
    schemaVersion: RECALL_MODEL_ARTIFACT_RECEIPT_SCHEMA_VERSION,
    profileId: profile.profileId,
    repository: profile.source.repository,
    revision: profile.source.revision,
    artifact: profile.source.artifact,
    byteSize: profile.source.byteSize,
    sha256: profile.source.sha256,
  };
}

function isExpectedArtifactReceipt(
  receipt: RecallModelArtifactReceipt,
  expected: RecallModelArtifactReceipt,
): boolean {
  return (
    receipt.schemaVersion === expected.schemaVersion &&
    receipt.profileId === expected.profileId &&
    receipt.repository === expected.repository &&
    receipt.revision === expected.revision &&
    receipt.artifact === expected.artifact &&
    receipt.byteSize === expected.byteSize &&
    receipt.sha256 === expected.sha256
  );
}

async function listPartialArtifactPaths(
  artifactDirectory: string,
  artifactName: string,
): Promise<string[]> {
  try {
    const partialPrefix = `${artifactName}.partial-`;
    return (await readdir(artifactDirectory))
      .filter((name) => name.startsWith(partialPrefix))
      .sort()
      .map((name) => join(artifactDirectory, name));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
}

async function writeArtifactReceiptAtomically(
  receiptPath: string,
  receipt: RecallModelArtifactReceipt,
): Promise<void> {
  const temporaryPath = `${receiptPath}.partial-${randomUUID()}`;
  await writeFile(temporaryPath, `${JSON.stringify(receipt, undefined, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  });
  await rename(temporaryPath, receiptPath);
}

/** Creates explicit model artifact operations without downloading until approval is supplied. */
export function createRecallModelArtifactCache(
  options: RecallModelArtifactCacheOptions,
): RecallModelArtifactCache {
  const artifactPath = join(
    options.cacheDirectory,
    options.profile.profileId,
    options.profile.source.revision,
    options.profile.source.artifact,
  );
  const artifactDirectory = dirname(artifactPath);
  const profileDirectory = join(options.cacheDirectory, options.profile.profileId);
  const receiptPath = `${artifactPath}.receipt.json`;
  const expectedReceipt = createExpectedArtifactReceipt(options.profile);
  const transport = options.transport ?? createRecallHttpsModelArtifactTransport();

  async function verifyArtifact(): Promise<RecallModelArtifactStatus> {
    const partialPaths = await listPartialArtifactPaths(
      artifactDirectory,
      options.profile.source.artifact,
    );
    if (!(await pathExists(artifactPath))) {
      if (partialPaths.length > 0) {
        return {
          state: 'partial',
          artifactPath,
          partialPaths,
          issue: 'A model download did not complete and no artifact was activated.',
          repair: 'Run model repair and explicitly approve resuming with a fresh pinned download.',
        };
      }
      return {
        state: 'missing',
        artifactPath,
        partialPaths,
        repair: 'Run model repair and explicitly approve the pinned model download.',
      };
    }

    try {
      await validateRecallGgufModelArtifact(artifactPath, options.profile.source);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        state: 'corrupt',
        artifactPath,
        partialPaths,
        issue: message,
        repair: 'Run model repair with explicit download approval to replace the corrupt artifact.',
      };
    }

    try {
      const receipt = Value.Parse(
        recallModelArtifactReceiptSchema,
        JSON.parse(await readFile(receiptPath, 'utf8')),
      );
      if (!isExpectedArtifactReceipt(receipt, expectedReceipt)) {
        throw new Error('the activation receipt belongs to another model profile');
      }
    } catch (error) {
      const message =
        readNodeErrorCode(error) === 'ENOENT'
          ? 'the verified artifact has no activation receipt'
          : error instanceof Error
            ? error.message
            : String(error);
      return {
        state: 'incompatible',
        artifactPath,
        partialPaths,
        issue: `Recall model artifact incompatible: ${message}`,
        repair: 'Run model repair with explicit approval to activate this pinned profile safely.',
      };
    }

    return {
      state: 'valid',
      artifactPath,
      partialPaths,
      repair: 'No repair required.',
    };
  }

  async function removePartialArtifacts(partialPaths: readonly string[]): Promise<void> {
    await Promise.all(partialPaths.map((path) => rm(path, { force: true })));
  }

  return {
    async inspectArtifact() {
      return {
        profile: options.profile,
        status: await verifyArtifact(),
      };
    },
    verifyArtifact,
    async diagnoseArtifact() {
      const status = await verifyArtifact();
      return {
        healthy: status.state === 'valid',
        status,
        action: status.repair,
      };
    },
    async downloadArtifact(approval) {
      if (!approval.approved) {
        throw new Error('Recall model download approval required: rerun with explicit approval');
      }
      const existingStatus = await verifyArtifact();
      if (existingStatus.state === 'valid') {
        return existingStatus;
      }

      await mkdir(artifactDirectory, { recursive: true, mode: 0o700 });
      const partialPath = `${artifactPath}.partial-${randomUUID()}`;
      await transport.downloadArtifact(
        options.profile.source.downloadUrl,
        partialPath,
        approval.signal,
      );
      try {
        await validateRecallGgufModelArtifact(partialPath, options.profile.source);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall downloaded model artifact rejected: ${message}`, { cause: error });
      }
      await rename(partialPath, artifactPath);
      await writeArtifactReceiptAtomically(receiptPath, expectedReceipt);
      await removePartialArtifacts(existingStatus.partialPaths);
      return verifyArtifact();
    },
    async repairArtifact(approval) {
      if (!approval.approved) {
        throw new Error('Recall model repair approval required: rerun with explicit approval');
      }
      const status = await verifyArtifact();
      if (status.state === 'valid') {
        return status;
      }
      if (status.state === 'incompatible') {
        await writeArtifactReceiptAtomically(receiptPath, expectedReceipt);
        await removePartialArtifacts(status.partialPaths);
        return verifyArtifact();
      }
      return this.downloadArtifact(approval);
    },
    async removeArtifact(approval) {
      if (!approval.approved) {
        throw new Error('Recall model removal approval required: rerun with explicit approval');
      }
      await rm(profileDirectory, { recursive: true, force: true });
      return verifyArtifact();
    },
  };
}
