import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';

import { LocalOctenModelDownloadProgressKind, LocalOctenModelStatusKind } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { probeLocalOctenEmbeddingRuntime } from './local-octen-embedding-provider.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const LOCAL_OCTEN_RELEASE_TAG = 'model-octen-embedding-0.6b-onnx-int8-v1';
const LOCAL_OCTEN_RELEASE_URL = `https://github.com/Whamp/pi-session-recall/releases/tag/${LOCAL_OCTEN_RELEASE_TAG}`;
const LOCAL_OCTEN_ASSET_BASE_URL = `https://github.com/Whamp/pi-session-recall/releases/download/${LOCAL_OCTEN_RELEASE_TAG}`;
const MODEL_RECEIPT_FILE_NAME = 'model-receipt.json';

/** One immutable file required by the local Octen embedding runtime. */
export interface LocalOctenArtifactFileIdentity {
  fileName: string;
  url: string;
  bytes: number;
  sha256: string;
}

/** Complete downloadable identity for one certified local Octen artifact. */
export interface LocalOctenArtifactIdentity {
  artifactId: string;
  releaseUrl: string;
  nativeDimensions: number;
  files: readonly LocalOctenArtifactFileIdentity[];
}

/** Certified project-controlled Octen 0.6B SmoothQuant ONNX artifact. */
export const LOCAL_OCTEN_ARTIFACT_IDENTITY: LocalOctenArtifactIdentity = Object.freeze({
  artifactId: 'local-octen-embedding-0.6b-onnx-int8-v1',
  releaseUrl: LOCAL_OCTEN_RELEASE_URL,
  nativeDimensions: 1_024,
  files: Object.freeze(
    [
      [
        'artifact-manifest.json',
        1_074,
        '893d00b0bda44277cae5f887fe58b302ae60a3a8db87d32221a6315379e549f0',
      ],
      ['LICENSE.txt', 11_358, 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'],
      [
        'model.int8.onnx',
        5_451_403,
        '48c4eb1401ba5a5d22d7a7e1fb3e94d63e8ed06231e3d124babc00ead78c8771',
      ],
      [
        'model.int8.onnx.data',
        1_062_674_432,
        '1ea5b1a2737474b819a301725cb71381e418d7baa8263769f73486fbe9a74b65',
      ],
      ['NOTICE.txt', 604, '283ce364ec7c107fd25e23092800dd261bdefd99bfefb5bd870ebc9b35c56451'],
      [
        'tokenizer.json',
        11_423_705,
        'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a',
      ],
      [
        'tokenizer_config.json',
        5_404,
        '443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0',
      ],
    ].map(([fileName, bytes, sha256]) => ({
      fileName: String(fileName),
      url: `${LOCAL_OCTEN_ASSET_BASE_URL}/${String(fileName)}`,
      bytes: Number(bytes),
      sha256: String(sha256),
    })),
  ),
});

/** Read-only local model cache state. */
export interface LocalOctenModelStatus {
  kind: LocalOctenModelStatusKind;
  artifactId: string;
  modelDirectory: string;
  totalBytes: number;
  partialDownloads: number;
  detail: string;
}

/** Observable progress for one approved artifact download. */
export interface LocalOctenModelDownloadProgress {
  kind: LocalOctenModelDownloadProgressKind;
  fileName?: string;
  completedBytes?: number;
  totalBytes?: number;
}

/** Explicit controls for a local model download or repair. */
export interface LocalOctenModelDownloadOptions {
  approved: boolean;
  signal?: AbortSignal;
  onProgress?: (event: LocalOctenModelDownloadProgress) => void;
}

/** Result of making the certified local model available. */
export interface LocalOctenModelDownloadResult {
  downloaded: boolean;
  modelDirectory: string;
}

/** Runtime result included only after full artifact verification. */
export interface LocalOctenRuntimeProbeResult {
  dimensions: number;
  norm: number;
}

/** Read-only artifact and runtime diagnosis. */
export interface LocalOctenModelDoctorResult {
  healthy: boolean;
  status: LocalOctenModelStatus;
  detail: string;
  runtime?: LocalOctenRuntimeProbeResult;
}

/** Replaceable boundaries used to verify model-manager behavior without network access. */
export interface LocalOctenModelManagerOptions {
  modelRootDirectory: string;
  artifact?: LocalOctenArtifactIdentity;
  downloadSource?: (url: string, signal?: AbortSignal) => AsyncIterable<Uint8Array>;
  probeRuntime?: (modelDirectory: string) => Promise<LocalOctenRuntimeProbeResult>;
}

/** Safe local artifact operations used by `psr model` and setup. */
export interface LocalOctenModelManager {
  status(): Promise<LocalOctenModelStatus>;
  download(options: LocalOctenModelDownloadOptions): Promise<LocalOctenModelDownloadResult>;
  doctor(): Promise<LocalOctenModelDoctorResult>;
}

interface LocalOctenModelReceipt {
  schemaVersion: 1;
  artifactId: string;
  releaseUrl: string;
  files: Array<{ fileName: string; bytes: number; sha256: string }>;
}

function createModelReceipt(artifact: LocalOctenArtifactIdentity): LocalOctenModelReceipt {
  return {
    schemaVersion: 1,
    artifactId: artifact.artifactId,
    releaseUrl: artifact.releaseUrl,
    files: artifact.files.map(({ fileName, bytes, sha256 }) => ({ fileName, bytes, sha256 })),
  };
}

function assertSafeArtifactIdentity(artifact: LocalOctenArtifactIdentity): void {
  if (!artifact.artifactId || basename(artifact.artifactId) !== artifact.artifactId) {
    throw new Error(`Local Octen artifact ID is unsafe: ${artifact.artifactId}`);
  }
  if (!Number.isInteger(artifact.nativeDimensions) || artifact.nativeDimensions < 1) {
    throw new Error('Local Octen artifact native dimensions must be a positive integer');
  }
  if (artifact.files.length === 0) {
    throw new Error('Local Octen artifact must contain at least one file');
  }
  const fileNames = new Set<string>();
  for (const file of artifact.files) {
    if (
      !file.fileName ||
      basename(file.fileName) !== file.fileName ||
      fileNames.has(file.fileName) ||
      !Number.isInteger(file.bytes) ||
      file.bytes < 0 ||
      !/^[a-f0-9]{64}$/u.test(file.sha256)
    ) {
      throw new Error(`Local Octen artifact file identity is invalid: ${file.fileName}`);
    }
    fileNames.add(file.fileName);
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function listPartialDirectories(root: string, artifactId: string): Promise<string[]> {
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
  const prefix = `.${artifactId}.partial-`;
  return entries
    .filter((entry) => entry.isDirectory() && entry.name.startsWith(prefix))
    .map((entry) => join(root, entry.name));
}

function parseModelReceipt(content: string): LocalOctenModelReceipt | null {
  try {
    const parsed: unknown = JSON.parse(content);
    if (
      !isUnknownRecord(parsed) ||
      parsed.schemaVersion !== 1 ||
      typeof parsed.artifactId !== 'string' ||
      typeof parsed.releaseUrl !== 'string' ||
      !Array.isArray(parsed.files)
    ) {
      return null;
    }
    const files: LocalOctenModelReceipt['files'] = [];
    for (const value of parsed.files) {
      if (
        !isUnknownRecord(value) ||
        typeof value.fileName !== 'string' ||
        typeof value.bytes !== 'number' ||
        typeof value.sha256 !== 'string'
      ) {
        return null;
      }
      files.push({ fileName: value.fileName, bytes: value.bytes, sha256: value.sha256 });
    }
    return {
      schemaVersion: 1,
      artifactId: parsed.artifactId,
      releaseUrl: parsed.releaseUrl,
      files,
    };
  } catch {
    return null;
  }
}

function receiptsMatch(
  actual: LocalOctenModelReceipt | null,
  expected: LocalOctenModelReceipt,
): boolean {
  return actual !== null && JSON.stringify(actual) === JSON.stringify(expected);
}

async function inspectInstalledArtifact(
  modelDirectory: string,
  artifact: LocalOctenArtifactIdentity,
  partialDownloads: number,
): Promise<LocalOctenModelStatus> {
  const totalBytes = artifact.files.reduce((sum, file) => sum + file.bytes, 0);
  if (!(await pathExists(modelDirectory))) {
    return {
      kind:
        partialDownloads > 0
          ? LocalOctenModelStatusKind.PARTIAL
          : LocalOctenModelStatusKind.MISSING,
      artifactId: artifact.artifactId,
      modelDirectory,
      totalBytes,
      partialDownloads,
      detail:
        partialDownloads > 0
          ? 'Local Octen model download is incomplete'
          : 'Local Octen model is not downloaded',
    };
  }

  let receipt: LocalOctenModelReceipt | null = null;
  try {
    receipt = parseModelReceipt(
      await readFile(join(modelDirectory, MODEL_RECEIPT_FILE_NAME), 'utf8'),
    );
  } catch (error) {
    if (readNodeErrorCode(error) !== 'ENOENT') {
      throw error;
    }
  }
  const expectedReceipt = createModelReceipt(artifact);
  if (!receiptsMatch(receipt, expectedReceipt)) {
    return {
      kind: LocalOctenModelStatusKind.CORRUPT,
      artifactId: artifact.artifactId,
      modelDirectory,
      totalBytes,
      partialDownloads,
      detail: 'Local Octen model receipt is missing or incompatible',
    };
  }

  for (const file of artifact.files) {
    try {
      const fileStatus = await stat(join(modelDirectory, file.fileName));
      if (!fileStatus.isFile() || fileStatus.size !== file.bytes) {
        throw new Error('size mismatch');
      }
    } catch (error) {
      const detail = readNodeErrorCode(error) === 'ENOENT' ? 'is missing' : 'has the wrong size';
      return {
        kind: LocalOctenModelStatusKind.CORRUPT,
        artifactId: artifact.artifactId,
        modelDirectory,
        totalBytes,
        partialDownloads,
        detail: `Local Octen model file ${file.fileName} ${detail}`,
      };
    }
  }

  return {
    kind: LocalOctenModelStatusKind.READY,
    artifactId: artifact.artifactId,
    modelDirectory,
    totalBytes,
    partialDownloads,
    detail: 'Local Octen model is ready',
  };
}

async function hashFile(path: string): Promise<string> {
  const hash = createHash('sha256');
  const handle = await open(path, 'r');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  try {
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.byteLength, null);
      if (bytesRead === 0) {
        break;
      }
      hash.update(buffer.subarray(0, bytesRead));
    }
  } finally {
    await handle.close();
  }
  return hash.digest('hex');
}

async function verifyInstalledArtifact(
  modelDirectory: string,
  artifact: LocalOctenArtifactIdentity,
): Promise<string | null> {
  for (const file of artifact.files) {
    const path = join(modelDirectory, file.fileName);
    let fileStatus;
    try {
      fileStatus = await stat(path);
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        return `${file.fileName} is missing`;
      }
      throw error;
    }
    if (!fileStatus.isFile() || fileStatus.size !== file.bytes) {
      return `${file.fileName} size mismatch: expected ${file.bytes}, received ${fileStatus.size}`;
    }
    const actualSha256 = await hashFile(path);
    if (actualSha256 !== file.sha256) {
      return `${file.fileName} checksum mismatch: expected ${file.sha256}, received ${actualSha256}`;
    }
  }
  return null;
}

async function* fetchDownloadSource(url: string, signal?: AbortSignal): AsyncIterable<Uint8Array> {
  const response = await fetch(url, {
    redirect: 'follow',
    ...(signal ? { signal } : {}),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Local Octen model download failed for ${url}: HTTP ${response.status}`);
  }
  const reader = response.body.getReader();
  try {
    while (true) {
      signal?.throwIfAborted();
      const result = await reader.read();
      if (result.done) {
        return;
      }
      yield result.value;
    }
  } finally {
    reader.releaseLock();
  }
}

async function downloadVerifiedFile(
  path: string,
  file: LocalOctenArtifactFileIdentity,
  downloadSource: (url: string, signal?: AbortSignal) => AsyncIterable<Uint8Array>,
  signal?: AbortSignal,
): Promise<void> {
  const hash = createHash('sha256');
  let receivedBytes = 0;
  const handle = await open(path, 'wx', 0o600);
  try {
    for await (const chunk of downloadSource(file.url, signal)) {
      signal?.throwIfAborted();
      let offset = 0;
      while (offset < chunk.byteLength) {
        const writeResult = await handle.write(chunk, offset, chunk.byteLength - offset, null);
        offset += writeResult.bytesWritten;
      }
      hash.update(chunk);
      receivedBytes += chunk.byteLength;
      if (receivedBytes > file.bytes) {
        throw new Error(
          `${file.fileName} size mismatch: expected ${file.bytes}, received more than expected`,
        );
      }
    }
    await handle.sync();
  } finally {
    await handle.close();
  }
  if (receivedBytes !== file.bytes) {
    throw new Error(
      `${file.fileName} size mismatch: expected ${file.bytes}, received ${receivedBytes}`,
    );
  }
  const actualSha256 = hash.digest('hex');
  if (actualSha256 !== file.sha256) {
    throw new Error(
      `${file.fileName} checksum mismatch: expected ${file.sha256}, received ${actualSha256}`,
    );
  }
}

/** Resolves the immutable installed directory for one local Octen artifact. */
export function resolveLocalOctenModelDirectory(
  modelRootDirectory: string,
  artifact: LocalOctenArtifactIdentity = LOCAL_OCTEN_ARTIFACT_IDENTITY,
): string {
  assertSafeArtifactIdentity(artifact);
  return join(modelRootDirectory, artifact.artifactId);
}

/** Creates one deep artifact manager that never exposes a partial model directory. */
export function createLocalOctenModelManager(
  options: LocalOctenModelManagerOptions,
): LocalOctenModelManager {
  const artifact = options.artifact ?? LOCAL_OCTEN_ARTIFACT_IDENTITY;
  assertSafeArtifactIdentity(artifact);
  const modelDirectory = resolveLocalOctenModelDirectory(options.modelRootDirectory, artifact);
  const totalBytes = artifact.files.reduce((sum, file) => sum + file.bytes, 0);
  const downloadSource = options.downloadSource ?? fetchDownloadSource;
  const probeRuntime =
    options.probeRuntime ??
    ((directory: string) => probeLocalOctenEmbeddingRuntime(directory, artifact.nativeDimensions));

  async function status(): Promise<LocalOctenModelStatus> {
    const partialDirectories = await listPartialDirectories(
      options.modelRootDirectory,
      artifact.artifactId,
    );
    return inspectInstalledArtifact(modelDirectory, artifact, partialDirectories.length);
  }

  return {
    status,
    async download(downloadOptions) {
      if (!downloadOptions.approved) {
        throw new Error(
          `Local Octen model download requires explicit approval for ${totalBytes} bytes`,
        );
      }
      downloadOptions.signal?.throwIfAborted();
      const currentStatus = await status();
      if (
        currentStatus.kind === LocalOctenModelStatusKind.READY &&
        (await verifyInstalledArtifact(modelDirectory, artifact)) === null
      ) {
        return { downloaded: false, modelDirectory };
      }

      await mkdir(options.modelRootDirectory, { recursive: true });
      for (const partialDirectory of await listPartialDirectories(
        options.modelRootDirectory,
        artifact.artifactId,
      )) {
        await rm(partialDirectory, { recursive: true, force: true });
      }
      downloadOptions.onProgress?.({
        kind: LocalOctenModelDownloadProgressKind.PREPARING,
        totalBytes,
      });
      const uniqueId = randomUUID();
      const partialDirectory = join(
        options.modelRootDirectory,
        `.${artifact.artifactId}.partial-${uniqueId}`,
      );
      const replacedDirectory = join(
        options.modelRootDirectory,
        `.${artifact.artifactId}.replaced-${uniqueId}`,
      );
      await mkdir(partialDirectory);
      let movedExistingModel = false;
      try {
        let completedBytes = 0;
        for (const file of artifact.files) {
          downloadOptions.onProgress?.({
            kind: LocalOctenModelDownloadProgressKind.DOWNLOADING_FILE,
            fileName: file.fileName,
            completedBytes,
            totalBytes,
          });
          await downloadVerifiedFile(
            join(partialDirectory, file.fileName),
            file,
            downloadSource,
            downloadOptions.signal,
          );
          completedBytes += file.bytes;
          downloadOptions.onProgress?.({
            kind: LocalOctenModelDownloadProgressKind.FILE_VERIFIED,
            fileName: file.fileName,
            completedBytes,
            totalBytes,
          });
        }
        await writeFile(
          join(partialDirectory, MODEL_RECEIPT_FILE_NAME),
          `${JSON.stringify(createModelReceipt(artifact))}\n`,
          { encoding: 'utf8', mode: 0o600, flag: 'wx' },
        );
        if (await pathExists(modelDirectory)) {
          await rename(modelDirectory, replacedDirectory);
          movedExistingModel = true;
        }
        await rename(partialDirectory, modelDirectory);
        if (movedExistingModel) {
          await rm(replacedDirectory, { recursive: true, force: true });
        }
        downloadOptions.onProgress?.({
          kind: LocalOctenModelDownloadProgressKind.ACTIVATED,
          completedBytes: totalBytes,
          totalBytes,
        });
        return { downloaded: true, modelDirectory };
      } catch (error) {
        await rm(partialDirectory, { recursive: true, force: true });
        if (movedExistingModel && !(await pathExists(modelDirectory))) {
          await rename(replacedDirectory, modelDirectory);
        }
        throw error;
      }
    },
    async doctor() {
      const currentStatus = await status();
      if (currentStatus.kind !== LocalOctenModelStatusKind.READY) {
        return { healthy: false, status: currentStatus, detail: currentStatus.detail };
      }
      const integrityFailure = await verifyInstalledArtifact(modelDirectory, artifact);
      if (integrityFailure) {
        return { healthy: false, status: currentStatus, detail: integrityFailure };
      }
      try {
        const runtime = await probeRuntime(modelDirectory);
        if (
          runtime.dimensions !== artifact.nativeDimensions ||
          !Number.isFinite(runtime.norm) ||
          Math.abs(runtime.norm - 1) > 0.001
        ) {
          return {
            healthy: false,
            status: currentStatus,
            detail: `Local Octen runtime probe invalid: expected ${artifact.nativeDimensions} normalized dimensions, received ${runtime.dimensions} dimensions with norm ${runtime.norm}`,
            runtime,
          };
        }
        return {
          healthy: true,
          status: currentStatus,
          detail: 'Local Octen model files and runtime probe are healthy',
          runtime,
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return {
          healthy: false,
          status: currentStatus,
          detail: `Local Octen runtime probe failed: ${message}`,
        };
      }
    },
  };
}
