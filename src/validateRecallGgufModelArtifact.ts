import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { open, stat } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';

const GGUF_FIXED_VALUE_BYTES = new Map<number, number>([
  [0, 1],
  [1, 1],
  [2, 2],
  [3, 2],
  [4, 4],
  [5, 4],
  [6, 4],
  [7, 1],
  [10, 8],
  [11, 8],
  [12, 8],
]);
const GGUF_STRING_VALUE_TYPE = 8;
const GGUF_ARRAY_VALUE_TYPE = 9;
const MAX_GGUF_DIRECTORY_ENTRIES = 10_000_000;
const MAX_GGUF_STRING_BYTES = 64 * 1024 * 1024;

/** Expected immutable size and SHA-256 for one GGUF model artifact. */
export interface RecallGgufArtifactExpectation {
  byteSize: number;
  sha256: string;
}

class GgufDirectoryReader {
  private offset = 0;

  constructor(
    private readonly file: Awaited<ReturnType<typeof open>>,
    private readonly fileSize: number,
  ) {}

  async readBytes(length: number): Promise<Buffer> {
    if (!Number.isSafeInteger(length) || length < 0 || this.offset + length > this.fileSize) {
      throw new Error(`GGUF directory truncated at byte ${this.offset}`);
    }
    const buffer = Buffer.alloc(length);
    const { bytesRead } = await this.file.read(buffer, 0, length, this.offset);
    if (bytesRead !== length) {
      throw new Error(`GGUF directory truncated at byte ${this.offset}`);
    }
    this.offset += length;
    return buffer;
  }

  async readCount(label: string): Promise<number> {
    const value = (await this.readBytes(8)).readBigUInt64LE();
    if (value > BigInt(MAX_GGUF_DIRECTORY_ENTRIES)) {
      throw new Error(`GGUF ${label} exceeds ${MAX_GGUF_DIRECTORY_ENTRIES}`);
    }
    return Number(value);
  }

  async skipString(): Promise<void> {
    const length = (await this.readBytes(8)).readBigUInt64LE();
    if (length > BigInt(MAX_GGUF_STRING_BYTES)) {
      throw new Error(`GGUF string exceeds ${MAX_GGUF_STRING_BYTES} bytes`);
    }
    await this.readBytes(Number(length));
  }

  async skipValue(valueType: number): Promise<void> {
    const fixedBytes = GGUF_FIXED_VALUE_BYTES.get(valueType);
    if (fixedBytes !== undefined) {
      await this.readBytes(fixedBytes);
      return;
    }
    if (valueType === GGUF_STRING_VALUE_TYPE) {
      await this.skipString();
      return;
    }
    if (valueType !== GGUF_ARRAY_VALUE_TYPE) {
      throw new Error(`GGUF metadata value type unsupported: ${valueType}`);
    }
    const elementType = (await this.readBytes(4)).readUInt32LE();
    if (elementType === GGUF_ARRAY_VALUE_TYPE) {
      throw new Error('GGUF nested metadata arrays are invalid');
    }
    const elementCount = await this.readCount('metadata array length');
    const elementBytes = GGUF_FIXED_VALUE_BYTES.get(elementType);
    if (elementBytes !== undefined) {
      await this.readBytes(elementCount * elementBytes);
      return;
    }
    if (elementType !== GGUF_STRING_VALUE_TYPE) {
      throw new Error(`GGUF metadata array type unsupported: ${elementType}`);
    }
    for (let index = 0; index < elementCount; index += 1) {
      await this.skipString();
    }
  }
}

async function assertGgufStructure(path: string, fileSize: number): Promise<void> {
  const file = await open(path, 'r');
  try {
    const reader = new GgufDirectoryReader(file, fileSize);
    const magic = (await reader.readBytes(4)).toString('ascii');
    if (magic !== 'GGUF') {
      throw new Error(`GGUF magic invalid: expected GGUF, received ${JSON.stringify(magic)}`);
    }
    const version = (await reader.readBytes(4)).readUInt32LE();
    if (version !== 2 && version !== 3) {
      throw new Error(`GGUF version unsupported: ${version}`);
    }
    const tensorCount = await reader.readCount('tensor count');
    const metadataCount = await reader.readCount('metadata count');
    if (tensorCount === 0 || metadataCount === 0) {
      throw new Error('GGUF directory invalid: expected metadata and at least one tensor');
    }
    for (let index = 0; index < metadataCount; index += 1) {
      await reader.skipString();
      await reader.skipValue((await reader.readBytes(4)).readUInt32LE());
    }
    for (let index = 0; index < tensorCount; index += 1) {
      await reader.skipString();
      const dimensions = (await reader.readBytes(4)).readUInt32LE();
      if (dimensions < 1 || dimensions > 4) {
        throw new Error(`GGUF tensor dimensions invalid at index ${index}: ${dimensions}`);
      }
      await reader.readBytes(dimensions * 8);
      await reader.readBytes(4);
      const tensorOffset = (await reader.readBytes(8)).readBigUInt64LE();
      if (tensorOffset >= BigInt(fileSize)) {
        throw new Error(`GGUF tensor offset outside artifact at index ${index}`);
      }
    }
  } finally {
    await file.close();
  }
}

async function hashFileSha256(path: string): Promise<string> {
  const hash = createHash('sha256');
  await pipeline(createReadStream(path), hash);
  return hash.digest('hex');
}

/** Fails unless a model file has the pinned size, SHA-256, and a bounded valid GGUF directory. */
export async function validateRecallGgufModelArtifact(
  path: string,
  expectation: Readonly<RecallGgufArtifactExpectation>,
): Promise<void> {
  const file = await stat(path);
  if (!file.isFile()) {
    throw new Error(`GGUF artifact invalid: expected a regular file at ${path}`);
  }
  if (file.size !== expectation.byteSize) {
    throw new Error(
      `GGUF artifact size mismatch: expected ${expectation.byteSize} bytes, received ${file.size}`,
    );
  }
  const sha256 = await hashFileSha256(path);
  if (sha256 !== expectation.sha256) {
    throw new Error(
      `GGUF artifact SHA-256 mismatch: expected ${expectation.sha256}, received ${sha256}`,
    );
  }
  await assertGgufStructure(path, file.size);
}
