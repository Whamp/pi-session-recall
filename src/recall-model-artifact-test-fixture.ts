import { createHash } from 'node:crypto';

import {
  createRecommendedEmbeddingGemmaModelProfile,
  type RecommendedEmbeddingGemmaModelProfile,
} from './recall-model-profiles.js';

function encodeFixtureGgufString(value: string): Buffer {
  const text = Buffer.from(value, 'utf8');
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(text.length));
  return Buffer.concat([length, text]);
}

/** Creates a tiny structurally valid GGUF with one metadata value and one tensor. */
export function createRecallModelArtifactFixtureGguf(): Buffer {
  const header = Buffer.alloc(24);
  header.write('GGUF', 0, 'ascii');
  header.writeUInt32LE(3, 4);
  header.writeBigUInt64LE(1n, 8);
  header.writeBigUInt64LE(1n, 16);
  const stringType = Buffer.alloc(4);
  stringType.writeUInt32LE(8);
  const dimensions = Buffer.alloc(4);
  dimensions.writeUInt32LE(1);
  const dimensionSize = Buffer.alloc(8);
  dimensionSize.writeBigUInt64LE(1n);
  const tensorTypeAndOffset = Buffer.alloc(12);
  tensorTypeAndOffset.writeUInt32LE(0, 0);
  tensorTypeAndOffset.writeBigUInt64LE(0n, 4);
  const metadataAndTensorDirectory = Buffer.concat([
    header,
    encodeFixtureGgufString('general.architecture'),
    stringType,
    encodeFixtureGgufString('embeddinggemma'),
    encodeFixtureGgufString('token_embd.weight'),
    dimensions,
    dimensionSize,
    tensorTypeAndOffset,
  ]);
  const paddingLength = (32 - (metadataAndTensorDirectory.length % 32)) % 32;
  return Buffer.concat([metadataAndTensorDirectory, Buffer.alloc(paddingLength), Buffer.alloc(4)]);
}

/** Creates a recommended profile whose immutable artifact identity matches a local GGUF fixture. */
export function createRecallModelArtifactFixtureProfile(
  artifact: Buffer,
): RecommendedEmbeddingGemmaModelProfile {
  const recommended = createRecommendedEmbeddingGemmaModelProfile();
  return {
    ...recommended,
    source: {
      ...recommended.source,
      byteSize: artifact.length,
      sha256: createHash('sha256').update(artifact).digest('hex'),
      downloadUrl: 'https://models.invalid/pinned/fixture.gguf',
    },
  };
}
