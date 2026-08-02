import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';

import { Tokenizer } from '@huggingface/tokenizers';

import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

/** Hugging Face model repository whose tokenizer defines Octen embedding token limits. */
export const OCTEN_TOKENIZER_MODEL = 'Octen/Octen-Embedding-4B';

/** Immutable Octen model revision validated against the local llama.cpp tokenizer endpoint. */
export const OCTEN_TOKENIZER_REVISION = '6e188e3b072c3e3678b235ad84e6e97bcbb71e8f';

/** SHA-256 of the pinned 11 MB Octen tokenizer definition; the file is cached, not committed. */
export const OCTEN_TOKENIZER_JSON_SHA256 =
  '83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d';

/** SHA-256 of the pinned Octen tokenizer configuration. */
export const OCTEN_TOKENIZER_CONFIG_SHA256 =
  '0a04a9d7d4a62b28482bdfe726c122756de85714fb64166ace92ae75b8f57614';

/** Exact local tokenizer library version used to produce chunk geometry. */
export const OCTEN_TOKENIZER_LIBRARY_VERSION = '0.1.3';

/** Encode options that match raw llama.cpp tokenization without injected special tokens. */
export const OCTEN_TOKENIZER_ENCODE_OPTIONS = {
  addSpecialTokens: false,
  returnTokenTypeIds: false,
};

/** One immutable tokenizer asset downloaded into the recall data cache. */
export interface ConversationTokenizerAssetFileIdentity {
  fileName: string;
  url: string;
  sha256: string;
}

/** Full tokenizer identity needed to reproduce exact token-aware chunk geometry. */
export interface ConversationTokenizerAssetIdentity {
  model: string;
  revision: string;
  library: { name: string; version: string };
  encodeOptions: { addSpecialTokens: boolean; returnTokenTypeIds: boolean };
  tokenizerJson: ConversationTokenizerAssetFileIdentity;
  tokenizerConfigJson: ConversationTokenizerAssetFileIdentity;
}

/** Dependencies and cache path for a checksum-verified tokenizer load. */
export interface ChecksummedConversationTokenizerLoadOptions {
  cacheDirectory: string;
  identity: ConversationTokenizerAssetIdentity;
  fetchAsset: (url: string) => Promise<Uint8Array>;
}

/** Cache path and optional transport override for loading the pinned Octen tokenizer. */
export interface OctenConversationTokenizerLoadOptions {
  cacheDirectory: string;
  fetchAsset?: (url: string) => Promise<Uint8Array>;
}

/** Production identity for the exact Octen tokenizer and immutable Hugging Face assets. */
export const OCTEN_TOKENIZER_IDENTITY: ConversationTokenizerAssetIdentity = {
  model: OCTEN_TOKENIZER_MODEL,
  revision: OCTEN_TOKENIZER_REVISION,
  library: { name: '@huggingface/tokenizers', version: OCTEN_TOKENIZER_LIBRARY_VERSION },
  encodeOptions: OCTEN_TOKENIZER_ENCODE_OPTIONS,
  tokenizerJson: {
    fileName: 'tokenizer.json',
    url: `https://huggingface.co/${OCTEN_TOKENIZER_MODEL}/resolve/${OCTEN_TOKENIZER_REVISION}/tokenizer.json`,
    sha256: OCTEN_TOKENIZER_JSON_SHA256,
  },
  tokenizerConfigJson: {
    fileName: 'tokenizer_config.json',
    url: `https://huggingface.co/${OCTEN_TOKENIZER_MODEL}/resolve/${OCTEN_TOKENIZER_REVISION}/tokenizer_config.json`,
    sha256: OCTEN_TOKENIZER_CONFIG_SHA256,
  },
};

function hashTokenizerAsset(content: Uint8Array): string {
  return createHash('sha256').update(content).digest('hex');
}

function assertSafeTokenizerAssetIdentity(identity: ConversationTokenizerAssetIdentity): void {
  if (
    identity.library.name !== '@huggingface/tokenizers' ||
    identity.library.version !== OCTEN_TOKENIZER_LIBRARY_VERSION
  ) {
    throw new Error(
      `Recall tokenizer library incompatible: expected @huggingface/tokenizers@${OCTEN_TOKENIZER_LIBRARY_VERSION}, received ${identity.library.name}@${identity.library.version}`,
    );
  }
  for (const asset of [identity.tokenizerJson, identity.tokenizerConfigJson]) {
    if (basename(asset.fileName) !== asset.fileName || !/^[a-f0-9]{64}$/u.test(asset.sha256)) {
      throw new Error(`Recall tokenizer asset identity invalid: ${asset.fileName}`);
    }
  }
}

async function readVerifiedCachedTokenizerAsset(
  path: string,
  asset: ConversationTokenizerAssetFileIdentity,
): Promise<Uint8Array | null> {
  let content: Uint8Array;
  try {
    content = await readFile(path);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
  const actualChecksum = hashTokenizerAsset(content);
  if (actualChecksum !== asset.sha256) {
    throw new Error(
      `Recall tokenizer cache corrupt at ${path}: expected SHA-256 ${asset.sha256}, actual ${actualChecksum}; remove the corrupt asset and rerun psr index`,
    );
  }
  return content;
}

async function cacheVerifiedTokenizerAsset(
  revisionDirectory: string,
  asset: ConversationTokenizerAssetFileIdentity,
  fetchAsset: (url: string) => Promise<Uint8Array>,
): Promise<Uint8Array> {
  const cachePath = join(revisionDirectory, asset.fileName);
  const cached = await readVerifiedCachedTokenizerAsset(cachePath, asset);
  if (cached) {
    return cached;
  }

  let downloaded: Uint8Array;
  try {
    downloaded = await fetchAsset(asset.url);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall tokenizer cache miss at ${cachePath}; download ${asset.url} failed: ${message}; pre-populate the cache or reconnect and rerun psr index`,
      { cause: error },
    );
  }

  const temporaryPath = `${cachePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, downloaded, { flag: 'wx' });
    const verified = await readVerifiedCachedTokenizerAsset(temporaryPath, asset);
    if (!verified) {
      throw new Error(`Recall tokenizer temporary asset disappeared: ${temporaryPath}`);
    }
    await rename(temporaryPath, cachePath);
    return verified;
  } catch (error) {
    await rm(temporaryPath, { force: true });
    if (error instanceof Error && error.message.startsWith('Recall tokenizer cache corrupt')) {
      throw new Error(
        `Recall tokenizer download checksum mismatch for ${asset.url}: ${error.message}; no cache file was installed`,
        { cause: error },
      );
    }
    throw error;
  }
}

function parseTokenizerJson(content: Uint8Array, path: string): Record<string, unknown> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(content));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall tokenizer JSON invalid at ${path}: ${message}`, { cause: error });
  }
  if (!isUnknownRecord(parsed)) {
    throw new Error(`Recall tokenizer JSON invalid at ${path}: expected an object`);
  }
  return parsed;
}

/** Loads any immutable tokenizer asset pair after verifying cache or download checksums. */
export async function loadChecksummedConversationTokenizer(
  options: ChecksummedConversationTokenizerLoadOptions,
): Promise<ConversationTextTokenizer> {
  assertSafeTokenizerAssetIdentity(options.identity);
  const revisionDirectory = join(options.cacheDirectory, options.identity.revision);
  await mkdir(revisionDirectory, { recursive: true });
  const tokenizerContent = await cacheVerifiedTokenizerAsset(
    revisionDirectory,
    options.identity.tokenizerJson,
    options.fetchAsset,
  );
  const configContent = await cacheVerifiedTokenizerAsset(
    revisionDirectory,
    options.identity.tokenizerConfigJson,
    options.fetchAsset,
  );
  const tokenizerPath = join(revisionDirectory, options.identity.tokenizerJson.fileName);
  const configPath = join(revisionDirectory, options.identity.tokenizerConfigJson.fileName);
  const tokenizer = new Tokenizer(
    parseTokenizerJson(tokenizerContent, tokenizerPath),
    parseTokenizerJson(configContent, configPath),
  );

  return {
    encodeConversationText(text) {
      const encoding = tokenizer.encode(text, {
        'add_special_tokens': options.identity.encodeOptions.addSpecialTokens,
        'return_token_type_ids': options.identity.encodeOptions.returnTokenTypeIds,
      });
      return { ids: [...encoding.ids] };
    },
  };
}

async function fetchOctenTokenizerAsset(url: string): Promise<Uint8Array> {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/** Loads the exact pinned Octen tokenizer from the recall data cache, downloading only on a miss. */
export function loadOctenConversationTokenizer(
  options: OctenConversationTokenizerLoadOptions,
): Promise<ConversationTextTokenizer> {
  return loadChecksummedConversationTokenizer({
    cacheDirectory: options.cacheDirectory,
    identity: OCTEN_TOKENIZER_IDENTITY,
    fetchAsset: options.fetchAsset ?? fetchOctenTokenizerAsset,
  });
}
