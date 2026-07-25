import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  OCTEN_TOKENIZER_CONFIG_SHA256,
  OCTEN_TOKENIZER_ENCODE_OPTIONS,
  OCTEN_TOKENIZER_JSON_SHA256,
  OCTEN_TOKENIZER_LIBRARY_VERSION,
  OCTEN_TOKENIZER_MODEL,
  OCTEN_TOKENIZER_REVISION,
  loadChecksummedConversationTokenizer,
  type ConversationTokenizerAssetIdentity,
} from './octen-conversation-tokenizer.js';

function sha256(content: Uint8Array): string {
  return createHash('sha256').update(content).digest('hex');
}

const textEncoder = new TextEncoder();
const tokenizerJson = textEncoder.encode(
  JSON.stringify({
    version: '1.0',
    truncation: null,
    padding: null,
    added_tokens: [],
    normalizer: null,
    pre_tokenizer: { type: 'Whitespace' },
    post_processor: null,
    decoder: { type: 'WordPiece', prefix: '##', cleanup: true },
    model: {
      type: 'WordPiece',
      unk_token: '[UNK]',
      continuing_subword_prefix: '##',
      max_input_chars_per_word: 100,
      vocab: { '[UNK]': 0, hello: 1, world: 2 },
    },
  }),
);
const tokenizerConfigJson = textEncoder.encode('{}');
const fixtureIdentity: ConversationTokenizerAssetIdentity = {
  model: 'fixture/tokenizer',
  revision: 'immutable-fixture-revision',
  library: { name: '@huggingface/tokenizers', version: '0.1.3' },
  encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
  tokenizerJson: {
    fileName: 'tokenizer.json',
    url: 'https://fixtures.test/tokenizer.json',
    sha256: sha256(tokenizerJson),
  },
  tokenizerConfigJson: {
    fileName: 'tokenizer_config.json',
    url: 'https://fixtures.test/tokenizer_config.json',
    sha256: sha256(tokenizerConfigJson),
  },
};

void test('checksummed tokenizer assets load atomically and are verified on every cache load', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-tokenizer-cache-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const assets = new Map([
    [fixtureIdentity.tokenizerJson.url, tokenizerJson],
    [fixtureIdentity.tokenizerConfigJson.url, tokenizerConfigJson],
  ]);
  const fetchedUrls: string[] = [];
  const fetchAsset = async (url: string): Promise<Uint8Array> => {
    fetchedUrls.push(url);
    const content = assets.get(url);
    if (!content) {
      throw new Error(`fixture asset missing: ${url}`);
    }
    return content;
  };

  const first = await loadChecksummedConversationTokenizer({
    cacheDirectory: directory,
    identity: fixtureIdentity,
    fetchAsset,
  });
  assert.deepEqual(first.encodeConversationText('hello world').ids, [1, 2]);
  assert.deepEqual(fetchedUrls, [
    fixtureIdentity.tokenizerJson.url,
    fixtureIdentity.tokenizerConfigJson.url,
  ]);

  const second = await loadChecksummedConversationTokenizer({
    cacheDirectory: directory,
    identity: fixtureIdentity,
    async fetchAsset() {
      throw new Error('offline fetch must not run for a complete cache');
    },
  });
  assert.deepEqual(second.encodeConversationText('world').ids, [2]);
  assert.deepEqual(await readdir(join(directory, fixtureIdentity.revision)), [
    'tokenizer.json',
    'tokenizer_config.json',
  ]);

  await writeFile(
    join(directory, fixtureIdentity.revision, fixtureIdentity.tokenizerJson.fileName),
    'corrupted',
  );
  const corruptCacheLoad = () =>
    loadChecksummedConversationTokenizer({
      cacheDirectory: directory,
      identity: fixtureIdentity,
      async fetchAsset() {
        throw new Error('corrupt cache must not fall back to a download');
      },
    });
  await assert.rejects(
    corruptCacheLoad,
    /Recall tokenizer cache corrupt.*tokenizer\.json.*expected.*actual.*remove the corrupt asset/,
  );

  await writeFile(
    join(directory, fixtureIdentity.revision, fixtureIdentity.tokenizerJson.fileName),
    tokenizerJson,
  );
  await writeFile(
    join(directory, fixtureIdentity.revision, fixtureIdentity.tokenizerConfigJson.fileName),
    'corrupted',
  );
  await assert.rejects(
    corruptCacheLoad,
    /Recall tokenizer cache corrupt.*tokenizer_config\.json.*expected.*actual.*remove the corrupt asset/,
  );
});

void test('checksummed tokenizer reports an actionable offline cache miss', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-tokenizer-offline-'));
  t.after(() => rm(directory, { recursive: true, force: true }));

  await assert.rejects(
    () =>
      loadChecksummedConversationTokenizer({
        cacheDirectory: directory,
        identity: fixtureIdentity,
        async fetchAsset() {
          throw new Error('network offline');
        },
      }),
    /Recall tokenizer cache miss.*network offline.*pre-populate the cache or reconnect/,
  );
});

void test('Octen tokenizer identity pins the immutable revision, exact checksums, library, and options', () => {
  assert.equal(OCTEN_TOKENIZER_MODEL, 'Octen/Octen-Embedding-4B');
  assert.equal(OCTEN_TOKENIZER_REVISION, '6e188e3b072c3e3678b235ad84e6e97bcbb71e8f');
  assert.equal(
    OCTEN_TOKENIZER_JSON_SHA256,
    '83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d',
  );
  assert.equal(
    OCTEN_TOKENIZER_CONFIG_SHA256,
    '0a04a9d7d4a62b28482bdfe726c122756de85714fb64166ace92ae75b8f57614',
  );
  assert.equal(OCTEN_TOKENIZER_LIBRARY_VERSION, '0.1.3');
  assert.deepEqual(OCTEN_TOKENIZER_ENCODE_OPTIONS, {
    addSpecialTokens: false,
    returnTokenTypeIds: false,
  });
});
