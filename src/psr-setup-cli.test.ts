import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { LocalOctenModelStatusKind, RecallEmbeddingProfile } from './enums.js';
import type { LocalOctenModelManager } from './local-octen-model-manager.js';
import { runPsrSetupCli } from './psr-setup-cli.js';

function createFixture(options: {
  homeDirectory: string;
  selectedProfile?: RecallEmbeddingProfile;
  confirmations?: boolean[];
  modelStatus?: LocalOctenModelStatusKind;
  promptResponses?: string[];
}): {
  dependencies: Parameters<typeof runPsrSetupCli>[1];
  output: string[];
  progress: string[];
  managerRoots: string[];
  downloadApprovals: boolean[];
} {
  const output: string[] = [];
  const progress: string[] = [];
  const managerRoots: string[] = [];
  const downloadApprovals: boolean[] = [];
  const confirmations = [...(options.confirmations ?? [])];
  const promptResponses = [...(options.promptResponses ?? [])];
  return {
    output,
    progress,
    managerRoots,
    downloadApprovals,
    dependencies: {
      getHomeDirectory() {
        return options.homeDirectory;
      },
      async selectProfile() {
        return options.selectedProfile ?? RecallEmbeddingProfile.LOCAL_OCTEN;
      },
      async confirm() {
        return confirmations.shift() ?? false;
      },
      async promptText(_question, defaultValue) {
        return promptResponses.shift() ?? defaultValue;
      },
      createModelManager(modelRootDirectory) {
        managerRoots.push(modelRootDirectory);
        const status = {
          kind: options.modelStatus ?? LocalOctenModelStatusKind.MISSING,
          artifactId: RecallEmbeddingProfile.LOCAL_OCTEN,
          modelDirectory: join(modelRootDirectory, RecallEmbeddingProfile.LOCAL_OCTEN),
          totalBytes: 1_073_741_824,
          partialDownloads: 0,
          detail: 'fixture model status',
        };
        return {
          async status() {
            return status;
          },
          async download(downloadOptions) {
            downloadApprovals.push(downloadOptions.approved);
            return { downloaded: true, modelDirectory: status.modelDirectory };
          },
          async doctor() {
            throw new Error('setup must not run doctor');
          },
        } satisfies LocalOctenModelManager;
      },
      writeOutput(text) {
        output.push(text);
      },
      writeProgress(text) {
        progress.push(text);
      },
    },
  };
}

void test('psr setup --help prints complete flags without selecting a profile', async () => {
  const fixture = createFixture({ homeDirectory: '/tmp/psr-setup-help' });

  assert.deepEqual(await runPsrSetupCli(['--help'], fixture.dependencies), {
    exitCode: 0,
    runInitialIndex: false,
  });
  assert.match(fixture.output.join(''), /psr setup \[--local\|--external\][\s\S]*--base-url/u);
  assert.deepEqual(fixture.managerRoots, []);
});

void test('psr setup --local writes a verified local profile atomically', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'psr-setup-local-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const configPath = join(home, 'config', 'recall.json');
  const modelRoot = join(home, 'models');
  const fixture = createFixture({ homeDirectory: home });

  const result = await runPsrSetupCli(
    ['--local', '--yes', '--config', configPath, '--model-root', modelRoot],
    fixture.dependencies,
  );

  assert.deepEqual(result, { exitCode: 0, runInitialIndex: false });
  assert.deepEqual(fixture.managerRoots, [modelRoot]);
  assert.deepEqual(fixture.downloadApprovals, [true]);
  assert.deepEqual(JSON.parse(await readFile(configPath, 'utf8')), {
    embeddingProfile: RecallEmbeddingProfile.LOCAL_OCTEN,
    localModelRootDirectory: modelRoot,
  });
  assert.equal((await stat(configPath)).mode & 0o777, 0o600);
  assert.match(
    fixture.output.join(''),
    /Local Octen embeddings configured[\s\S]*psr index --rebuild/u,
  );
});

void test('psr setup preserves unrelated settings and removes stale HTTP fields', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'psr-setup-preserve-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const configPath = join(home, 'recall.json');
  await writeFile(
    configPath,
    `${JSON.stringify({
      sessionsDirectory: '/sessions',
      projectLineages: { project: ['/old'] },
      embeddingProfile: RecallEmbeddingProfile.OCTEN_HTTP,
      embeddingBaseUrl: 'http://old.test/v1',
      embeddingModel: 'old',
      embeddingServedModelId: 'old/served',
      embeddingNativeDimensions: 2_560,
      embeddingStoredDimensions: 1_024,
      embeddingBatchSize: 16,
    })}\n`,
    'utf8',
  );
  const fixture = createFixture({ homeDirectory: home });

  await runPsrSetupCli(['--local', '--yes', '--config', configPath], fixture.dependencies);

  assert.deepEqual(JSON.parse(await readFile(configPath, 'utf8')), {
    sessionsDirectory: '/sessions',
    projectLineages: { project: ['/old'] },
    embeddingProfile: RecallEmbeddingProfile.LOCAL_OCTEN,
    localModelRootDirectory: join(home, '.pi', 'agent', 'recall-models'),
  });
});

void test('psr setup cancellation leaves existing configuration unchanged', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'psr-setup-cancel-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const configPath = join(home, 'recall.json');
  const existing = '{"embeddingProfile":"octen-http-v1"}\n';
  await writeFile(configPath, existing, 'utf8');
  const fixture = createFixture({ homeDirectory: home, confirmations: [false] });

  const result = await runPsrSetupCli(['--local', '--config', configPath], fixture.dependencies);

  assert.deepEqual(result, { exitCode: 1, runInitialIndex: false });
  assert.equal(await readFile(configPath, 'utf8'), existing);
  assert.deepEqual(fixture.downloadApprovals, []);
});

void test('psr setup rejects unsupported existing settings before model access', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'psr-setup-invalid-existing-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const configPath = join(home, 'recall.json');
  const existing = '{"unsupportedSetting":true}\n';
  await writeFile(configPath, existing, 'utf8');
  const fixture = createFixture({ homeDirectory: home });

  await assert.rejects(
    runPsrSetupCli(['--local', '--yes', '--config', configPath], fixture.dependencies),
    /Recall configuration invalid/u,
  );

  assert.equal(await readFile(configPath, 'utf8'), existing);
  assert.deepEqual(fixture.managerRoots, []);
});

void test('psr setup does not write local config when model download is declined', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'psr-setup-download-cancel-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const configPath = join(home, 'recall.json');
  const fixture = createFixture({ homeDirectory: home, confirmations: [false] });

  const result = await runPsrSetupCli(['--local', '--config', configPath], fixture.dependencies);

  assert.deepEqual(result, { exitCode: 1, runInitialIndex: false });
  await assert.rejects(readFile(configPath, 'utf8'), { code: 'ENOENT' });
  assert.deepEqual(fixture.downloadApprovals, []);
});

void test('psr setup writes an explicit external HTTP profile without model access', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'psr-setup-external-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const configPath = join(home, 'recall.json');
  const fixture = createFixture({ homeDirectory: home });

  const result = await runPsrSetupCli(
    [
      '--external',
      '--yes',
      '--index',
      '--config',
      configPath,
      '--base-url',
      'http://127.0.0.1:8090/v1',
      '--model',
      'octen-local-server',
      '--served-model-id',
      'Octen/Octen-Embedding-4B',
      '--native-dimensions',
      '2560',
      '--batch-size',
      '64',
    ],
    fixture.dependencies,
  );

  assert.deepEqual(result, { exitCode: 0, runInitialIndex: true });
  assert.deepEqual(fixture.managerRoots, []);
  assert.deepEqual(JSON.parse(await readFile(configPath, 'utf8')), {
    embeddingProfile: RecallEmbeddingProfile.OCTEN_HTTP,
    embeddingBaseUrl: 'http://127.0.0.1:8090/v1',
    embeddingModel: 'octen-local-server',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions: 2_560,
    embeddingStoredDimensions: 1_024,
    embeddingBatchSize: 64,
  });
});

void test('interactive external setup prompts for endpoint and model details', async (t) => {
  const home = await mkdtemp(join(tmpdir(), 'psr-setup-interactive-external-'));
  t.after(() => rm(home, { recursive: true, force: true }));
  const configPath = join(home, 'recall.json');
  const fixture = createFixture({
    homeDirectory: home,
    selectedProfile: RecallEmbeddingProfile.OCTEN_HTTP,
    promptResponses: [
      'https://embeddings.example.test/v1',
      'octen-request',
      'Octen/Octen-Embedding-4B',
      '2560',
      '32',
    ],
  });

  const result = await runPsrSetupCli(['--yes', '--config', configPath], fixture.dependencies);

  assert.deepEqual(result, { exitCode: 0, runInitialIndex: false });
  assert.deepEqual(JSON.parse(await readFile(configPath, 'utf8')), {
    embeddingProfile: RecallEmbeddingProfile.OCTEN_HTTP,
    embeddingBaseUrl: 'https://embeddings.example.test/v1',
    embeddingModel: 'octen-request',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions: 2_560,
    embeddingStoredDimensions: 1_024,
    embeddingBatchSize: 32,
  });
});

void test('psr setup rejects ambiguous modes and incomplete external settings', async () => {
  const fixture = createFixture({ homeDirectory: '/tmp/psr-setup-invalid' });

  await assert.rejects(
    runPsrSetupCli(['--local', '--external', '--yes'], fixture.dependencies),
    /psr setup \[--local\|--external\]/u,
  );
  await assert.rejects(
    runPsrSetupCli(['--external', '--yes'], fixture.dependencies),
    /--external requires --base-url/u,
  );
});
