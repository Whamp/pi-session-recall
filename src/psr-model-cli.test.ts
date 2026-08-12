import assert from 'node:assert/strict';
import test from 'node:test';

import {
  LocalOctenModelDownloadProgressKind,
  LocalOctenModelStatusKind,
} from './enums.js';
import type { LocalOctenModelManager } from './local-octen-model-manager.js';
import { runPsrModelCli } from './psr-model-cli.js';

function createFixture(options: {
  statusKind?: LocalOctenModelStatusKind;
  approved?: boolean;
  doctorHealthy?: boolean;
} = {}) {
  const output: string[] = [];
  const progress: string[] = [];
  let downloadCalls = 0;
  let doctorCalls = 0;
  const status = {
    kind: options.statusKind ?? LocalOctenModelStatusKind.MISSING,
    artifactId: 'local-octen-test-v1',
    modelDirectory: '/models/local-octen-test-v1',
    totalBytes: 1_073_741_824,
    partialDownloads: 0,
    detail: 'test model status',
  };
  const manager: LocalOctenModelManager = {
    async status() {
      return status;
    },
    async download(downloadOptions) {
      downloadCalls += 1;
      assert.equal(downloadOptions.approved, true);
      downloadOptions.onProgress?.({
        kind: LocalOctenModelDownloadProgressKind.DOWNLOADING_FILE,
        fileName: 'model.onnx.data',
        completedBytes: 0,
        totalBytes: status.totalBytes,
      });
      downloadOptions.onProgress?.({
        kind: LocalOctenModelDownloadProgressKind.ACTIVATED,
        completedBytes: status.totalBytes,
        totalBytes: status.totalBytes,
      });
      return { downloaded: true, modelDirectory: status.modelDirectory };
    },
    async doctor() {
      doctorCalls += 1;
      const healthy = options.doctorHealthy ?? true;
      return {
        healthy,
        status,
        detail: healthy ? 'runtime healthy' : 'runtime failed',
        ...(healthy ? { runtime: { dimensions: 1_024, norm: 1 } } : {}),
      };
    },
  };
  return {
    output,
    progress,
    get downloadCalls() {
      return downloadCalls;
    },
    get doctorCalls() {
      return doctorCalls;
    },
    dependencies: {
      async loadManager() {
        return manager;
      },
      async confirm() {
        return options.approved ?? false;
      },
      writeOutput(text: string) {
        output.push(text);
      },
      writeProgress(text: string) {
        progress.push(text);
      },
    },
  };
}

void test('psr model status reports cache state without downloading or probing', async () => {
  const fixture = createFixture({ statusKind: LocalOctenModelStatusKind.READY });

  assert.equal(await runPsrModelCli(['status'], fixture.dependencies), 0);
  assert.match(fixture.output.join(''), /ready[\s\S]*1\.00 GiB[\s\S]*\/models\/local/u);
  assert.equal(fixture.downloadCalls, 0);
  assert.equal(fixture.doctorCalls, 0);
});

void test('psr model download asks for approval before downloading one GiB', async () => {
  const declined = createFixture({ approved: false });

  assert.equal(await runPsrModelCli(['download'], declined.dependencies), 1);
  assert.equal(declined.downloadCalls, 0);
  assert.match(declined.output.join(''), /cancelled/u);

  const approved = createFixture({ approved: true });
  assert.equal(await runPsrModelCli(['download'], approved.dependencies), 0);
  assert.equal(approved.downloadCalls, 1);
  assert.match(approved.progress.join(''), /Downloading model\.onnx\.data[\s\S]*activated/u);
});

void test('psr model download --yes is the noninteractive approval path', async () => {
  const fixture = createFixture({ approved: false });

  assert.equal(await runPsrModelCli(['download', '--yes'], fixture.dependencies), 0);
  assert.equal(fixture.downloadCalls, 1);
});

void test('psr model doctor returns failure when artifact or runtime diagnosis fails', async () => {
  const fixture = createFixture({ doctorHealthy: false });

  assert.equal(await runPsrModelCli(['doctor'], fixture.dependencies), 1);
  assert.equal(fixture.doctorCalls, 1);
  assert.match(fixture.output.join(''), /unhealthy[\s\S]*runtime failed/u);
});

void test('psr model rejects unsupported subcommands and flags', async () => {
  const fixture = createFixture();

  await assert.rejects(runPsrModelCli(['repair'], fixture.dependencies), /psr model status/u);
  await assert.rejects(
    runPsrModelCli(['status', '--yes'], fixture.dependencies),
    /psr model status/u,
  );
});
