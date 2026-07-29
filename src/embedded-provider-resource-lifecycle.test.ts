import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createEmbeddedProviderResourceLifecycle,
  disposeEmbeddedProviderResourceLayers,
} from './embedded-provider-resource-lifecycle.js';
import { EmbeddedInferenceComputeBackend } from './enums.js';

interface FixtureResources {
  id: number;
}

function createFixtureInitialization(options: {
  resources: FixtureResources;
  fallbackWarningEmitted?: boolean;
  disposeResources(): Promise<void>;
}) {
  return {
    resources: options.resources,
    async disposeResources() {
      await options.disposeResources();
    },
    selectedComputeBackend: EmbeddedInferenceComputeBackend.CPU,
    fallbackFromComputeBackend: null,
    fallbackWarningEmitted: options.fallbackWarningEmitted ?? false,
    probedComputeBackends: [],
    deviceNames: [],
  };
}

void test('embedded provider resource lifecycle shares one load across concurrent operations', async () => {
  const loadGate = Promise.withResolvers<void>();
  let loadCount = 0;
  let disposalCount = 0;
  const lifecycle = createEmbeddedProviderResourceLifecycle<FixtureResources>({
    disposedErrorMessage: 'Fixture provider disposed',
    idleTimeoutMilliseconds: 0,
    async loadResources() {
      loadCount += 1;
      await loadGate.promise;
      return createFixtureInitialization({
        resources: { id: loadCount },
        async disposeResources() {
          disposalCount += 1;
        },
      });
    },
    writeIdleDisposalWarning() {},
  });

  const first = lifecycle.runWithResources(async ({ id }) => id);
  const second = lifecycle.runWithResources(async ({ id }) => id);
  await new Promise<void>((resolve) => {
    setImmediate(resolve);
  });
  assert.equal(loadCount, 1);

  loadGate.resolve();
  assert.deepEqual(await Promise.all([first, second]), [1, 1]);
  await lifecycle.dispose();
  await lifecycle.dispose();
  assert.equal(disposalCount, 1);
  await assert.rejects(
    () => lifecycle.runWithResources(async ({ id }) => id),
    /Fixture provider disposed/u,
  );
});

void test('embedded provider resource lifecycle disposes only after every operation becomes idle', async () => {
  const releaseLongOperation = Promise.withResolvers<void>();
  const idleDisposal = Promise.withResolvers<void>();
  let operationCount = 0;
  let disposalCount = 0;
  const lifecycle = createEmbeddedProviderResourceLifecycle<FixtureResources>({
    disposedErrorMessage: 'Fixture provider disposed',
    idleTimeoutMilliseconds: 5,
    async loadResources() {
      return createFixtureInitialization({
        resources: { id: 1 },
        async disposeResources() {
          disposalCount += 1;
          idleDisposal.resolve();
        },
      });
    },
    writeIdleDisposalWarning() {},
  });

  const longOperation = lifecycle.runWithResources(async () => {
    operationCount += 1;
    await releaseLongOperation.promise;
  });
  await lifecycle.runWithResources(async () => {
    operationCount += 1;
  });
  assert.equal(operationCount, 2);
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 15);
  });
  assert.equal(disposalCount, 0);

  releaseLongOperation.resolve();
  await longOperation;
  await Promise.race([
    idleDisposal.promise,
    new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('Fixture resources were not disposed when idle')), 500);
    }),
  ]);
  assert.equal(disposalCount, 1);
  await lifecycle.dispose();
});

void test('embedded provider resource lifecycle carries CPU fallback warning state across reloads', async () => {
  const fallbackWarningInputs: boolean[] = [];
  const idleDisposals: Array<PromiseWithResolvers<void>> = [];
  let nextResourceId = 0;
  const lifecycle = createEmbeddedProviderResourceLifecycle<FixtureResources>({
    disposedErrorMessage: 'Fixture provider disposed',
    idleTimeoutMilliseconds: 5,
    async loadResources(fallbackWarningAlreadyEmitted) {
      fallbackWarningInputs.push(fallbackWarningAlreadyEmitted);
      nextResourceId += 1;
      const idleDisposal = Promise.withResolvers<void>();
      idleDisposals.push(idleDisposal);
      return createFixtureInitialization({
        resources: { id: nextResourceId },
        fallbackWarningEmitted: true,
        async disposeResources() {
          idleDisposal.resolve();
        },
      });
    },
    writeIdleDisposalWarning() {},
  });

  assert.equal(await lifecycle.runWithResources(async ({ id }) => id), 1);
  const firstIdleDisposal = idleDisposals[0];
  assert.ok(firstIdleDisposal);
  await firstIdleDisposal.promise;
  assert.equal(await lifecycle.runWithResources(async ({ id }) => id), 2);
  assert.deepEqual(fallbackWarningInputs, [false, true]);
  await lifecycle.dispose();
});

void test('embedded provider disposal layers preserve order and aggregate every failure', async () => {
  const events: string[] = [];
  const contextError = new Error('fixture context disposal failed');
  const modelError = new Error('fixture model disposal failed');

  await assert.rejects(
    () =>
      disposeEmbeddedProviderResourceLayers({
        failureMessage: 'Fixture layered disposal failed',
        layers: [
          [async () => void events.push('operation')],
          [
            async () => {
              events.push('context');
              throw contextError;
            },
          ],
          [
            async () => {
              events.push('model');
              throw modelError;
            },
          ],
          [async () => void events.push('runtime')],
        ],
      }),
    (error: unknown) => {
      assert.ok(error instanceof AggregateError);
      assert.equal(error.message, 'Fixture layered disposal failed');
      assert.deepEqual(error.errors, [contextError, modelError]);
      return true;
    },
  );
  assert.deepEqual(events, ['operation', 'context', 'model', 'runtime']);
});
