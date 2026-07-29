import assert from 'node:assert/strict';
import test from 'node:test';

import { createRecallDiagnosticHostIdentity } from './create-recall-diagnostic-host-identity.js';

void test('diagnostic host identity records pinned zvec and current runtime scalars', () => {
  const identity = createRecallDiagnosticHostIdentity();

  assert.equal(identity.nodeVersion, process.version);
  assert.equal(identity.zvecVersion, '0.6.0');
  assert.ok(identity.platform.length > 0);
  assert.ok(identity.architecture.length > 0);
  assert.ok(identity.cpuModel.length > 0);
});
