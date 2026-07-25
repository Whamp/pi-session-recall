import assert from 'node:assert/strict';
import test from 'node:test';

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

import recallExtension from './recall-extension.js';

void test('recall extension registers the recall tool guidance and explicit index command', async () => {
  const toolNames: string[] = [];
  const toolGuidelines: string[] = [];
  const commandNames: string[] = [];
  const registrar: Pick<ExtensionAPI, 'registerTool' | 'registerCommand'> = {
    registerTool(definition) {
      toolNames.push(definition.name);
      toolGuidelines.push(...(definition.promptGuidelines ?? []));
    },
    registerCommand(name) {
      commandNames.push(name);
    },
  };

  await recallExtension(registrar);

  assert.deepEqual(toolNames, ['recall']);
  assert.deepEqual(commandNames, ['recall-index']);
  assert.ok(
    toolGuidelines.some(
      (guideline) =>
        guideline.includes('Use recall') &&
        guideline.includes('conversation or detail from a past session'),
    ),
  );
});
