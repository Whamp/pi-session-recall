import assert from 'node:assert/strict';
import test from 'node:test';

import { confirmTerminalAction, promptTerminalText } from './terminal-confirm.js';

void test('terminal prompts reject noninteractive use before reading stdin', async (t) => {
  const stdinDescriptor = Object.getOwnPropertyDescriptor(process.stdin, 'isTTY');
  const stderrDescriptor = Object.getOwnPropertyDescriptor(process.stderr, 'isTTY');
  Object.defineProperty(process.stdin, 'isTTY', { configurable: true, value: false });
  Object.defineProperty(process.stderr, 'isTTY', { configurable: true, value: false });
  t.after(() => {
    if (stdinDescriptor) {
      Object.defineProperty(process.stdin, 'isTTY', stdinDescriptor);
    } else {
      Reflect.deleteProperty(process.stdin, 'isTTY');
    }
    if (stderrDescriptor) {
      Object.defineProperty(process.stderr, 'isTTY', stderrDescriptor);
    } else {
      Reflect.deleteProperty(process.stderr, 'isTTY');
    }
  });

  await assert.rejects(confirmTerminalAction('Proceed?'), /interactive terminal/u);
  await assert.rejects(promptTerminalText('Value', 'default'), /interactive terminal/u);
});
