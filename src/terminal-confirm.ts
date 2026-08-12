import { createInterface } from 'node:readline/promises';

function assertInteractiveTerminal(): void {
  if (!process.stdin.isTTY || !process.stderr.isTTY) {
    throw new Error('Confirmation requires an interactive terminal; rerun with explicit flags');
  }
}

/** Asks one explicit yes/no question and defaults every non-yes answer to no. */
export async function confirmTerminalAction(question: string): Promise<boolean> {
  assertInteractiveTerminal();
  const interface_ = createInterface({ input: process.stdin, output: process.stderr });
  try {
    const answer = await interface_.question(`${question} [y/N] `);
    return answer.trim().toLowerCase() === 'y' || answer.trim().toLowerCase() === 'yes';
  } finally {
    interface_.close();
  }
}

/** Reads one terminal value, accepting the displayed default on a blank answer. */
export async function promptTerminalText(question: string, defaultValue: string): Promise<string> {
  assertInteractiveTerminal();
  const interface_ = createInterface({ input: process.stdin, output: process.stderr });
  try {
    const answer = await interface_.question(`${question} [${defaultValue}] `);
    return answer.trim() || defaultValue;
  } finally {
    interface_.close();
  }
}
