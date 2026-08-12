import { createInterface } from 'node:readline/promises';

/** Asks one explicit yes/no question and defaults every non-yes answer to no. */
export async function confirmTerminalAction(question: string): Promise<boolean> {
  if (!process.stdin.isTTY || !process.stderr.isTTY) {
    throw new Error('Confirmation requires an interactive terminal; rerun with --yes');
  }
  const interface_ = createInterface({ input: process.stdin, output: process.stderr });
  try {
    const answer = await interface_.question(`${question} [y/N] `);
    return answer.trim().toLowerCase() === 'y' || answer.trim().toLowerCase() === 'yes';
  } finally {
    interface_.close();
  }
}
