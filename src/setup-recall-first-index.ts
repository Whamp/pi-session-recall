import { pathToFileURL } from 'node:url';

import { applyRecallQualityPolicyToConversationConfig } from './apply-recall-quality-policy.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { runRecallFirstIndexSetupCommand } from './recall-first-index-setup-command.js';
import {
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

/** Loads local recall paths and runs one deterministic first-index setup command. */
export async function runRecallFirstIndexSetupCli(argumentsList: readonly string[]): Promise<void> {
  const qualityGateDecision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);
  const configured = await loadRecallConversationConfig();
  const config = applyRecallQualityPolicyToConversationConfig(configured, qualityGateDecision);
  await runRecallFirstIndexSetupCommand(argumentsList, { config, qualityGateDecision });
}

const invokedPath = process.argv[1];
if (invokedPath && import.meta.url === pathToFileURL(invokedPath).href) {
  runRecallFirstIndexSetupCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
