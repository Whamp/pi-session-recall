import { dirname, join } from 'node:path';

import { RecallInstallationMode } from './enums.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import {
  readRecallFirstIndexSetupState,
  resolveRecallFirstIndexSetupStatePath,
} from './recall-first-index-setup-command.js';
import { readRecallInferenceConfiguration } from './recall-inference-configuration.js';

function resolveRecallInferenceConfigurationPath(config: RecallConversationConfig): string {
  return join(dirname(config.manifestPath), 'inference-configuration.json');
}

/** Distinguishes verified target-generation inference setup from an unconfigured installation. */
export async function resolveRecallInstallationMode(
  config: RecallConversationConfig,
): Promise<RecallInstallationMode> {
  const inferenceConfiguration = await readRecallInferenceConfiguration(
    resolveRecallInferenceConfigurationPath(config),
  );
  const firstIndexSetupState = await readRecallFirstIndexSetupState(
    resolveRecallFirstIndexSetupStatePath(config),
  );
  return inferenceConfiguration.embedding || firstIndexSetupState.embedding
    ? RecallInstallationMode.CONFIGURED
    : RecallInstallationMode.UNCONFIGURED;
}

/** Refuses recall until setup verifies an embedding capability for target generations. */
export function assertRecallInstallationConfigured(mode: RecallInstallationMode): void {
  if (mode === RecallInstallationMode.CONFIGURED) {
    return;
  }
  throw new Error(
    'Recall inference is unconfigured: run pi-session-recall setup and configure a verified embedding before search or rebuild',
  );
}
