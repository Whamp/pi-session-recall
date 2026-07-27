import { pathToFileURL } from 'node:url';

import { applyRecallQualityPolicyToConversationConfig } from './apply-recall-quality-policy.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { runRecallFirstIndexSetupCommand } from './recall-first-index-setup-command.js';
import type { RecallInferenceConfigurationCandidate } from './recall-inference-configuration.js';
import { runRecallInferenceSetupCommand } from './recall-inference-setup-command.js';
import {
  createRecommendedEmbeddingGemmaHttpInferenceCandidate,
  createRecommendedEmbeddingGemmaInferenceCandidate,
} from './recommended-embeddinggemma-inference-candidate.js';
import { createRecommendedOptionalInferenceCandidates } from './recommended-optional-inference-candidates.js';
import { resolveRecallInferenceConfigurationPath } from './configured-recall-inference-runtime.js';
import {
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

/** Loads local recall paths and runs first-index or mixed-inference setup commands. */
export async function runRecallFirstIndexSetupCli(
  argumentsList: readonly string[],
  inferenceCandidates?: readonly RecallInferenceConfigurationCandidate[],
): Promise<void> {
  const qualityGateDecision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);
  const configured = await loadRecallConversationConfig();
  const config = applyRecallQualityPolicyToConversationConfig(configured, qualityGateDecision);
  if (argumentsList[0] === 'inference') {
    await runRecallInferenceSetupCommand(argumentsList.slice(1), {
      statePath: resolveRecallInferenceConfigurationPath(config),
      candidates: inferenceCandidates ?? [
        createRecommendedEmbeddingGemmaInferenceCandidate(config),
        createRecommendedEmbeddingGemmaHttpInferenceCandidate(config),
        ...createRecommendedOptionalInferenceCandidates(config),
      ],
    });
    return;
  }
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
