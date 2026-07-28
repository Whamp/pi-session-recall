import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { applyRecallQualityPolicyToConversationConfig } from './applyRecallQualityPolicyToConversationConfig.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { runRecallFirstIndexSetupCommand } from './recall-first-index-setup-command.js';
import {
  readRecallInferenceConfiguration,
  type RecallInferenceConfigurationCandidate,
} from './recall-inference-configuration.js';
import { runRecallInferenceSetupCommand } from './runRecallInferenceSetupCommand.js';
import {
  createRecommendedEmbeddingGemmaHttpInferenceCandidate,
  createRecommendedEmbeddingGemmaInferenceCandidate,
} from './recommended-embeddinggemma-inference-candidate.js';
import { createRecommendedEmbeddingGemmaConversationRuntime } from './recommended-embeddinggemma-conversation-service.js';
import {
  createRecommendedOptionalInferenceCandidates,
  readRecommendedOptionalInferenceConformance,
} from './createRecommendedOptionalInferenceCandidates.js';
import {
  createConfiguredRecallInferenceRuntime,
  resolveRecallInferenceConfigurationPath,
  type RecallInferenceAdapterRegistry,
} from './configured-recall-inference-runtime.js';
import {
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

/** Loads local recall paths and runs first-index or mixed-inference setup commands. */
export async function runRecallFirstIndexSetupCli(
  argumentsList: readonly string[],
  inferenceCandidates?: readonly RecallInferenceConfigurationCandidate[],
  adapterRegistries: readonly RecallInferenceAdapterRegistry[] = [],
): Promise<void> {
  const qualityGateDecision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);
  const configured = await loadRecallConversationConfig();
  const config = applyRecallQualityPolicyToConversationConfig(configured, qualityGateDecision);
  if (argumentsList[0] === 'inference') {
    const optionalConformance = await readRecommendedOptionalInferenceConformance(
      join(dirname(config.manifestPath), 'inference-conformance.json'),
    );
    await runRecallInferenceSetupCommand(argumentsList.slice(1), {
      statePath: resolveRecallInferenceConfigurationPath(config),
      ...(config.activeGenerationPath ? { activeGenerationPath: config.activeGenerationPath } : {}),
      candidates: inferenceCandidates ?? [
        createRecommendedEmbeddingGemmaInferenceCandidate(config),
        createRecommendedEmbeddingGemmaHttpInferenceCandidate(config),
        ...createRecommendedOptionalInferenceCandidates(config, optionalConformance),
        ...adapterRegistries.flatMap((registry) => registry.candidates),
      ],
    });
    return;
  }
  await runRecallFirstIndexSetupCommand(argumentsList, {
    config,
    async createConfiguredServiceRuntime() {
      const inferenceConfiguration = await readRecallInferenceConfiguration(
        resolveRecallInferenceConfigurationPath(config),
        {
          ...(config.activeGenerationPath
            ? { activeGenerationPath: config.activeGenerationPath }
            : {}),
        },
      );
      return inferenceConfiguration.embedding
        ? createConfiguredRecallInferenceRuntime(config, { adapterRegistries })
        : createRecommendedEmbeddingGemmaConversationRuntime(config);
    },
  });
}

const INVOKED_PATH = process.argv[1];
if (INVOKED_PATH && import.meta.url === pathToFileURL(INVOKED_PATH).href) {
  runRecallFirstIndexSetupCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
