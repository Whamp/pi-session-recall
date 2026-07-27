import type { RecallConversationConfig } from './recall-conversation-config.js';
import type { RecallQualityGateDecision } from './recall-quality-gate.js';

/** Applies the quality gate's selected chunk and candidate policy without changing other config. */
export function applyRecallQualityPolicyToConversationConfig(
  config: RecallConversationConfig,
  decision: RecallQualityGateDecision,
): RecallConversationConfig {
  const selectedPolicy = decision.selectedPolicy;
  return selectedPolicy
    ? {
        ...config,
        chunkPolicy: {
          maxTokens: selectedPolicy.chunkPolicy.maxTokens,
          overlapTokens: selectedPolicy.chunkPolicy.overlapTokens,
        },
        searchCandidateLimits: {
          dense: selectedPolicy.candidateCount,
          lexical: selectedPolicy.candidateCount,
          identifier: selectedPolicy.candidateCount,
        },
      }
    : config;
}
