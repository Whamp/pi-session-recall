import { createHash } from 'node:crypto';

import { RECALL_EMBEDDING_CANARY_TEXT } from './recall-index-manifest.js';
import type { RecallQualityEvaluationDependencies } from './run-recall-quality-evaluation.js';

/** Fixed dimensions used only by the committed synthetic quality corpus. */
export const DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS = 64;

const QUALITY_CONCEPT_PATTERNS = [
  /queued deliveries|worker crash|append-only sqlite outbox|remote acknowledgement|outbox/iu,
  /failed picture|picture ingestion|decorrelated jitter|forty-two seconds|amber-orbit-17/iu,
  /rollout in europe|european rollout|paris queue|frankfurt replica|maintenance window/iu,
  /discarded redis|redis approach|redis streams|glacier-lantern/iu,
  /offline laptops|sqlite wal queue|idempotent upload acknowledgements/iu,
  /hosted queue services|without a network connection/iu,
  /copper finch|snapshot delta-29|device certificate|recovery steps/iu,
  /release meridian|sha256:4c91d7e2|meridian-safe-3/iu,
] as const;

function createDeterministicRecallQualityVector(text: string): number[] {
  const values: number[] = Array.from(
    { length: DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS },
    () => 0,
  );
  if (text === RECALL_EMBEDDING_CANARY_TEXT) {
    values[DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS - 1] = 1;
    return values;
  }
  for (const [index, pattern] of QUALITY_CONCEPT_PATTERNS.entries()) {
    if (pattern.test(text)) {
      values[index] = 1_000;
    }
  }
  for (const token of text.toLowerCase().match(/[a-z0-9_.:/-]+/gu) ?? []) {
    const digest = createHash('sha256').update(token).digest();
    const bucket = 8 + ((digest[0] ?? 0) % 55);
    values[bucket] = (values[bucket] ?? 0) + 1;
  }
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  if (norm === 0) {
    return values.map((value, index) =>
      index === DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS - 2 ? 1 : value,
    );
  }
  return values.map((value) => value / norm);
}

/** Creates network-free deterministic model and tokenizer boundaries for the fixed quality corpus. */
export function createDeterministicRecallQualityDependencies(): RecallQualityEvaluationDependencies {
  return {
    embeddingProvider: {
      embedQuery(query) {
        const vector = createDeterministicRecallQualityVector(query);
        return Promise.resolve(vector);
      },
      async embedDocuments(documents) {
        return documents.map(createDeterministicRecallQualityVector);
      },
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text) {
          const tokenCount = text.split(/\s+/u).filter(Boolean).length;
          return { ids: Array.from({ length: tokenCount }, (_, index) => index) };
        },
      };
    },
  };
}
