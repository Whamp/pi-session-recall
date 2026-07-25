import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

/** Highest final-result count accepted from evidence and exposed by the recall tool. */
export const MAX_RECALL_FINAL_RESULT_COUNT = 10;

/** Packaged bounded-evaluation evidence consulted before production indexing. */
export const RECALL_QUALITY_RESULTS_PATH = fileURLToPath(
  new URL('../docs/evaluation/recall-quality-results.json', import.meta.url),
);

/** Exact measured chunk, candidate, and final-result policy from a clean passing gate. */
export interface RecallQualityApprovedPolicy {
  chunkPolicy: {
    id: string;
    maxTokens: number;
    overlapTokens: number;
  };
  candidateCount: number;
  finalCount: number;
}

/** Automated backfill decision reconstructed from committed bounded-evaluation evidence. */
export interface RecallQualityGateDecision {
  automatedGatePassed: boolean;
  selectedPolicy: RecallQualityApprovedPolicy | null;
  blockers: string[];
}

const selectedPolicySchema = Type.Object({
  chunkPolicy: Type.Object({
    id: Type.String({ minLength: 1 }),
    maxTokens: Type.Integer({ minimum: 1, maximum: 1_024 }),
    overlapTokens: Type.Integer({ minimum: 0, maximum: 128 }),
  }),
  candidateCount: Type.Integer({ minimum: 1, maximum: 200 }),
  finalCount: Type.Integer({ minimum: 1, maximum: MAX_RECALL_FINAL_RESULT_COUNT }),
  gatePassed: Type.Boolean(),
});

const recallQualityGateEvidenceSchema = Type.Object({
  version: Type.Literal(1),
  environment: Type.Object({
    gitDirty: Type.Boolean(),
  }),
  result: Type.Object({
    version: Type.Integer({ minimum: 1 }),
    selection: Type.Object({
      passed: Type.Boolean(),
      selected: Type.Union([Type.Null(), selectedPolicySchema]),
      blockers: Type.Array(Type.String()),
    }),
  }),
});

/** Reads committed quality evidence and returns a policy only for a consistent clean pass. */
export async function readRecallQualityGateDecision(
  resultsPath: string,
): Promise<RecallQualityGateDecision> {
  let evidence: ReturnType<typeof Value.Parse<typeof recallQualityGateEvidenceSchema>>;
  try {
    const parsed: unknown = JSON.parse(await readFile(resultsPath, 'utf8'));
    evidence = Value.Parse(recallQualityGateEvidenceSchema, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall quality gate evidence invalid at ${resultsPath}: ${message}`, {
      cause: error,
    });
  }

  const { selection } = evidence.result;
  if (selection.passed !== (selection.selected !== null && selection.selected.gatePassed)) {
    throw new Error(
      `Recall quality gate evidence inconsistent at ${resultsPath}: pass and selected-policy decisions disagree`,
    );
  }
  if (evidence.result.version !== 3) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        `Recall quality evidence version ${evidence.result.version} predates rerank-free fused top-N measurement; rerun npm run evaluate:recall`,
      ],
    };
  }
  if (!selection.passed) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [...selection.blockers],
    };
  }
  if (evidence.environment.gitDirty) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: ['Recall quality evidence was generated from a dirty worktree'],
    };
  }
  const selected = selection.selected;
  if (!selected) {
    throw new Error(
      `Recall quality gate evidence inconsistent at ${resultsPath}: passing selection is missing`,
    );
  }
  return {
    automatedGatePassed: true,
    selectedPolicy: {
      chunkPolicy: { ...selected.chunkPolicy },
      candidateCount: selected.candidateCount,
      finalCount: selected.finalCount,
    },
    blockers: [],
  };
}
