import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RECALL_PROJECT_SCOPE_POLICY_VERSION, RecallSearchScope } from './enums.js';
import {
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-search-candidates.js';
import { RECALL_ACTIVE_BRANCH_PRIOR } from './rank-recall-search-results.js';
import {
  createRecallProjectLineageDigest,
  RECALL_PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
  RECALL_PROJECT_IDENTITY_POLICY_VERSION,
  RECALL_PROJECT_LINEAGE_POLICY_VERSION,
} from './resolve-project-identity.js';

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

const evaluationIdentitySchema = Type.Object({
  defaultScope: Type.Enum(RecallSearchScope),
  projectScopePolicyVersion: Type.Integer({ minimum: 1 }),
  repositoryIdentityPolicyVersion: Type.Integer({ minimum: 1 }),
  projectIdentityMetadataSchemaVersion: Type.Integer({ minimum: 1 }),
  lineagePolicyVersion: Type.Integer({ minimum: 1 }),
  lineageDigest: Type.String({ pattern: '^[a-f0-9]{64}$' }),
  rankingMode: Type.String(),
  rankFusionVersion: Type.Integer({ minimum: 1 }),
  reciprocalRankConstant: Type.Number({ exclusiveMinimum: 0 }),
  activeBranchPrior: Type.Number({ minimum: 0 }),
  candidateLimits: Type.Object({
    dense: Type.Integer({ minimum: 1 }),
    lexical: Type.Integer({ minimum: 1 }),
    identifier: Type.Integer({ minimum: 1 }),
  }),
  finalResultCount: Type.Integer({ minimum: 1, maximum: MAX_RECALL_FINAL_RESULT_COUNT }),
});

const recallQualityGateEvidenceSchema = Type.Object({
  version: Type.Integer({ minimum: 1 }),
  environment: Type.Object({
    gitDirty: Type.Boolean(),
  }),
  specification: Type.Optional(
    Type.Object({
      version: Type.Integer({ minimum: 1 }),
      projectLineages: Type.Optional(
        Type.Record(Type.String({ minLength: 1 }), Type.Array(Type.String({ minLength: 1 }))),
      ),
    }),
  ),
  result: Type.Object({
    version: Type.Integer({ minimum: 1 }),
    evaluationIdentity: Type.Optional(evaluationIdentitySchema),
    selection: Type.Object({
      passed: Type.Boolean(),
      selected: Type.Union([Type.Null(), selectedPolicySchema]),
      blockers: Type.Array(Type.String()),
      combinations: Type.Optional(Type.Array(selectedPolicySchema)),
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
  if (evidence.version !== 2 || evidence.result.version !== 4) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        `Recall quality evidence version ${evidence.result.version} predates project-scoped measurement; rerun npm run evaluate:recall`,
      ],
    };
  }
  const evaluationIdentity = evidence.result.evaluationIdentity;
  const projectLineages = evidence.specification?.projectLineages;
  const expectedLineageDigest = projectLineages
    ? createRecallProjectLineageDigest(projectLineages)
    : null;
  if (
    !evaluationIdentity ||
    evidence.specification?.version !== 3 ||
    evaluationIdentity.defaultScope !== RecallSearchScope.PROJECT ||
    evaluationIdentity.projectScopePolicyVersion !== RECALL_PROJECT_SCOPE_POLICY_VERSION ||
    evaluationIdentity.repositoryIdentityPolicyVersion !== RECALL_PROJECT_IDENTITY_POLICY_VERSION ||
    evaluationIdentity.projectIdentityMetadataSchemaVersion !==
      RECALL_PROJECT_IDENTITY_METADATA_SCHEMA_VERSION ||
    evaluationIdentity.lineagePolicyVersion !== RECALL_PROJECT_LINEAGE_POLICY_VERSION ||
    evaluationIdentity.lineageDigest !== expectedLineageDigest
  ) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        'Recall quality project identity does not match the current scope, repository, or lineage policy; rerun npm run evaluate:recall',
      ],
    };
  }
  if (
    evaluationIdentity.rankingMode !== 'hybrid' ||
    evaluationIdentity.rankFusionVersion !== RECALL_RANK_FUSION_VERSION ||
    evaluationIdentity.reciprocalRankConstant !== RECALL_RRF_RANK_CONSTANT ||
    evaluationIdentity.activeBranchPrior !== RECALL_ACTIVE_BRANCH_PRIOR
  ) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        'Recall quality ranking identity does not match the current hybrid policy; rerun npm run evaluate:recall',
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
  if (
    selected.chunkPolicy.id !== '512-64' ||
    selected.chunkPolicy.maxTokens !== 512 ||
    selected.chunkPolicy.overlapTokens !== 64 ||
    selected.candidateCount !== 8 ||
    selected.finalCount !== 5 ||
    evaluationIdentity.candidateLimits.dense !== 8 ||
    evaluationIdentity.candidateLimits.lexical !== 8 ||
    evaluationIdentity.candidateLimits.identifier !== 8 ||
    evaluationIdentity.finalResultCount !== 5
  ) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        'Recall quality selected policy differs from the approved 512/64, 8 candidates/channel, 5-result policy; obtain human approval before changing production',
      ],
    };
  }
  const wasMeasuredPassing = selection.combinations?.some(
    (combination) =>
      combination.gatePassed &&
      combination.chunkPolicy.id === selected.chunkPolicy.id &&
      combination.chunkPolicy.maxTokens === selected.chunkPolicy.maxTokens &&
      combination.chunkPolicy.overlapTokens === selected.chunkPolicy.overlapTokens &&
      combination.candidateCount === selected.candidateCount &&
      combination.finalCount === selected.finalCount,
  );
  if (!wasMeasuredPassing) {
    throw new Error(
      `Recall quality gate evidence inconsistent at ${resultsPath}: selected policy was not a passing measured combination`,
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
