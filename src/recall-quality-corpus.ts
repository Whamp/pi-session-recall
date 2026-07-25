import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallChunkPolicy } from './recall-index-manifest.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

/** One immutable session file in the bounded recall quality corpus. */
export interface RecallQualitySessionFile {
  fileName: string;
  sha256: string;
}

/** One exact source location declared independently from retrieval results. */
export interface RecallQualityExpectedSource {
  sessionFile: string;
  entryId: string;
  requiredText: string[];
  expectedEvidenceKind?: SessionConversationChunk['evidenceKind'];
  expectedSummaryKind?: Exclude<SessionConversationChunk['summaryKind'], null>;
  expectedBranch?: 'active' | 'abandoned';
}

/** One fixed query and its independently declared source and context requirements. */
export interface RecallQualityEvaluationCase {
  id: string;
  category:
    | 'semantic_paraphrase'
    | 'exact_identifier'
    | 'tool_evidence'
    | 'context_dependent_reply'
    | 'branch'
    | 'summary'
    | 'duplicate_content';
  query: string;
  expectedSources: RecallQualityExpectedSource[];
  requiredContext: string[];
  minimumPreservedSourceOccurrences: number;
}

/** One required token ceiling and overlap pair in the bounded comparison. */
export interface RecallQualityChunkPolicy extends RecallChunkPolicy {
  id: string;
}

/** Fixed quality and interactive-latency thresholds applied without post-run changes. */
export interface RecallQualityGate {
  minimumPreRerankRecall: number;
  minimumPostRerankRecall: number;
  minimumContextUsefulness: number;
  minimumSourceOccurrencePreservation: number;
  maximumPostRerankDuplicateRate: number;
  maximumQueryP95Milliseconds: number;
  maximumRerankerP95Milliseconds: number;
}

/** Hard limits that prevent the evaluation command from becoming a corpus backfill. */
export interface RecallQualityWorkBounds {
  maximumSessionFiles: number;
  maximumEvaluationCases: number;
  maximumChunkPolicies: number;
  maximumCandidateCounts: number;
  maximumFinalCounts: number;
  maximumSearchRequests: number;
}

/** Complete independent corpus, count grid, work bounds, and quality gate specification. */
export interface RecallQualityCorpusSpecification {
  version: 1;
  corpus: {
    id: string;
    sessionDirectory: 'corpus';
    sessionFiles: RecallQualitySessionFile[];
  };
  bounds: RecallQualityWorkBounds;
  chunkPolicies: RecallQualityChunkPolicy[];
  candidateCounts: number[];
  finalCounts: number[];
  warmupQueriesPerCombination: number;
  qualityGate: RecallQualityGate;
  cases: RecallQualityEvaluationCase[];
}

/** Resolved and checksum-verified corpus ready for bounded indexing. */
export interface LoadedRecallQualityCorpus {
  specification: RecallQualityCorpusSpecification;
  specificationPath: string;
  specificationSha256: string;
  sessionDirectory: string;
  sessionFiles: Array<RecallQualitySessionFile & { path: string }>;
}

const sha256Schema = Type.String({ pattern: '^[a-f0-9]{64}$' });
const nonemptyStringSchema = Type.String({ minLength: 1 });
const probabilitySchema = Type.Number({ minimum: 0, maximum: 1 });
const positiveIntegerSchema = Type.Integer({ minimum: 1 });

const expectedSourceSchema = Type.Object(
  {
    sessionFile: Type.String({ pattern: '^[a-z0-9][a-z0-9-]*\\.jsonl$' }),
    entryId: nonemptyStringSchema,
    requiredText: Type.Array(nonemptyStringSchema, { minItems: 1 }),
    expectedEvidenceKind: Type.Optional(
      Type.Union([
        Type.Literal('conversation'),
        Type.Literal('turn_context'),
        Type.Literal('tool_call'),
        Type.Literal('tool_result'),
        Type.Literal('bash_execution'),
        Type.Literal('compaction_summary'),
        Type.Literal('branch_summary'),
      ]),
    ),
    expectedSummaryKind: Type.Optional(
      Type.Union([Type.Literal('compaction'), Type.Literal('branch')]),
    ),
    expectedBranch: Type.Optional(Type.Union([Type.Literal('active'), Type.Literal('abandoned')])),
  },
  { additionalProperties: false },
);

const evaluationCaseSchema = Type.Object(
  {
    id: Type.String({ pattern: '^[a-z0-9][a-z0-9-]*$' }),
    category: Type.Union([
      Type.Literal('semantic_paraphrase'),
      Type.Literal('exact_identifier'),
      Type.Literal('tool_evidence'),
      Type.Literal('context_dependent_reply'),
      Type.Literal('branch'),
      Type.Literal('summary'),
      Type.Literal('duplicate_content'),
    ]),
    query: nonemptyStringSchema,
    expectedSources: Type.Array(expectedSourceSchema, { minItems: 1 }),
    requiredContext: Type.Array(nonemptyStringSchema, { minItems: 1 }),
    minimumPreservedSourceOccurrences: positiveIntegerSchema,
  },
  { additionalProperties: false },
);

function createChunkPolicySchema(id: string, maxTokens: number, overlapTokens: number) {
  return Type.Object(
    {
      id: Type.Literal(id),
      maxTokens: Type.Literal(maxTokens),
      overlapTokens: Type.Literal(overlapTokens),
    },
    { additionalProperties: false },
  );
}

const recallQualityCorpusSchema = Type.Object(
  {
    version: Type.Literal(1),
    corpus: Type.Object(
      {
        id: Type.String({ pattern: '^[a-z0-9][a-z0-9-]*$' }),
        sessionDirectory: Type.Literal('corpus'),
        sessionFiles: Type.Array(
          Type.Object(
            {
              fileName: Type.String({ pattern: '^[a-z0-9][a-z0-9-]*\\.jsonl$' }),
              sha256: sha256Schema,
            },
            { additionalProperties: false },
          ),
          { minItems: 1 },
        ),
      },
      { additionalProperties: false },
    ),
    bounds: Type.Object(
      {
        maximumSessionFiles: positiveIntegerSchema,
        maximumEvaluationCases: positiveIntegerSchema,
        maximumChunkPolicies: positiveIntegerSchema,
        maximumCandidateCounts: positiveIntegerSchema,
        maximumFinalCounts: positiveIntegerSchema,
        maximumSearchRequests: positiveIntegerSchema,
      },
      { additionalProperties: false },
    ),
    chunkPolicies: Type.Tuple([
      createChunkPolicySchema('512-64', 512, 64),
      createChunkPolicySchema('768-96', 768, 96),
      createChunkPolicySchema('1024-128', 1_024, 128),
    ]),
    candidateCounts: Type.Array(Type.Integer({ minimum: 1, maximum: 200 }), { minItems: 1 }),
    finalCounts: Type.Array(Type.Integer({ minimum: 1, maximum: 200 }), { minItems: 1 }),
    warmupQueriesPerCombination: Type.Integer({ minimum: 0, maximum: 3 }),
    qualityGate: Type.Object(
      {
        minimumPreRerankRecall: probabilitySchema,
        minimumPostRerankRecall: probabilitySchema,
        minimumContextUsefulness: probabilitySchema,
        minimumSourceOccurrencePreservation: probabilitySchema,
        maximumPostRerankDuplicateRate: probabilitySchema,
        maximumQueryP95Milliseconds: Type.Number({ exclusiveMinimum: 0 }),
        maximumRerankerP95Milliseconds: Type.Number({ exclusiveMinimum: 0 }),
      },
      { additionalProperties: false },
    ),
    cases: Type.Array(evaluationCaseSchema, { minItems: 1 }),
  },
  { additionalProperties: false },
);

function createSha256(content: string | Buffer): string {
  return createHash('sha256').update(content).digest('hex');
}

function assertUniqueValues<Item>(values: readonly Item[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`Recall quality corpus invalid: ${label} must contain unique values`);
  }
}

function assertAscendingIntegers(values: readonly number[], label: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined || current <= previous) {
      throw new Error(`Recall quality corpus invalid: ${label} must be strictly ascending`);
    }
  }
}

function assertRecallQualityBounds(specification: RecallQualityCorpusSpecification): void {
  const { bounds } = specification;
  const dimensions: Array<[number, number, string]> = [
    [specification.corpus.sessionFiles.length, bounds.maximumSessionFiles, 'session files'],
    [specification.cases.length, bounds.maximumEvaluationCases, 'evaluation cases'],
    [specification.chunkPolicies.length, bounds.maximumChunkPolicies, 'chunk policies'],
    [specification.candidateCounts.length, bounds.maximumCandidateCounts, 'candidate counts'],
    [specification.finalCounts.length, bounds.maximumFinalCounts, 'final counts'],
  ];
  for (const [actual, maximum, label] of dimensions) {
    if (actual > maximum) {
      throw new Error(
        `Recall quality corpus bound exceeded: ${label} ${actual} exceeds maximum ${maximum}`,
      );
    }
  }
  const searchRequests =
    specification.chunkPolicies.length *
    specification.candidateCounts.length *
    (specification.cases.length + specification.warmupQueriesPerCombination);
  if (searchRequests > bounds.maximumSearchRequests) {
    throw new Error(
      `Recall quality corpus bound exceeded: search requests ${searchRequests} exceeds maximum ${bounds.maximumSearchRequests}`,
    );
  }
  assertUniqueValues(
    specification.corpus.sessionFiles.map(({ fileName }) => fileName),
    'session file names',
  );
  assertUniqueValues(
    specification.chunkPolicies.map(({ id }) => id),
    'chunk policy ids',
  );
  assertUniqueValues(specification.candidateCounts, 'candidate counts');
  assertUniqueValues(specification.finalCounts, 'final counts');
  assertUniqueValues(
    specification.cases.map(({ id }) => id),
    'case ids',
  );
  assertAscendingIntegers(specification.candidateCounts, 'candidate counts');
  assertAscendingIntegers(specification.finalCounts, 'final counts');
}

interface RecallQualitySourceEntry {
  parentId: string | null;
  sourceText: string;
  evidenceKinds: ReadonlySet<SessionConversationChunk['evidenceKind']>;
  summaryKind: SessionConversationChunk['summaryKind'];
}

interface RecallQualitySourceEvidence {
  entries: ReadonlyMap<string, RecallQualitySourceEntry>;
  activeEntryIds: ReadonlySet<string>;
}

function isRecallQualitySourceRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function readSourceEvidenceKinds(
  record: Record<string, unknown>,
  recordType: unknown,
): {
  evidenceKinds: ReadonlySet<SessionConversationChunk['evidenceKind']>;
  summaryKind: SessionConversationChunk['summaryKind'];
} {
  const evidenceKinds = new Set<SessionConversationChunk['evidenceKind']>();
  let summaryKind: SessionConversationChunk['summaryKind'] = null;
  if (recordType === 'compaction') {
    evidenceKinds.add('compaction_summary');
    summaryKind = 'compaction';
  } else if (recordType === 'branch_summary') {
    evidenceKinds.add('branch_summary');
    summaryKind = 'branch';
  } else if (recordType === 'custom_message') {
    evidenceKinds.add('conversation');
  } else if (recordType === 'message') {
    const message = Reflect.get(record, 'message');
    if (isRecallQualitySourceRecord(message)) {
      const role = Reflect.get(message, 'role');
      const messageContent = Reflect.get(message, 'content');
      if (role === 'toolResult') {
        evidenceKinds.add('tool_result');
      } else if (role === 'bashExecution') {
        evidenceKinds.add('bash_execution');
      } else if (role === 'user') {
        evidenceKinds.add('conversation');
      } else if (role === 'assistant') {
        if (typeof messageContent === 'string' && messageContent.trim()) {
          evidenceKinds.add('conversation');
        }
        if (Array.isArray(messageContent)) {
          for (const block of messageContent) {
            if (!isRecallQualitySourceRecord(block)) {
              continue;
            }
            if (Reflect.get(block, 'type') === 'text') {
              const text = Reflect.get(block, 'text');
              if (typeof text === 'string' && text.trim()) {
                evidenceKinds.add('conversation');
              }
            }
            if (Reflect.get(block, 'type') === 'toolCall') {
              evidenceKinds.add('tool_call');
            }
          }
        }
      }
    }
  }
  return { evidenceKinds, summaryKind };
}

function readSourceEvidence(content: string, fileName: string): RecallQualitySourceEvidence {
  const entries = new Map<string, RecallQualitySourceEntry>();
  const orderedEntryIds: string[] = [];
  let leafTargetId: string | null | undefined;
  for (const [index, line] of content.split('\n').entries()) {
    if (!line.trim()) {
      continue;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Recall quality corpus source invalid at ${fileName}:${index + 1}: malformed JSON: ${message}`,
        { cause: error },
      );
    }
    if (!isRecallQualitySourceRecord(parsed)) {
      throw new Error(
        `Recall quality corpus source invalid at ${fileName}:${index + 1}: expected an object`,
      );
    }
    const recordType = Reflect.get(parsed, 'type');
    if (recordType === 'session') {
      continue;
    }
    if (recordType === 'leaf') {
      const targetId = Reflect.get(parsed, 'targetId');
      if (targetId !== null && typeof targetId !== 'string') {
        throw new Error(
          `Recall quality corpus source invalid at ${fileName}:${index + 1}: leaf target must be a string or null`,
        );
      }
      leafTargetId = targetId;
      continue;
    }
    const id = Reflect.get(parsed, 'id');
    const parentId = Reflect.get(parsed, 'parentId');
    if (typeof id !== 'string' || (parentId !== null && typeof parentId !== 'string')) {
      throw new Error(
        `Recall quality corpus source invalid at ${fileName}:${index + 1}: entry id and parentId are required`,
      );
    }
    if (entries.has(id)) {
      throw new Error(`Recall quality corpus source invalid: duplicate entry ${fileName}#${id}`);
    }
    entries.set(id, {
      parentId,
      sourceText: JSON.stringify(parsed),
      ...readSourceEvidenceKinds(parsed, recordType),
    });
    orderedEntryIds.push(id);
  }

  const activeEntryIds = new Set<string>();
  let currentId = leafTargetId === undefined ? (orderedEntryIds.at(-1) ?? null) : leafTargetId;
  while (currentId !== null) {
    if (activeEntryIds.has(currentId)) {
      throw new Error(
        `Recall quality corpus source invalid: parent cycle includes ${fileName}#${currentId}`,
      );
    }
    const entry = entries.get(currentId);
    if (!entry) {
      throw new Error(
        `Recall quality corpus source invalid: active path entry missing ${fileName}#${currentId}`,
      );
    }
    activeEntryIds.add(currentId);
    currentId = entry.parentId;
  }
  return { entries, activeEntryIds };
}

function assertExpectedSourcesExist(
  specification: RecallQualityCorpusSpecification,
  sourceEvidenceByFile: ReadonlyMap<string, RecallQualitySourceEvidence>,
): void {
  const declaredFiles = new Set(specification.corpus.sessionFiles.map(({ fileName }) => fileName));
  for (const evaluationCase of specification.cases) {
    for (const expectedSource of evaluationCase.expectedSources) {
      if (!declaredFiles.has(expectedSource.sessionFile)) {
        throw new Error(
          `Recall quality case ${evaluationCase.id} references undeclared session file ${expectedSource.sessionFile}`,
        );
      }
      const sourceEvidence = sourceEvidenceByFile.get(expectedSource.sessionFile);
      const sourceEntry = sourceEvidence?.entries.get(expectedSource.entryId);
      if (!sourceEntry) {
        throw new Error(
          `Recall quality case ${evaluationCase.id} source missing: ${expectedSource.sessionFile}#${expectedSource.entryId}`,
        );
      }
      for (const requiredText of expectedSource.requiredText) {
        if (!sourceEntry.sourceText.includes(requiredText)) {
          throw new Error(
            `Recall quality case ${evaluationCase.id} source text missing at ${expectedSource.sessionFile}#${expectedSource.entryId}: ${requiredText}`,
          );
        }
      }
      if (
        expectedSource.expectedEvidenceKind &&
        !sourceEntry.evidenceKinds.has(expectedSource.expectedEvidenceKind)
      ) {
        throw new Error(
          `Recall quality case ${evaluationCase.id} evidence kind mismatch at ${expectedSource.sessionFile}#${expectedSource.entryId}: source cannot produce ${expectedSource.expectedEvidenceKind}`,
        );
      }
      if (
        expectedSource.expectedSummaryKind &&
        sourceEntry.summaryKind !== expectedSource.expectedSummaryKind
      ) {
        throw new Error(
          `Recall quality case ${evaluationCase.id} summary kind mismatch at ${expectedSource.sessionFile}#${expectedSource.entryId}: expected ${expectedSource.expectedSummaryKind}, received ${sourceEntry.summaryKind ?? 'none'}`,
        );
      }
      if (expectedSource.expectedBranch) {
        const actualBranch = sourceEvidence?.activeEntryIds.has(expectedSource.entryId)
          ? 'active'
          : 'abandoned';
        if (actualBranch !== expectedSource.expectedBranch) {
          throw new Error(
            `Recall quality case ${evaluationCase.id} branch expectation mismatch at ${expectedSource.sessionFile}#${expectedSource.entryId}: expected ${expectedSource.expectedBranch}, received ${actualBranch}`,
          );
        }
      }
    }
  }
}

/** Loads one checksum-fixed corpus and rejects any source, policy, or work-bound drift. */
export async function loadRecallQualityCorpus(
  specificationPath: string,
): Promise<LoadedRecallQualityCorpus> {
  const resolvedSpecificationPath = resolve(specificationPath);
  const specificationContent = await readFile(resolvedSpecificationPath, 'utf8');
  let specification: RecallQualityCorpusSpecification;
  try {
    const parsed: unknown = JSON.parse(specificationContent);
    specification = Value.Parse(recallQualityCorpusSchema, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall quality corpus invalid at ${resolvedSpecificationPath}: ${message}`, {
      cause: error,
    });
  }
  assertRecallQualityBounds(specification);

  const sessionDirectory = join(
    dirname(resolvedSpecificationPath),
    specification.corpus.sessionDirectory,
  );
  const sourceEvidenceByFile = new Map<string, RecallQualitySourceEvidence>();
  const sessionFiles: LoadedRecallQualityCorpus['sessionFiles'] = [];
  for (const sessionFile of specification.corpus.sessionFiles) {
    const sessionPath = join(sessionDirectory, sessionFile.fileName);
    const content = await readFile(sessionPath, 'utf8');
    const actualSha256 = createSha256(content);
    if (actualSha256 !== sessionFile.sha256) {
      throw new Error(
        `Recall quality corpus checksum mismatch for ${sessionFile.fileName}: expected ${sessionFile.sha256}, received ${actualSha256}`,
      );
    }
    sourceEvidenceByFile.set(
      sessionFile.fileName,
      readSourceEvidence(content, sessionFile.fileName),
    );
    sessionFiles.push({ ...sessionFile, path: sessionPath });
  }
  assertExpectedSourcesExist(specification, sourceEvidenceByFile);

  return {
    specification,
    specificationPath: resolvedSpecificationPath,
    specificationSha256: createSha256(specificationContent),
    sessionDirectory,
    sessionFiles,
  };
}
