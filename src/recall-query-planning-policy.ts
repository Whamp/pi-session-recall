import type { RecallPlannedRetrievalQuery } from './recall-inference-capabilities.js';
import type { RecallQueryPlanningModelProfile } from './recall-model-profiles.js';

const MAXIMUM_QMD_QUERY_CONTENT_CODE_POINTS = 512;

/** Formats the QMD no-think query expansion prompt, including optional recall intent. */
export function formatQmdQueryPlanningPrompt(query: string, recallIntent?: string): string {
  const normalizedQuery = query.trim();
  if (!normalizedQuery || /[\r\n]/u.test(normalizedQuery)) {
    throw new Error('Recall query planning query invalid: expected non-blank single-line text');
  }
  if (recallIntent === undefined) {
    return `/no_think Expand this search query: ${normalizedQuery}`;
  }
  const normalizedRecallIntent = recallIntent.trim();
  if (!normalizedRecallIntent || /[\r\n]/u.test(normalizedRecallIntent)) {
    throw new Error('Recall query planning intent invalid: expected non-blank single-line text');
  }
  return `/no_think Expand this search query: ${normalizedQuery}\nQuery intent: ${normalizedRecallIntent}`;
}

/** Validates ordered QMD plan bounds using at most 512 Unicode code points after trimming. */
export function validateQmdQueryPlanningPlan(
  plan: readonly RecallPlannedRetrievalQuery[],
  profile: RecallQueryPlanningModelProfile,
): RecallPlannedRetrievalQuery[] {
  const normalizedPlan: RecallPlannedRetrievalQuery[] = [];
  let seenLexicalQuery = false;
  let seenVectorQuery = false;
  let seenHypotheticalAnswerQuery = false;
  for (const [index, plannedQuery] of plan.entries()) {
    if (
      plannedQuery.type !== 'lex' &&
      plannedQuery.type !== 'vec' &&
      plannedQuery.type !== 'hyde'
    ) {
      throw new Error(
        `Recall query planning output grammar invalid at entry ${index + 1}: expected lex, vec, or hyde`,
      );
    }
    if (typeof plannedQuery.query !== 'string' || !plannedQuery.query.trim()) {
      throw new Error(
        `Recall query planning output grammar invalid at entry ${index + 1}: expected non-blank single-line text`,
      );
    }
    if (/[\r\n]/u.test(plannedQuery.query)) {
      throw new Error(
        `Recall query planning output grammar invalid at entry ${index + 1}: expected non-blank single-line text`,
      );
    }
    const normalizedQuery = plannedQuery.query.trim();
    if (Array.from(normalizedQuery).length > MAXIMUM_QMD_QUERY_CONTENT_CODE_POINTS) {
      throw new Error(
        `Recall query planning output grammar invalid at entry ${index + 1}: expected at most ${MAXIMUM_QMD_QUERY_CONTENT_CODE_POINTS} Unicode code points`,
      );
    }
    if (seenHypotheticalAnswerQuery) {
      throw new Error(
        `Recall query planning output ordering invalid at entry ${index + 1}: no query may follow a hypothetical-answer query`,
      );
    }
    if (plannedQuery.type === 'lex') {
      if (seenVectorQuery) {
        throw new Error(
          `Recall query planning output ordering invalid at entry ${index + 1}: lexical queries must precede vector queries`,
        );
      }
      seenLexicalQuery = true;
    } else if (plannedQuery.type === 'vec') {
      if (!seenLexicalQuery) {
        throw new Error(
          `Recall query planning output ordering invalid at entry ${index + 1}: vector queries must follow at least one lexical query`,
        );
      }
      seenVectorQuery = true;
    } else {
      if (!seenVectorQuery) {
        throw new Error(
          `Recall query planning output ordering invalid at entry ${index + 1}: a hypothetical-answer query must follow all lexical and vector queries`,
        );
      }
      seenHypotheticalAnswerQuery = true;
    }
    normalizedPlan.push({ type: plannedQuery.type, query: normalizedQuery });
  }
  const lexQueryCount = normalizedPlan.filter((query) => query.type === 'lex').length;
  const vecQueryCount = normalizedPlan.filter((query) => query.type === 'vec').length;
  const hydeQueryCount = normalizedPlan.filter((query) => query.type === 'hyde').length;
  const bounds = profile.planBounds;
  if (
    lexQueryCount < bounds.minimumLexQueries ||
    lexQueryCount > bounds.maximumLexQueries ||
    vecQueryCount < bounds.minimumVecQueries ||
    vecQueryCount > bounds.maximumVecQueries ||
    hydeQueryCount > bounds.maximumHydeQueries
  ) {
    throw new Error(
      `Recall query planning output bounds invalid: expected ${bounds.minimumLexQueries}-${bounds.maximumLexQueries} lex, ${bounds.minimumVecQueries}-${bounds.maximumVecQueries} vec, and 0-${bounds.maximumHydeQueries} hyde queries; received ${lexQueryCount} lex, ${vecQueryCount} vec, and ${hydeQueryCount} hyde`,
    );
  }
  const uniqueQueries = new Set(normalizedPlan.map(({ type, query }) => `${type}:${query}`));
  if (uniqueQueries.size !== normalizedPlan.length) {
    throw new Error(
      'Recall query planning output invalid: duplicate typed queries are not allowed',
    );
  }
  return normalizedPlan;
}

/** Parses grammar-constrained QMD lex, vec, and hyde planner text into a validated plan. */
export function parseQmdQueryPlanningOutput(
  output: string,
  profile: RecallQueryPlanningModelProfile,
): RecallPlannedRetrievalQuery[] {
  const lines = output
    .trim()
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);
  const plan = lines.map((line, index): RecallPlannedRetrievalQuery => {
    const match = /^(lex|vec|hyde): (.+)$/u.exec(line);
    if (!match) {
      throw new Error(
        `Recall query planning output grammar invalid at line ${index + 1}: expected "lex: ", "vec: ", or "hyde: " followed by single-line text`,
      );
    }
    const type = match[1];
    const query = match[2]
      ?.trim()
      .replace(/<\/think>$/u, '')
      .trim();
    if ((type !== 'lex' && type !== 'vec' && type !== 'hyde') || !query) {
      throw new Error(`Recall query planning output invalid at line ${index + 1}`);
    }
    return { type, query };
  });
  return validateQmdQueryPlanningPlan(plan, profile);
}
