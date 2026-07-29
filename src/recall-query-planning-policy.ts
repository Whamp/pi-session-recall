import type { RecallPlannedRetrievalQuery } from './recall-inference-capabilities.js';
import type { RecallQueryPlanningModelProfile } from './recall-model-profiles.js';

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

/** Parses and bounds grammar-constrained QMD lex, vec, and hyde planner output. */
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
    const query = match[2]?.trim();
    if ((type !== 'lex' && type !== 'vec' && type !== 'hyde') || !query) {
      throw new Error(`Recall query planning output invalid at line ${index + 1}`);
    }
    return { type, query };
  });
  const lexQueryCount = plan.filter((query) => query.type === 'lex').length;
  const vecQueryCount = plan.filter((query) => query.type === 'vec').length;
  const hydeQueryCount = plan.filter((query) => query.type === 'hyde').length;
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
  const uniqueQueries = new Set(plan.map(({ type, query }) => `${type}:${query}`));
  if (uniqueQueries.size !== plan.length) {
    throw new Error(
      'Recall query planning output invalid: duplicate typed queries are not allowed',
    );
  }
  return plan;
}
