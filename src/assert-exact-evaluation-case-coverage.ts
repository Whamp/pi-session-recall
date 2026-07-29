/** Rejects empty, duplicate, missing, or extra case IDs across publishable evaluation collections. */
export function assertExactEvaluationCaseCoverage(
  collections: Readonly<Record<string, readonly string[]>> & {
    readonly controls: readonly string[];
  },
): void {
  const expectedCaseIds = new Set(collections.controls);
  for (const [collectionName, caseIds] of Object.entries(collections)) {
    if (caseIds.length === 0) {
      throw new Error(
        `Evaluation case coverage invalid: ${collectionName} must contain at least one case ID`,
      );
    }
    if (new Set(caseIds).size !== caseIds.length) {
      throw new Error(
        `Evaluation case coverage invalid: ${collectionName} case IDs must be unique`,
      );
    }
    if (
      caseIds.length !== expectedCaseIds.size ||
      caseIds.some((caseId) => !expectedCaseIds.has(caseId))
    ) {
      throw new Error(
        `Evaluation case coverage invalid: ${collectionName} must exactly match control case IDs`,
      );
    }
  }
}
