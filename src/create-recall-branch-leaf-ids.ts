/** Minimal branch graph needed to map every entry to its descendant branch leaves. */
export interface RecallBranchEntryGraph {
  entryIds: readonly string[];
  parentEntryIds: readonly (string | null)[];
}

/** Maps each recall entry to descendant branch leaves in time proportional to the output. */
export function createRecallBranchLeafIdsByEntryId(
  graph: Readonly<RecallBranchEntryGraph>,
): Map<string, string[]> {
  const parentEntryIdByEntryId = new Map<string, string | null>();
  const parentEntryIds = new Set<string>();
  const branchLeafIdsByEntryId = new Map<string, string[]>();
  for (const [index, entryId] of graph.entryIds.entries()) {
    const parentEntryId = graph.parentEntryIds[index] ?? null;
    parentEntryIdByEntryId.set(entryId, parentEntryId);
    branchLeafIdsByEntryId.set(entryId, []);
    if (parentEntryId !== null) {
      parentEntryIds.add(parentEntryId);
    }
  }

  const leafEntryIds = graph.entryIds.filter((entryId) => !parentEntryIds.has(entryId));
  for (const leafEntryId of leafEntryIds) {
    let currentEntryId: string | null = leafEntryId;
    while (currentEntryId !== null) {
      const branchLeafIds = branchLeafIdsByEntryId.get(currentEntryId);
      if (branchLeafIds === undefined) {
        break;
      }
      branchLeafIds.push(leafEntryId);
      currentEntryId = parentEntryIdByEntryId.get(currentEntryId) ?? null;
    }
  }

  for (const [entryId, branchLeafIds] of branchLeafIdsByEntryId) {
    branchLeafIdsByEntryId.set(entryId, branchLeafIds.toSorted());
  }
  return branchLeafIdsByEntryId;
}
