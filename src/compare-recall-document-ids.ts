/** Orders recall document IDs bytewise for deterministic ranking tie-breaks. */
export function compareRecallDocumentIds(leftId: string, rightId: string): number {
  if (leftId < rightId) {
    return -1;
  }
  if (leftId > rightId) {
    return 1;
  }
  return 0;
}
