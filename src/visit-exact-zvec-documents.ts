import type { ZVecCollection, ZVecDoc } from '@zvec/zvec';

/** Maximum supported zvec `topk` used by exact scalar-only enumeration. */
export const ZVEC_EXACT_ENUMERATION_QUERY_LIMIT = 100_000;

/** One exact scalar enumeration partitioned by a unique string field. */
export interface ExactZvecDocumentEnumeration {
  filter?: string;
  uniquePartitionField: string;
  outputFields: readonly string[];
}

function escapeZvecStringLiteral(value: string): string {
  return value.replaceAll("'", "''");
}

function combineZvecFilters(baseFilter: string | undefined, partitionFilter: string): string {
  return baseFilter === undefined ? partitionFilter : `(${baseFilter}) AND (${partitionFilter})`;
}

function readUniquePartitionValues(
  documents: readonly ZVecDoc[],
  uniquePartitionField: string,
): string[] {
  const uniqueValues = new Set<string>();
  for (const document of documents) {
    const value: unknown = document.fields[uniquePartitionField];
    if (typeof value !== 'string') {
      throw new Error(
        `Exact zvec enumeration partition field ${uniquePartitionField} invalid for ${document.id}: expected string`,
      );
    }
    if (uniqueValues.has(value)) {
      throw new Error(
        `Exact zvec enumeration partition field ${uniquePartitionField} is not unique: ${value}`,
      );
    }
    uniqueValues.add(value);
  }
  return [...uniqueValues].toSorted((left, right) => left.localeCompare(right));
}

/** Visits every exact scalar-filter match once without an over-limit or approximate query. */
export function visitExactZvecDocuments(
  collection: ZVecCollection,
  enumeration: Readonly<ExactZvecDocumentEnumeration>,
  visit: (document: ZVecDoc) => void,
): number {
  const outputFields = enumeration.outputFields.includes(enumeration.uniquePartitionField)
    ? [...enumeration.outputFields]
    : [...enumeration.outputFields, enumeration.uniquePartitionField];
  const pendingFilters: Array<string | undefined> = [enumeration.filter];
  let visitedDocumentCount = 0;

  while (pendingFilters.length > 0) {
    const filter = pendingFilters.pop();
    const documents = collection.querySync({
      ...(filter === undefined ? {} : { filter }),
      topk: ZVEC_EXACT_ENUMERATION_QUERY_LIMIT,
      outputFields,
      includeVector: false,
    });
    const partitionValues = readUniquePartitionValues(documents, enumeration.uniquePartitionField);
    if (documents.length < ZVEC_EXACT_ENUMERATION_QUERY_LIMIT) {
      for (const document of documents) {
        visit(document);
        visitedDocumentCount += 1;
      }
      continue;
    }

    const pivot = partitionValues[Math.floor(partitionValues.length / 2)];
    if (pivot === undefined) {
      throw new Error('Exact zvec enumeration could not select a nonempty partition pivot');
    }
    const escapedPivot = escapeZvecStringLiteral(pivot);
    pendingFilters.push(
      combineZvecFilters(filter, `${enumeration.uniquePartitionField} >= '${escapedPivot}'`),
      combineZvecFilters(filter, `${enumeration.uniquePartitionField} < '${escapedPivot}'`),
    );
  }

  return visitedDocumentCount;
}
