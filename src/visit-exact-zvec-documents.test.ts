import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  type ZVecCollection,
} from '@zvec/zvec';
import fc from 'fast-check';

import { visitExactZvecDocuments } from './visit-exact-zvec-documents.js';

const uniqueMembershipKeys = fc.uniqueArray(fc.stringMatching(/^[a-f0-9]{1,16}$/u), {
  maxLength: 200,
});

void test('exact zvec enumeration escapes quoted partition pivots beyond the query limit', () => {
  const root = mkdtempSync(join(tmpdir(), 'recall-exact-zvec-quoted-pivot-'));
  let collection: ZVecCollection | undefined;
  try {
    const openedCollection = ZVecCreateAndOpen(
      join(root, 'collection'),
      new ZVecCollectionSchema({
        name: 'exact_membership_quoted_pivot',
        fields: [{ name: 'membershipKey', dataType: ZVecDataType.STRING }],
      }),
    );
    collection = openedCollection;
    const expectedIds = Array.from({ length: 100_001 }, (_, index) => `record_${index}`);
    for (let start = 0; start < expectedIds.length; start += 1_000) {
      const statuses = openedCollection.insertSync(
        expectedIds.slice(start, start + 1_000).map((id, offset) => ({
          id,
          fields: {
            membershipKey: `quoted'\\key-${String(start + offset).padStart(6, '0')}`,
          },
        })),
      );
      assert.equal(
        statuses.every(({ ok }) => ok),
        true,
      );
    }

    const visitedIds: string[] = [];
    const visitedCount = visitExactZvecDocuments(
      openedCollection,
      { uniquePartitionField: 'membershipKey', outputFields: [] },
      ({ id }) => visitedIds.push(id),
    );

    assert.equal(visitedCount, expectedIds.length);
    assert.deepEqual(visitedIds.toSorted(), expectedIds.toSorted());
  } finally {
    collection?.closeSync();
    rmSync(root, { recursive: true, force: true });
  }
});

void test('exact zvec enumeration preserves generated membership and scalar filters', () => {
  fc.assert(
    fc.property(uniqueMembershipKeys, (membershipKeys) => {
      const root = mkdtempSync(join(tmpdir(), 'recall-exact-zvec-property-'));
      let collection: ZVecCollection | undefined;
      try {
        const openedCollection = ZVecCreateAndOpen(
          join(root, 'collection'),
          new ZVecCollectionSchema({
            name: 'exact_membership_property',
            fields: [
              { name: 'membershipKey', dataType: ZVecDataType.STRING },
              { name: 'category', dataType: ZVecDataType.STRING },
            ],
          }),
        );
        collection = openedCollection;
        const statuses =
          membershipKeys.length === 0
            ? []
            : openedCollection.insertSync(
                membershipKeys.map((membershipKey, index) => ({
                  id: `record_${membershipKey}`,
                  fields: {
                    membershipKey,
                    category: index % 2 === 0 ? 'included' : 'excluded',
                  },
                })),
              );
        assert.equal(
          statuses.every(({ ok }) => ok),
          true,
        );

        const readIds = (filter?: string): string[] => {
          const ids: string[] = [];
          visitExactZvecDocuments(
            openedCollection,
            {
              ...(filter === undefined ? {} : { filter }),
              uniquePartitionField: 'membershipKey',
              outputFields: [],
            },
            ({ id }) => ids.push(id),
          );
          return ids.toSorted();
        };
        assert.deepEqual(
          readIds(),
          membershipKeys.map((membershipKey) => `record_${membershipKey}`).toSorted(),
        );
        assert.deepEqual(
          readIds("category = 'included'"),
          membershipKeys
            .filter((membershipKey, index) => {
              void membershipKey;
              return index % 2 === 0;
            })
            .map((membershipKey) => `record_${membershipKey}`)
            .toSorted(),
        );
      } finally {
        collection?.closeSync();
        rmSync(root, { recursive: true, force: true });
      }
    }),
    { numRuns: 20 },
  );
});
