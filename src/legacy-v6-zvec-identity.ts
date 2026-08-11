/** Frozen Zvec identity retained only for exact manifest-v6 rollback compatibility. */
export const LEGACY_V6_ZVEC_IDENTITY = Object.freeze({
  schemaVersion: 8,
  ftsConfigurationVersion: 2,
  vectorQuantization: 'fp32' as const,
  metric: 'inner-product' as const,
  hnswM: 50,
  hnswEfConstruction: 500,
  hnswEfSearch: 300,
});
