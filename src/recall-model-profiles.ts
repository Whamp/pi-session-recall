import type { RecallEmbeddingModelIdentity } from './recall-index-manifest.js';

/** Embedding semantics shared by all backends serving one compatible model profile. */
export interface RecallEmbeddingModelProfile {
  identity: Readonly<RecallEmbeddingModelIdentity>;
  queryInputPrefix: string;
  documentInputPrefix: string;
  canary?: Readonly<RecallEmbeddingCanaryPolicy>;
}

/** Reranking semantics shared by all backends serving one compatible model profile. */
export interface RecallRerankingModelProfile {
  profileId: string;
  model: string;
  scoreMeaning: 'higher-is-more-relevant';
  scoreRange: Readonly<{ minimum: number; maximum: number }>;
  scorePolicy: string;
}

/** Bounds for model-generated lexical, semantic, and hypothetical-answer planned queries. */
export interface RecallQueryPlanBounds {
  minimumLexQueries: 1;
  maximumLexQueries: 3;
  minimumVecQueries: 1;
  maximumVecQueries: 3;
  maximumHydeQueries: 1;
}

/** Sampling and context policy shared by every backend executing one query planner profile. */
export interface RecallQueryPlanningGenerationPolicy {
  contextSize: number;
  maximumOutputTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  repeatPenaltyLastTokens: number;
  presencePenalty: number;
}

/** Fixed setup probe that protects original query terms while exercising recall intent. */
export interface RecallQueryPlanningConformanceCanary {
  query: string;
  recallIntent: string;
  protectedTerms: readonly string[];
}

/** Query planning semantics shared by all backends serving one compatible model profile. */
export interface RecallQueryPlanningModelProfile {
  profileId: string;
  model: string;
  promptPolicy: string;
  grammarVersion: string;
  grammar: string;
  planBounds: Readonly<RecallQueryPlanBounds>;
  generationPolicy: Readonly<RecallQueryPlanningGenerationPolicy>;
  conformanceCanary: Readonly<RecallQueryPlanningConformanceCanary>;
}

/** Recommended QMD query planner semantics and immutable downloadable artifact identity. */
export interface RecommendedQmdQueryPlanningModelProfile extends RecallQueryPlanningModelProfile {
  profileId: 'qmd-query-expansion-1.7b-q4-k-m-v1';
  model: 'qmd-query-expansion-1.7B-q4_k_m';
  promptPolicy: 'qmd-query-expansion-no-think-v1';
  grammarVersion: 'qmd-typed-query-plan-v1';
  purpose: string;
  source: Readonly<RecallModelArtifactSource>;
  license: Readonly<RecallModelLicenseIdentity>;
}

/** Immutable Octen embedding semantics, excluding backend URL and adapter execution details. */
export interface OctenEmbeddingModelProfile extends RecallEmbeddingModelProfile {
  queryInputPrefix: '';
  documentInputPrefix: '';
}

/** Immutable Qwen reranking semantics, excluding backend URL and adapter execution details. */
export type QwenRerankingModelProfile = RecallRerankingModelProfile;

/** Recommended Qwen reranker semantics and immutable downloadable artifact identity. */
export interface RecommendedQwenRerankingModelProfile extends RecallRerankingModelProfile {
  profileId: 'qwen3-reranker-0.6b-q8-0-v1';
  model: 'qwen3-reranker-0.6b-q8_0';
  purpose: string;
  scorePolicy: 'llama-cpp-qwen3-rank-probability-v1';
  source: Readonly<RecallModelArtifactSource>;
  license: Readonly<RecallModelLicenseIdentity>;
}

/** Immutable artifact source pinned to one repository revision and SHA-256 identity. */
export interface RecallModelArtifactSource {
  repository: string;
  revision: string;
  artifact: string;
  byteSize: number;
  sha256: string;
  downloadUrl: string;
}

/** Distribution terms and review status for one recommended model artifact. */
export interface RecallModelLicenseIdentity {
  id: string;
  name: string;
  url: string;
  distributionStatus: 'review-required' | 'approved';
}

/** Exact tokenizer identity carried inside the pinned GGUF artifact. */
export interface RecallGgufTokenizerIdentity {
  kind: 'gguf-metadata';
  model: string;
  artifactSha256: string;
  identity: string;
}

/** Repeatability policy for accepting one loaded embedding model without a frozen live vector. */
export interface RecallEmbeddingCanaryPolicy {
  policy: 'repeat-cosine-v1';
  operation: 'query';
  query: string;
  expectedDimensions: number;
  expectedNormalization: 'l2';
  minimumRepeatCosineSimilarity: number;
}

/** Recommended EmbeddingGemma semantics and immutable downloadable artifact identity. */
export interface RecommendedEmbeddingGemmaModelProfile extends RecallEmbeddingModelProfile {
  profileId: 'embeddinggemma-300m-q8-0-v1';
  purpose: string;
  source: Readonly<RecallModelArtifactSource>;
  license: Readonly<RecallModelLicenseIdentity>;
  nativeDimensions: 768;
  queryInputPrefix: 'task: search result | query: ';
  documentInputPrefix: 'title: none | text: ';
  normalization: 'l2';
  tokenizer: Readonly<RecallGgufTokenizerIdentity>;
  canary: Readonly<RecallEmbeddingCanaryPolicy>;
}

/** Creates the legacy-compatible Octen profile whose query and document inputs remain unchanged. */
export function createOctenEmbeddingModelProfile(
  identity: RecallEmbeddingModelIdentity,
): OctenEmbeddingModelProfile {
  return Object.freeze({
    identity: Object.freeze({ ...identity }),
    queryInputPrefix: '',
    documentInputPrefix: '',
  });
}

/** Creates the Qwen profile whose llama.cpp probability scores increase with relevance. */
export function createQwenRerankingModelProfile(model: string): QwenRerankingModelProfile {
  const normalizedModel = model.trim();
  if (!normalizedModel) {
    throw new Error('Recall Qwen reranking model profile invalid: expected a non-blank model name');
  }
  return Object.freeze({
    profileId: `qwen-reranking:${normalizedModel}`,
    model: normalizedModel,
    scoreMeaning: 'higher-is-more-relevant',
    scoreRange: Object.freeze({ minimum: 0, maximum: 1 }),
    scorePolicy: 'llama-cpp-qwen3-rank-probability-v1',
  });
}

/** Creates the recommended Qwen3 reranker profile pinned by immutable revision and checksum. */
export function createRecommendedQwenRerankingModelProfile(): RecommendedQwenRerankingModelProfile {
  return Object.freeze({
    profileId: 'qwen3-reranker-0.6b-q8-0-v1',
    model: 'qwen3-reranker-0.6b-q8_0',
    purpose: 'Score recall evidence against a submitted query for deep reranking.',
    scoreMeaning: 'higher-is-more-relevant',
    scoreRange: Object.freeze({ minimum: 0, maximum: 1 }),
    scorePolicy: 'llama-cpp-qwen3-rank-probability-v1',
    source: Object.freeze({
      repository: 'ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF',
      revision: 'a02f48bb4f057028298c21fa033da2b30d7742d5',
      artifact: 'qwen3-reranker-0.6b-q8_0.gguf',
      byteSize: 639_153_184,
      sha256: '22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48',
      downloadUrl:
        'https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/a02f48bb4f057028298c21fa033da2b30d7742d5/qwen3-reranker-0.6b-q8_0.gguf',
    }),
    license: Object.freeze({
      id: 'apache-2.0',
      name: 'Apache License 2.0',
      url: 'https://www.apache.org/licenses/LICENSE-2.0',
      distributionStatus: 'review-required',
    }),
  });
}

/** Creates the recommended QMD query planner profile pinned by immutable revision and checksum. */
export function createRecommendedQmdQueryPlanningModelProfile(): RecommendedQmdQueryPlanningModelProfile {
  return Object.freeze({
    profileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
    model: 'qmd-query-expansion-1.7B-q4_k_m',
    purpose: 'Plan bounded lexical, semantic, and hypothetical-answer recall queries.',
    promptPolicy: 'qmd-query-expansion-no-think-v1',
    grammarVersion: 'qmd-typed-query-plan-v1',
    grammar: [
      'root ::= line+',
      'line ::= type ": " content "\\n"',
      'type ::= "lex" | "vec" | "hyde"',
      'content ::= [^\\n]+',
    ].join('\n'),
    planBounds: Object.freeze({
      minimumLexQueries: 1,
      maximumLexQueries: 3,
      minimumVecQueries: 1,
      maximumVecQueries: 3,
      maximumHydeQueries: 1,
    }),
    generationPolicy: Object.freeze({
      contextSize: 2_048,
      maximumOutputTokens: 600,
      temperature: 0.7,
      topK: 20,
      topP: 0.8,
      repeatPenaltyLastTokens: 64,
      presencePenalty: 0.5,
    }),
    conformanceCanary: Object.freeze({
      query: 'source provenance',
      recallIntent: 'Find Pi conversation evidence about retained source provenance.',
      protectedTerms: Object.freeze(['source', 'provenance']),
    }),
    source: Object.freeze({
      repository: 'tobil/qmd-query-expansion-1.7B-gguf',
      revision: '7816de0b72572c6c860ca1eddf97ba9e7fb8cc65',
      artifact: 'qmd-query-expansion-1.7B-q4_k_m.gguf',
      byteSize: 1_282_438_912,
      sha256: '000dfb1c06efa6a049e9f64ba921c3740e2454f62abab6fa10e77bd30bb2bcc0',
      downloadUrl:
        'https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/resolve/7816de0b72572c6c860ca1eddf97ba9e7fb8cc65/qmd-query-expansion-1.7B-q4_k_m.gguf',
    }),
    license: Object.freeze({
      id: 'mit',
      name: 'MIT License',
      url: 'https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/blob/7816de0b72572c6c860ca1eddf97ba9e7fb8cc65/README.md',
      distributionStatus: 'review-required',
    }),
  });
}

/** Creates the recommended native-dimension EmbeddingGemma profile pinned by immutable revision. */
export function createRecommendedEmbeddingGemmaModelProfile(): RecommendedEmbeddingGemmaModelProfile {
  const artifactSha256 = 'b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63';
  return Object.freeze({
    profileId: 'embeddinggemma-300m-q8-0-v1',
    purpose: 'Embed recall queries and conversation documents for local semantic retrieval.',
    identity: Object.freeze({
      requestModel: 'embeddinggemma-300M-Q8_0',
      servedModelId: 'google/embeddinggemma-300M',
      artifact: 'embeddinggemma-300M-Q8_0.gguf',
      artifactRepository: 'ggml-org/embeddinggemma-300M-GGUF',
      artifactRevision: '0f741b5a6585bd53aeb15cd1372c56f2a0f65e12',
      artifactSha256,
      dimensions: 768,
      quantization: 'Q8_0',
      pooling: 'mean',
      normalization: 'l2',
    }),
    source: Object.freeze({
      repository: 'ggml-org/embeddinggemma-300M-GGUF',
      revision: '0f741b5a6585bd53aeb15cd1372c56f2a0f65e12',
      artifact: 'embeddinggemma-300M-Q8_0.gguf',
      byteSize: 333_590_944,
      sha256: artifactSha256,
      downloadUrl:
        'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf',
    }),
    license: Object.freeze({
      id: 'gemma',
      name: 'Gemma Terms of Use',
      url: 'https://ai.google.dev/gemma/terms',
      distributionStatus: 'review-required',
    }),
    nativeDimensions: 768,
    queryInputPrefix: 'task: search result | query: ',
    documentInputPrefix: 'title: none | text: ',
    normalization: 'l2',
    tokenizer: Object.freeze({
      kind: 'gguf-metadata',
      model: 'google/embeddinggemma-300M',
      artifactSha256,
      identity: `embeddinggemma-300M-Q8_0.gguf@${artifactSha256}`,
    }),
    canary: Object.freeze({
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'Which session evidence explains the retained implementation decision?',
      expectedDimensions: 768,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    }),
  });
}
