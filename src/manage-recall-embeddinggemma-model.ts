import { pathToFileURL } from 'node:url';

import { runRecallModelArtifactCommand } from './manage-recall-model-artifact.js';
import type { RecallModelArtifactTransport } from './recall-model-artifact-cache.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  type RecommendedEmbeddingGemmaModelProfile,
} from './recall-model-profiles.js';

const RECALL_EMBEDDINGGEMMA_MODEL_COMMAND_USAGE =
  'usage: npm run model:embeddinggemma -- <inspect|status|verify|doctor|download|repair|remove> [--approve]';

/** Injectable paths, profile, transport, and output for the deterministic model command. */
export interface RecallEmbeddingGemmaModelCommandOptions {
  homeDirectory?: string;
  cacheDirectory?: string;
  profile?: RecommendedEmbeddingGemmaModelProfile;
  transport?: RecallModelArtifactTransport;
  writeOutput?: (value: string) => void;
}

/** Runs one EmbeddingGemma artifact inspection or explicitly approved mutation and emits JSON. */
export async function runRecallEmbeddingGemmaModelCommand(
  argumentsList: readonly string[],
  options: RecallEmbeddingGemmaModelCommandOptions = {},
): Promise<void> {
  await runRecallModelArtifactCommand(argumentsList, {
    commandUsage: RECALL_EMBEDDINGGEMMA_MODEL_COMMAND_USAGE,
    errorPrefix: 'Recall EmbeddingGemma model',
    profile: options.profile ?? createRecommendedEmbeddingGemmaModelProfile(),
    ...(options.homeDirectory ? { homeDirectory: options.homeDirectory } : {}),
    ...(options.cacheDirectory ? { cacheDirectory: options.cacheDirectory } : {}),
    ...(options.transport ? { transport: options.transport } : {}),
    ...(options.writeOutput ? { writeOutput: options.writeOutput } : {}),
  });
}

const invokedPath = process.argv[1];
if (invokedPath && import.meta.url === pathToFileURL(invokedPath).href) {
  runRecallEmbeddingGemmaModelCommand(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
