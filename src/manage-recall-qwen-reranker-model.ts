import { pathToFileURL } from 'node:url';

import { runRecallModelArtifactCommand } from './manage-recall-model-artifact.js';
import type { RecallModelArtifactTransport } from './recall-model-artifact-cache.js';
import {
  createRecommendedQwenRerankingModelProfile,
  type RecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

const RECALL_QWEN_RERANKER_MODEL_COMMAND_USAGE =
  'usage: npm run model:qwen-reranker -- <inspect|status|verify|doctor|download|repair|remove> [--approve]';

/** Injectable paths, profile, transport, and output for the Qwen reranker model command. */
export interface RecallQwenRerankerModelCommandOptions {
  homeDirectory?: string;
  cacheDirectory?: string;
  profile?: RecommendedQwenRerankingModelProfile;
  transport?: RecallModelArtifactTransport;
  writeOutput?: (value: string) => void;
}

/** Runs one Qwen reranker artifact inspection or explicitly approved mutation and emits JSON. */
export async function runRecallQwenRerankerModelCommand(
  argumentsList: readonly string[],
  options: RecallQwenRerankerModelCommandOptions = {},
): Promise<void> {
  await runRecallModelArtifactCommand(argumentsList, {
    commandUsage: RECALL_QWEN_RERANKER_MODEL_COMMAND_USAGE,
    errorPrefix: 'Recall Qwen reranker model',
    profile: options.profile ?? createRecommendedQwenRerankingModelProfile(),
    ...(options.homeDirectory ? { homeDirectory: options.homeDirectory } : {}),
    ...(options.cacheDirectory ? { cacheDirectory: options.cacheDirectory } : {}),
    ...(options.transport ? { transport: options.transport } : {}),
    ...(options.writeOutput ? { writeOutput: options.writeOutput } : {}),
  });
}

const invokedPath = process.argv[1];
if (invokedPath && import.meta.url === pathToFileURL(invokedPath).href) {
  runRecallQwenRerankerModelCommand(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
