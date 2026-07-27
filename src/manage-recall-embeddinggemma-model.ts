import { homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  createRecallModelArtifactCache,
  type RecallModelArtifactTransport,
} from './recall-model-artifact-cache.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  type RecommendedEmbeddingGemmaModelProfile,
} from './recall-model-profiles.js';

const RECALL_EMBEDDINGGEMMA_MODEL_COMMAND_USAGE =
  'usage: npm run model:embeddinggemma -- <inspect|status|verify|doctor|download|repair|remove> [--approve]';

type RecallEmbeddingGemmaModelAction =
  | 'inspect'
  | 'status'
  | 'verify'
  | 'doctor'
  | 'download'
  | 'repair'
  | 'remove';

function isRecallEmbeddingGemmaModelAction(
  value: string,
): value is RecallEmbeddingGemmaModelAction {
  return (
    value === 'inspect' ||
    value === 'status' ||
    value === 'verify' ||
    value === 'doctor' ||
    value === 'download' ||
    value === 'repair' ||
    value === 'remove'
  );
}

/** Injectable paths, profile, transport, and output for the deterministic model command. */
export interface RecallEmbeddingGemmaModelCommandOptions {
  homeDirectory?: string;
  cacheDirectory?: string;
  profile?: RecommendedEmbeddingGemmaModelProfile;
  transport?: RecallModelArtifactTransport;
  writeOutput?: (value: string) => void;
}

function parseRecallEmbeddingGemmaModelArguments(argumentsList: readonly string[]): {
  action: RecallEmbeddingGemmaModelAction;
  approved: boolean;
} {
  const [action, ...flags] = argumentsList;
  if (!action || !isRecallEmbeddingGemmaModelAction(action)) {
    throw new Error(
      `Recall EmbeddingGemma model command invalid: ${RECALL_EMBEDDINGGEMMA_MODEL_COMMAND_USAGE}`,
    );
  }
  if (
    flags.some((flag) => flag !== '--approve') ||
    flags.filter((flag) => flag === '--approve').length > 1
  ) {
    throw new Error(
      `Recall EmbeddingGemma model command arguments invalid: ${argumentsList.join(' ')}; ${RECALL_EMBEDDINGGEMMA_MODEL_COMMAND_USAGE}`,
    );
  }
  return {
    action,
    approved: flags.includes('--approve'),
  };
}

/** Runs one inspect, status, doctor, download, repair, or remove operation and emits JSON. */
export async function runRecallEmbeddingGemmaModelCommand(
  argumentsList: readonly string[],
  options: RecallEmbeddingGemmaModelCommandOptions = {},
): Promise<void> {
  const { action, approved } = parseRecallEmbeddingGemmaModelArguments(argumentsList);
  const homeDirectory = options.homeDirectory ?? homedir();
  const cacheDirectory =
    options.cacheDirectory ??
    process.env.PI_RECALL_MODEL_CACHE_DIRECTORY ??
    join(homeDirectory, '.pi', 'agent', 'recall', 'models');
  const cache = createRecallModelArtifactCache({
    cacheDirectory,
    profile: options.profile ?? createRecommendedEmbeddingGemmaModelProfile(),
    ...(options.transport ? { transport: options.transport } : {}),
  });

  const result = await (async () => {
    switch (action) {
      case 'inspect':
        return cache.inspectArtifact();
      case 'status':
      case 'verify':
        return cache.verifyArtifact();
      case 'doctor':
        return cache.diagnoseArtifact();
      case 'download':
        return cache.downloadArtifact({ approved });
      case 'repair':
        return cache.repairArtifact({ approved });
      case 'remove':
        return cache.removeArtifact({ approved });
    }
  })();
  const writeOutput =
    options.writeOutput ?? ((value: string) => process.stdout.write(`${value}\n`));
  writeOutput(JSON.stringify(result));
}

const invokedPath = process.argv[1];
if (invokedPath && import.meta.url === pathToFileURL(invokedPath).href) {
  runRecallEmbeddingGemmaModelCommand(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
