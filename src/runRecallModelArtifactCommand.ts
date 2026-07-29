import { homedir } from 'node:os';
import { join } from 'node:path';

import {
  createRecallModelArtifactCache,
  type RecallDownloadableModelProfile,
  type RecallModelArtifactTransport,
} from './recall-model-artifact-cache.js';

type RecallModelArtifactAction =
  | 'inspect'
  | 'status'
  | 'verify'
  | 'doctor'
  | 'download'
  | 'repair'
  | 'remove';

/** Model-specific labels, profile, and injectable boundaries for one artifact command. */
export interface RecallModelArtifactCommandOptions {
  commandUsage: string;
  errorPrefix: string;
  profile: Readonly<RecallDownloadableModelProfile>;
  homeDirectory?: string;
  cacheDirectory?: string;
  transport?: RecallModelArtifactTransport;
  writeOutput?: (value: string) => void;
}

function isRecallModelArtifactAction(value: string): value is RecallModelArtifactAction {
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

function parseRecallModelArtifactArguments(
  argumentsList: readonly string[],
  options: RecallModelArtifactCommandOptions,
): { action: RecallModelArtifactAction; approved: boolean } {
  const [action, ...flags] = argumentsList;
  if (!action || !isRecallModelArtifactAction(action)) {
    throw new Error(`${options.errorPrefix} command invalid: ${options.commandUsage}`);
  }
  if (
    flags.some((flag) => flag !== '--approve') ||
    flags.filter((flag) => flag === '--approve').length > 1
  ) {
    throw new Error(
      `${options.errorPrefix} command arguments invalid: ${argumentsList.join(' ')}; ${options.commandUsage}`,
    );
  }
  return { action, approved: flags.includes('--approve') };
}

/** Runs one deterministic model artifact inspection or explicitly approved mutation. */
export async function runRecallModelArtifactCommand(
  argumentsList: readonly string[],
  options: RecallModelArtifactCommandOptions,
): Promise<void> {
  const { action, approved } = parseRecallModelArtifactArguments(argumentsList, options);
  const cacheDirectory =
    options.cacheDirectory ??
    process.env.PI_RECALL_MODEL_CACHE_DIRECTORY ??
    join(options.homeDirectory ?? homedir(), '.pi', 'agent', 'recall', 'models');
  const cache = createRecallModelArtifactCache({
    cacheDirectory,
    profile: options.profile,
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
      default:
        throw new Error(`Recall model artifact action unsupported: ${String(action)}`);
    }
  })();
  const writeOutput =
    options.writeOutput ?? ((value: string) => process.stdout.write(`${value}\n`));
  writeOutput(JSON.stringify(result));
}
