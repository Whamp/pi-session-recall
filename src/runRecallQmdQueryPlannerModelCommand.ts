import { pathToFileURL } from 'node:url';

import { runRecallModelArtifactCommand } from './runRecallModelArtifactCommand.js';
import type { RecallModelArtifactTransport } from './recall-model-artifact-cache.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  type RecommendedQmdQueryPlanningModelProfile,
} from './recall-model-profiles.js';

const RECALL_QMD_QUERY_PLANNER_MODEL_COMMAND_USAGE =
  'usage: npm run model:qmd-query-planner -- <inspect|status|verify|doctor|download|repair|remove> [--approve]';

/** Injectable paths, profile, transport, and output for the QMD planner model command. */
export interface RecallQmdQueryPlannerModelCommandOptions {
  homeDirectory?: string;
  cacheDirectory?: string;
  profile?: RecommendedQmdQueryPlanningModelProfile;
  transport?: RecallModelArtifactTransport;
  writeOutput?: (value: string) => void;
}

/** Runs one QMD planner artifact inspection or explicitly approved mutation and emits JSON. */
export async function runRecallQmdQueryPlannerModelCommand(
  argumentsList: readonly string[],
  options: RecallQmdQueryPlannerModelCommandOptions = {},
): Promise<void> {
  await runRecallModelArtifactCommand(argumentsList, {
    commandUsage: RECALL_QMD_QUERY_PLANNER_MODEL_COMMAND_USAGE,
    errorPrefix: 'Recall QMD query planner model',
    profile: options.profile ?? createRecommendedQmdQueryPlanningModelProfile(),
    ...(options.homeDirectory ? { homeDirectory: options.homeDirectory } : {}),
    ...(options.cacheDirectory ? { cacheDirectory: options.cacheDirectory } : {}),
    ...(options.transport ? { transport: options.transport } : {}),
    ...(options.writeOutput ? { writeOutput: options.writeOutput } : {}),
  });
}

const INVOKED_PATH = process.argv[1];
if (INVOKED_PATH && import.meta.url === pathToFileURL(INVOKED_PATH).href) {
  runRecallQmdQueryPlannerModelCommand(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
