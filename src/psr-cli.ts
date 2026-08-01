import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
  type RecallConversationService,
} from './recall-conversation-service.js';

const PSR_USAGE = 'psr usage: psr index [--rebuild]';

/** Replaceable process boundaries for the standalone `psr` command. */
export interface PsrCliDependencies {
  loadConfig: () => Promise<RecallConversationConfig>;
  createService: (config: RecallConversationConfig) => RecallConversationService;
  writeOutput: (text: string) => void;
}

const DEFAULT_PSR_CLI_DEPENDENCIES: PsrCliDependencies = {
  loadConfig: loadRecallConversationConfig,
  createService: createRecallConversationService,
  writeOutput(text) {
    process.stdout.write(text);
  },
};

/** Runs the complete standalone CLI; only explicit incremental indexing and rebuild can write. */
export async function runPsrCli(
  argumentsList: readonly string[],
  dependencies: PsrCliDependencies = DEFAULT_PSR_CLI_DEPENDENCIES,
): Promise<number> {
  if (argumentsList.length === 0 || argumentsList[0] === '--help') {
    dependencies.writeOutput(`${PSR_USAGE}\n`);
    return 0;
  }
  const rebuild = argumentsList.length === 2 && argumentsList[1] === '--rebuild';
  if (argumentsList[0] !== 'index' || (argumentsList.length !== 1 && !rebuild)) {
    throw new Error(PSR_USAGE);
  }

  const config = await dependencies.loadConfig();
  const result = await dependencies.createService(config).index({ rebuild, optimize: true });
  const summary = result.indexSummary;
  dependencies.writeOutput(
    [
      `Indexed ${summary.indexedSessions} of ${summary.scannedSessions} sessions`,
      `removed ${summary.removedSessions}`,
      `embedded ${summary.newlyEmbeddedChunks}`,
      `reused ${summary.reusedVectors} vectors`,
      `deleted ${summary.deletedChunks} documents`,
      `${result.totalChunks} searchable documents`,
      `${summary.failedSessions.length} failed sessions`,
    ].join(' · ') + '\n',
  );
  for (const failure of summary.failedSessions) {
    dependencies.writeOutput(`Failed: ${failure.sessionPath}: ${failure.error}\n`);
  }
  return summary.failedSessions.length === 0 ? 0 : 1;
}
