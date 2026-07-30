import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { runPiSessionRecallCli } from './pi-session-recall-cli.js';

const config = await loadRecallConversationConfig();
const command = process.argv[2];
const service =
  command === 'catch-up' || command === 'recover' || command === 'rollback'
    ? createRecallBackgroundIndexWorkerFixtureService(config)
    : createRecallConversationService(config, {
        backgroundIndexServiceFactory: {
          moduleUrl: new URL(
            './createRecallBackgroundIndexWorkerFixtureService.ts',
            import.meta.url,
          ).href,
          exportName: 'createRecallBackgroundIndexWorkerFixtureService',
        },
      });

await runPiSessionRecallCli(process.argv.slice(2), {
  async createServiceRuntime() {
    return { service, async dispose() {} };
  },
});
