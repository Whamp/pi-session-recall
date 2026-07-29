import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';

const DATA_DIRECTORY = process.argv[2];
const SESSIONS_DIRECTORY = process.argv[3];
if (!DATA_DIRECTORY || !SESSIONS_DIRECTORY) {
  throw new Error('Recall background index launcher fixture requires data and session directories');
}

const CONFIG = await loadRecallConversationConfig({
  environment: {
    PI_RECALL_DATA_DIRECTORY: DATA_DIRECTORY,
    PI_RECALL_SESSIONS_DIRECTORY: SESSIONS_DIRECTORY,
    PI_RECALL_EMBEDDING_DIMENSIONS: '3',
  },
});
const SERVICE = createRecallConversationService(CONFIG, {
  backgroundIndexServiceFactory: {
    moduleUrl: new URL('./createRecallBackgroundIndexWorkerFixtureService.ts', import.meta.url)
      .href,
    exportName: 'createRecallBackgroundIndexWorkerFixtureService',
  },
});
const STATUS = await SERVICE.startBackgroundIndexGeneration();
process.stdout.write(`${JSON.stringify(STATUS)}\n`);
