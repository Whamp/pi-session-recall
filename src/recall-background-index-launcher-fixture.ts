import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';

const dataDirectory = process.argv[2];
const sessionsDirectory = process.argv[3];
if (!dataDirectory || !sessionsDirectory) {
  throw new Error('Recall background index launcher fixture requires data and session directories');
}

const config = await loadRecallConversationConfig({
  environment: {
    PI_RECALL_DATA_DIRECTORY: dataDirectory,
    PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
    PI_RECALL_EMBEDDING_DIMENSIONS: '3',
  },
});
const service = createRecallConversationService(config, {
  backgroundIndexServiceFactory: {
    moduleUrl: new URL('./recall-background-index-worker-fixture.ts', import.meta.url).href,
    exportName: 'createRecallBackgroundIndexWorkerFixtureService',
  },
});
const status = await service.startBackgroundIndexGeneration();
process.stdout.write(`${JSON.stringify(status)}\n`);
