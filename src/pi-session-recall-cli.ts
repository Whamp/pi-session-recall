#!/usr/bin/env -S node --import tsx

import { pathToFileURL } from 'node:url';

import { createConfiguredRecallInferenceRuntime } from './configured-recall-inference-runtime.js';
import {
  loadRecallConversationConfig,
  type RecallConversationConfig,
} from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationService,
} from './recall-conversation-service.js';
import {
  createOctenEmbeddingModelProfile,
  createRecommendedEmbeddingGemmaModelProfile,
} from './recall-model-profiles.js';

const PI_SESSION_RECALL_USAGE =
  'usage: pi-session-recall <setup|status|rebuild|stop|resume|discard>';

interface RecallSetupProfilePresentation {
  profile: string;
  nativeDimensions: number;
  defaultStoredDimensions: number;
  allowedStoredDimensions: readonly number[] | Readonly<{ minimum: number; maximum: number }>;
  evidenceStatus: 'verified-mrl' | 'vendor-supported-prefix';
  evidenceSources: readonly string[];
}

interface PiSessionRecallCliServiceRuntime {
  service: Pick<
    RecallConversationService,
    | 'discardStagingIndexGeneration'
    | 'readOperatorStatus'
    | 'resumeBackgroundIndexGeneration'
    | 'startBackgroundIndexGeneration'
    | 'stopBackgroundIndexGeneration'
  >;
  dispose(): Promise<void>;
}

interface PiSessionRecallCliOptions {
  createServiceRuntime?(
    this: void,
    config: RecallConversationConfig,
    requiresInference: boolean,
  ): PiSessionRecallCliServiceRuntime | Promise<PiSessionRecallCliServiceRuntime>;
  writeOutput?(this: void, value: string): void;
}

function createRecallSetupProfilePresentations(): RecallSetupProfilePresentation[] {
  const octen = createOctenEmbeddingModelProfile({
    requestModel: 'octen-embed',
    servedModelId: 'Octen/Octen-Embedding-4B',
    artifact: 'Octen-Embedding-4B.Q8_0.gguf',
    artifactRepository: 'Octen/Octen-Embedding-4B',
    artifactRevision: 'configured',
    artifactSha256: '0'.repeat(64),
    dimensions: 2_560,
    quantization: 'Q8_0',
    pooling: 'last',
    normalization: 'l2',
  });
  const embeddingGemma = createRecommendedEmbeddingGemmaModelProfile();
  return [
    {
      profile: 'octen-embedding-4b',
      nativeDimensions: octen.identity.dimensions,
      defaultStoredDimensions: octen.storedDimensions,
      allowedStoredDimensions: {
        minimum: octen.storedDimensionRange.minimum,
        maximum: octen.storedDimensionRange.maximum,
      },
      evidenceStatus: octen.storedDimensionRange.evidenceStatus,
      evidenceSources: [...octen.storedDimensionEvidenceSources],
    },
    {
      profile: 'embeddinggemma-300m',
      nativeDimensions: embeddingGemma.nativeDimensions,
      defaultStoredDimensions: embeddingGemma.storedDimensions,
      allowedStoredDimensions: embeddingGemma.storedDimensionChoices.map(
        ({ dimensions }) => dimensions,
      ),
      evidenceStatus: 'verified-mrl',
      evidenceSources: [...embeddingGemma.storedDimensionEvidenceSources],
    },
  ];
}

async function createDefaultRecallOperatorRuntime(
  config: RecallConversationConfig,
  requiresInference: boolean,
): Promise<PiSessionRecallCliServiceRuntime> {
  if (requiresInference) {
    return createConfiguredRecallInferenceRuntime(config);
  }
  return {
    service: createRecallConversationService(config),
    async dispose() {},
  };
}

/** Adapts standalone operator arguments to configured recall service operations. */
export async function runPiSessionRecallCli(
  argumentsList: readonly string[],
  options: PiSessionRecallCliOptions = {},
): Promise<void> {
  const [command, ...commandArguments] = argumentsList;
  const configuredOutput = options.writeOutput;
  const writeOutput =
    configuredOutput === undefined
      ? (value: string) => process.stdout.write(`${value}\n`)
      : (value: string) => configuredOutput(value);
  if (command === 'setup' && commandArguments.length === 0) {
    await loadRecallConversationConfig();
    writeOutput(
      JSON.stringify({ command: 'setup', profiles: createRecallSetupProfilePresentations() }),
    );
    return;
  }
  const knownCommand =
    command === 'status' ||
    command === 'rebuild' ||
    command === 'stop' ||
    command === 'resume' ||
    command === 'discard';
  if (!knownCommand || commandArguments.length !== 0) {
    throw new Error(
      `Pi session recall command invalid: ${argumentsList.join(' ') || 'missing'}; ${PI_SESSION_RECALL_USAGE}`,
    );
  }
  const config = await loadRecallConversationConfig();
  const configuredRuntimeFactory = options.createServiceRuntime;
  const runtime = await (configuredRuntimeFactory === undefined
    ? createDefaultRecallOperatorRuntime(config, command === 'rebuild' || command === 'resume')
    : configuredRuntimeFactory(config, command === 'rebuild' || command === 'resume'));
  try {
    if (command === 'status') {
      const status = await runtime.service.readOperatorStatus();
      writeOutput(JSON.stringify({ command: 'status', ...status }));
      return;
    }
    if (command === 'rebuild') {
      const processStatus = await runtime.service.startBackgroundIndexGeneration();
      writeOutput(JSON.stringify({ command: 'rebuild', process: processStatus }));
      return;
    }
    if (command === 'resume') {
      const processStatus = await runtime.service.resumeBackgroundIndexGeneration();
      writeOutput(JSON.stringify({ command: 'resume', process: processStatus }));
      return;
    }
    if (command === 'stop') {
      const processStatus = await runtime.service.stopBackgroundIndexGeneration();
      writeOutput(JSON.stringify({ command: 'stop', process: processStatus }));
      return;
    }
    const discarded = await runtime.service.discardStagingIndexGeneration();
    writeOutput(JSON.stringify({ command: 'discard', discarded }));
  } finally {
    await runtime.dispose();
  }
}

const INVOKED_PATH = process.argv[1];
if (INVOKED_PATH && import.meta.url === pathToFileURL(INVOKED_PATH).href) {
  runPiSessionRecallCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
