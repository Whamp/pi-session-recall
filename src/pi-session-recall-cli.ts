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
import { createRecommendedOctenHttpCandidateId } from './recommended-octen-inference-candidate.js';
import { runRecallFirstIndexSetupCli } from './runRecallFirstIndexSetupCli.js';

const PI_SESSION_RECALL_USAGE =
  'usage: pi-session-recall <setup|status|catch-up|rebuild|stop|resume|discard|recover|rollback|cleanup>';
const PI_SESSION_RECALL_OCTEN_SETUP_USAGE =
  'usage: pi-session-recall setup select-octen [--stored-dimensions 1..2560] [--approve-replacement]';

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
    | 'catchUpRecallGeneration'
    | 'collectRetired'
    | 'discardStagingIndexGeneration'
    | 'readOperatorStatus'
    | 'recoverRecallMaintenance'
    | 'resumeBackgroundIndexGeneration'
    | 'rollback'
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

function createOctenSetupInferenceArguments(argumentsList: readonly string[]): string[] {
  let storedDimensions = 1_024;
  let storedDimensionsSelected = false;
  let approvedReplacement = false;
  for (let index = 1; index < argumentsList.length; index += 1) {
    const argument = argumentsList[index];
    if (argument === '--stored-dimensions' && !storedDimensionsSelected) {
      storedDimensions = Number(argumentsList[index + 1]);
      storedDimensionsSelected = true;
      index += 1;
      continue;
    }
    if (argument === '--approve-replacement' && !approvedReplacement) {
      approvedReplacement = true;
      continue;
    }
    throw new Error(
      `Pi session recall Octen setup arguments invalid: ${argumentsList.join(' ')}; ${PI_SESSION_RECALL_OCTEN_SETUP_USAGE}`,
    );
  }
  const candidateId = createRecommendedOctenHttpCandidateId(storedDimensions);
  return [
    'inference',
    'configure',
    'embedding',
    candidateId,
    ...(approvedReplacement ? ['--approve-replacement'] : []),
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
  if (command === 'setup') {
    if (commandArguments.length === 0) {
      await loadRecallConversationConfig();
      writeOutput(
        JSON.stringify({ command: 'setup', profiles: createRecallSetupProfilePresentations() }),
      );
      return;
    }
    await runRecallFirstIndexSetupCli(
      commandArguments[0] === 'select-octen'
        ? createOctenSetupInferenceArguments(commandArguments)
        : commandArguments,
    );
    return;
  }
  const knownCommand =
    command === 'status' ||
    command === 'catch-up' ||
    command === 'rebuild' ||
    command === 'stop' ||
    command === 'resume' ||
    command === 'discard' ||
    command === 'recover' ||
    command === 'rollback' ||
    command === 'cleanup';
  if (!knownCommand || commandArguments.length !== 0) {
    throw new Error(
      `Pi session recall command invalid: ${argumentsList.join(' ') || 'missing'}; ${PI_SESSION_RECALL_USAGE}`,
    );
  }
  const config = await loadRecallConversationConfig();
  const configuredRuntimeFactory = options.createServiceRuntime;
  const runtime = await (configuredRuntimeFactory === undefined
    ? createDefaultRecallOperatorRuntime(
        config,
        command === 'catch-up' ||
          command === 'rebuild' ||
          command === 'resume' ||
          command === 'recover',
      )
    : configuredRuntimeFactory(
        config,
        command === 'catch-up' ||
          command === 'rebuild' ||
          command === 'resume' ||
          command === 'recover',
      ));
  try {
    if (command === 'status') {
      const status = await runtime.service.readOperatorStatus();
      writeOutput(JSON.stringify({ command: 'status', ...status }));
      return;
    }
    if (command === 'catch-up') {
      const catchUp = await runtime.service.catchUpRecallGeneration();
      writeOutput(JSON.stringify({ command: 'catch-up', ...catchUp }));
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
    if (command === 'recover') {
      const recovery = await runtime.service.recoverRecallMaintenance();
      writeOutput(JSON.stringify({ command: 'recover', ...recovery }));
      return;
    }
    if (command === 'rollback') {
      const rollback = await runtime.service.rollback();
      writeOutput(JSON.stringify({ command: 'rollback', ...rollback }));
      return;
    }
    if (command === 'cleanup') {
      const cleanup = await runtime.service.collectRetired();
      writeOutput(JSON.stringify({ command: 'cleanup', ...cleanup }));
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
