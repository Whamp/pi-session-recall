import {
  configureRecallInferenceCapability,
  inspectRecallInferenceConfiguration,
  readRecallInferenceConfiguration,
  removeRecallInferenceCapability,
  repairRecallInferenceCapability,
  type RecallInferenceCapability,
  type RecallInferenceConfigurationCandidate,
} from './recall-inference-configuration.js';

const RECALL_INFERENCE_SETUP_USAGE =
  'usage: inference [status|doctor|configure CAPABILITY CANDIDATE [--approve-artifact] [--approve-replacement]|repair CAPABILITY [--approve-artifact]|remove CAPABILITY]';

/** Candidate catalog, state location, clock, and machine-readable output for inference setup. */
export interface RecallInferenceSetupCommandOptions {
  statePath: string;
  candidates: readonly RecallInferenceConfigurationCandidate[];
  nowIsoTimestamp?: () => string;
  writeOutput?: (value: string) => void;
}

type RecallInferenceSetupAction =
  | { action: 'status' }
  | { action: 'doctor' }
  | {
      action: 'configure';
      capability: RecallInferenceCapability;
      candidateId: string;
      approvedArtifactChange: boolean;
      approvedEmbeddingReplacement: boolean;
    }
  | {
      action: 'repair';
      capability: RecallInferenceCapability;
      approvedArtifactRepair: boolean;
    }
  | { action: 'remove'; capability: RecallInferenceCapability };

function parseRecallInferenceCapability(value: string | undefined): RecallInferenceCapability {
  if (value === 'embedding' || value === 'reranking' || value === 'query-planning') {
    return value;
  }
  throw new Error(
    `Recall inference setup capability invalid: ${value ?? 'missing'}; ${RECALL_INFERENCE_SETUP_USAGE}`,
  );
}

function parseRecallInferenceSetupAction(
  argumentsList: readonly string[],
): RecallInferenceSetupAction {
  if (argumentsList.length === 0 || (argumentsList.length === 1 && argumentsList[0] === 'status')) {
    return { action: 'status' };
  }
  if (argumentsList.length === 1 && argumentsList[0] === 'doctor') {
    return { action: 'doctor' };
  }
  const [action, capabilityValue, candidateId, ...flags] = argumentsList;
  const capability = parseRecallInferenceCapability(capabilityValue);
  if (action === 'configure' && candidateId) {
    const acceptedFlags = new Set(flags);
    if (
      acceptedFlags.size !== flags.length ||
      [...acceptedFlags].some(
        (flag) => flag !== '--approve-artifact' && flag !== '--approve-replacement',
      )
    ) {
      throw new Error(
        `Recall inference setup arguments invalid: ${argumentsList.join(' ')}; ${RECALL_INFERENCE_SETUP_USAGE}`,
      );
    }
    return {
      action,
      capability,
      candidateId,
      approvedArtifactChange: acceptedFlags.has('--approve-artifact'),
      approvedEmbeddingReplacement: acceptedFlags.has('--approve-replacement'),
    };
  }
  if (action === 'repair' && candidateId === undefined) {
    if (flags.length > 0) {
      throw new Error(
        `Recall inference setup arguments invalid: ${argumentsList.join(' ')}; ${RECALL_INFERENCE_SETUP_USAGE}`,
      );
    }
    return { action, capability, approvedArtifactRepair: false };
  }
  if (action === 'repair' && candidateId === '--approve-artifact' && flags.length === 0) {
    return { action, capability, approvedArtifactRepair: true };
  }
  if (action === 'remove' && candidateId === undefined && flags.length === 0) {
    return { action, capability };
  }
  throw new Error(
    `Recall inference setup arguments invalid: ${argumentsList.join(' ')}; ${RECALL_INFERENCE_SETUP_USAGE}`,
  );
}

function findRecallInferenceSetupCandidate(
  candidates: readonly RecallInferenceConfigurationCandidate[],
  capability: RecallInferenceCapability,
  candidateId: string,
): RecallInferenceConfigurationCandidate {
  const candidate = candidates.find(
    (entry) => entry.capability === capability && entry.candidateId === candidateId,
  );
  if (!candidate) {
    throw new Error(
      `Recall inference setup candidate unavailable for ${capability}: ${candidateId}; no adapter was substituted`,
    );
  }
  return candidate;
}

/** Runs one mixed-capability status, doctor, configure, repair, or removal operation. */
export async function runRecallInferenceSetupCommand(
  argumentsList: readonly string[],
  options: RecallInferenceSetupCommandOptions,
): Promise<void> {
  const parsedAction = parseRecallInferenceSetupAction(argumentsList);
  const writeOutput =
    options.writeOutput ?? ((value: string) => process.stdout.write(`${value}\n`));
  if (parsedAction.action === 'status' || parsedAction.action === 'doctor') {
    const status = await inspectRecallInferenceConfiguration(
      options.statePath,
      options.candidates,
      { verifyConformance: parsedAction.action === 'doctor' },
    );
    writeOutput(JSON.stringify({ action: parsedAction.action, ...status }));
    return;
  }
  if (parsedAction.action === 'configure') {
    const candidate = findRecallInferenceSetupCandidate(
      options.candidates,
      parsedAction.capability,
      parsedAction.candidateId,
    );
    const configuration = await configureRecallInferenceCapability(options.statePath, candidate, {
      approvedArtifactChange: parsedAction.approvedArtifactChange,
      approvedEmbeddingReplacement: parsedAction.approvedEmbeddingReplacement,
      ...(options.nowIsoTimestamp ? { nowIsoTimestamp: options.nowIsoTimestamp } : {}),
    });
    writeOutput(
      JSON.stringify({
        action: 'configure',
        capability: parsedAction.capability,
        selection:
          parsedAction.capability === 'embedding'
            ? configuration.embedding
            : parsedAction.capability === 'reranking'
              ? configuration.reranking
              : configuration.queryPlanning,
      }),
    );
    return;
  }
  if (parsedAction.action === 'repair') {
    const configuration = await readRecallInferenceConfiguration(options.statePath);
    const selection =
      parsedAction.capability === 'embedding'
        ? configuration.embedding
        : parsedAction.capability === 'reranking'
          ? configuration.reranking
          : configuration.queryPlanning;
    if (!selection) {
      throw new Error(
        `Recall inference setup cannot repair unconfigured ${parsedAction.capability}`,
      );
    }
    const candidate = findRecallInferenceSetupCandidate(
      options.candidates,
      parsedAction.capability,
      selection.candidateId,
    );
    const repaired = await repairRecallInferenceCapability(
      options.statePath,
      parsedAction.capability,
      candidate,
      {
        approvedArtifactRepair: parsedAction.approvedArtifactRepair,
        ...(options.nowIsoTimestamp ? { nowIsoTimestamp: options.nowIsoTimestamp } : {}),
      },
    );
    writeOutput(
      JSON.stringify({
        action: 'repair',
        capability: parsedAction.capability,
        configuration: repaired,
      }),
    );
    return;
  }
  const configuration = await removeRecallInferenceCapability(
    options.statePath,
    parsedAction.capability,
  );
  writeOutput(
    JSON.stringify({ action: 'remove', capability: parsedAction.capability, configuration }),
  );
}
