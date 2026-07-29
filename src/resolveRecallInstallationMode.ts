import { access, mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallInstallationMode } from './enums.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import {
  readRecallFirstIndexSetupState,
  resolveRecallFirstIndexSetupStatePath,
} from './recall-first-index-setup-command.js';
import { readRecallInferenceConfiguration } from './recall-inference-configuration.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const RECALL_LEGACY_OCTEN_MARKER_VERSION = 1;
const RECALL_LEGACY_OCTEN_MARKER_SCHEMA = Type.Object(
  {
    version: Type.Literal(RECALL_LEGACY_OCTEN_MARKER_VERSION),
    evidence: Type.Literal('legacy-index-manifest'),
  },
  { additionalProperties: false },
);

/** Resolves the durable migration marker that proves an installation predates inference setup. */
export function resolveRecallLegacyOctenMarkerPath(config: RecallConversationConfig): string {
  return join(dirname(config.manifestPath), 'legacy-octen-installation.json');
}

function resolveRecallInferenceConfigurationPath(config: RecallConversationConfig): string {
  return join(dirname(config.manifestPath), 'inference-configuration.json');
}

async function readLegacyOctenMarker(markerPath: string): Promise<boolean> {
  try {
    const parsed: unknown = JSON.parse(await readFile(markerPath, 'utf8'));
    Value.Parse(RECALL_LEGACY_OCTEN_MARKER_SCHEMA, parsed);
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall legacy Octen installation marker invalid at ${markerPath}: ${message}`,
      {
        cause: error,
      },
    );
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function writeLegacyOctenMarker(markerPath: string): Promise<void> {
  await mkdir(dirname(markerPath), { recursive: true });
  const temporaryPath = `${markerPath}.${process.pid}.tmp`;
  await writeFile(
    temporaryPath,
    `${JSON.stringify({ version: RECALL_LEGACY_OCTEN_MARKER_VERSION, evidence: 'legacy-index-manifest' })}\n`,
    'utf8',
  );
  await rename(temporaryPath, markerPath);
}

/** Distinguishes fresh setup from configured inference and proven pre-migration Octen installs. */
export async function resolveRecallInstallationMode(
  config: RecallConversationConfig,
): Promise<RecallInstallationMode> {
  const inferenceConfiguration = await readRecallInferenceConfiguration(
    resolveRecallInferenceConfigurationPath(config),
  );
  const firstIndexSetupState = await readRecallFirstIndexSetupState(
    resolveRecallFirstIndexSetupStatePath(config),
  );
  if (inferenceConfiguration.embedding || firstIndexSetupState.embedding) {
    return RecallInstallationMode.CONFIGURED;
  }

  const markerPath = resolveRecallLegacyOctenMarkerPath(config);
  if (await readLegacyOctenMarker(markerPath)) {
    return RecallInstallationMode.LEGACY_OCTEN;
  }
  if (await pathExists(config.manifestPath)) {
    await writeLegacyOctenMarker(markerPath);
    return RecallInstallationMode.LEGACY_OCTEN;
  }
  return RecallInstallationMode.UNCONFIGURED;
}

/** Refuses search or indexing until setup succeeds, except for proven legacy Octen installs. */
export function assertRecallInstallationConfigured(mode: RecallInstallationMode): void {
  if (mode !== RecallInstallationMode.UNCONFIGURED) {
    return;
  }
  throw new Error(
    'Recall inference is unconfigured: run setup:recall status and explicitly select a verified embedding before search or indexing',
  );
}
