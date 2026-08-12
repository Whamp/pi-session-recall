import {
  LocalOctenModelDownloadProgressKind,
  LocalOctenModelStatusKind,
} from './enums.js';
import {
  createLocalOctenModelManager,
  type LocalOctenModelManager,
} from './local-octen-model-manager.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { confirmTerminalAction } from './terminal-confirm.js';

const PSR_MODEL_USAGE = [
  'psr model status',
  'psr model download [--yes]',
  'psr model doctor',
].join('\n');

/** Replaceable command boundaries for local model CLI behavior. */
export interface PsrModelCliDependencies {
  loadManager: () => Promise<LocalOctenModelManager>;
  confirm: (question: string) => Promise<boolean>;
  writeOutput: (text: string) => void;
  writeProgress: (text: string) => void;
}

const DEFAULT_PSR_MODEL_CLI_DEPENDENCIES: PsrModelCliDependencies = {
  async loadManager() {
    const config = await loadRecallConversationConfig();
    return createLocalOctenModelManager({
      modelRootDirectory: config.localModelRootDirectory,
    });
  },
  confirm: confirmTerminalAction,
  writeOutput(text) {
    process.stdout.write(text);
  },
  writeProgress(text) {
    process.stderr.write(text);
  },
};

function formatGibibytes(bytes: number): string {
  return `${(bytes / 1_073_741_824).toFixed(2)} GiB`;
}

/** Runs `psr model` status, explicit download, and read-only diagnosis commands. */
export async function runPsrModelCli(
  argumentsList: readonly string[],
  dependencies: PsrModelCliDependencies = DEFAULT_PSR_MODEL_CLI_DEPENDENCIES,
): Promise<number> {
  const subcommand = argumentsList[0];
  const validDownload =
    subcommand === 'download' &&
    (argumentsList.length === 1 ||
      (argumentsList.length === 2 && argumentsList[1] === '--yes'));
  if (
    !(
      (subcommand === 'status' && argumentsList.length === 1) ||
      validDownload ||
      (subcommand === 'doctor' && argumentsList.length === 1)
    )
  ) {
    throw new Error(PSR_MODEL_USAGE);
  }

  const manager = await dependencies.loadManager();
  if (subcommand === 'status') {
    const status = await manager.status();
    dependencies.writeOutput(
      [
        `Model: ${status.kind}`,
        `Artifact: ${status.artifactId}`,
        `Size: ${formatGibibytes(status.totalBytes)}`,
        `Path: ${status.modelDirectory}`,
        `Detail: ${status.detail}`,
      ].join('\n') + '\n',
    );
    return status.kind === LocalOctenModelStatusKind.READY ? 0 : 1;
  }

  if (subcommand === 'doctor') {
    const diagnosis = await manager.doctor();
    dependencies.writeOutput(
      [
        `Model doctor: ${diagnosis.healthy ? 'healthy' : 'unhealthy'}`,
        `Artifact: ${diagnosis.status.artifactId}`,
        `Path: ${diagnosis.status.modelDirectory}`,
        `Detail: ${diagnosis.detail}`,
        ...(diagnosis.runtime
          ? [
              `Runtime vector: ${diagnosis.runtime.dimensions} dimensions`,
              `Runtime norm: ${diagnosis.runtime.norm}`,
            ]
          : []),
      ].join('\n') + '\n',
    );
    return diagnosis.healthy ? 0 : 1;
  }

  const status = await manager.status();
  const approved =
    argumentsList[1] === '--yes' ||
    (await dependencies.confirm(
      `Download ${formatGibibytes(status.totalBytes)} local Octen model from the project release?`,
    ));
  if (!approved) {
    dependencies.writeOutput('Local Octen model download cancelled.\n');
    return 1;
  }
  const result = await manager.download({
    approved: true,
    onProgress(event) {
      switch (event.kind) {
        case LocalOctenModelDownloadProgressKind.PREPARING:
          dependencies.writeProgress('Preparing local Octen model download...\n');
          break;
        case LocalOctenModelDownloadProgressKind.DOWNLOADING_FILE:
          dependencies.writeProgress(`Downloading ${event.fileName ?? 'artifact file'}...\n`);
          break;
        case LocalOctenModelDownloadProgressKind.FILE_VERIFIED:
          dependencies.writeProgress(`Verified ${event.fileName ?? 'artifact file'}.\n`);
          break;
        case LocalOctenModelDownloadProgressKind.ACTIVATED:
          dependencies.writeProgress('Local Octen model activated.\n');
          break;
        default:
          throw new Error('Local Octen model download progress kind is unsupported');
      }
    },
  });
  dependencies.writeOutput(
    result.downloaded
      ? `Local Octen model installed at ${result.modelDirectory}.\n`
      : `Local Octen model is already verified at ${result.modelDirectory}.\n`,
  );
  return 0;
}
