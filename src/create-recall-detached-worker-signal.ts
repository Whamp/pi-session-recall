import { spawn } from 'node:child_process';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { readNodeErrorCode } from './read-node-error-code.js';

/** Fire-and-forget signal that starts a detached incremental recall worker. */
export interface RecallDetachedWorkerSignal {
  signalDetachedWorker(): void;
}

/** Injectable worker command used to prove detached ownership handoff behavior. */
export interface CreateRecallDetachedWorkerSignalOptions {
  workerExecutablePath?: string;
  workerArguments?: readonly string[];
  workingDirectory?: string;
}

// File descriptor 9 coalesces one waiter until descriptor 8 owns the worker lock, then unlocks so the next occupied interval can queue its own successor.
const RECALL_WORKER_SIGNAL_HANDOFF_SCRIPT = `
signal_lock_path=$1
ownership_lock_path=$2
node_executable_path=$3
worker_executable_path=$4
shift 4
exec 9>"$signal_lock_path"
/usr/bin/flock --nonblock 9 || exit 0
exec 8>"$ownership_lock_path"
/usr/bin/flock 8
/usr/bin/flock --unlock 9
exec "$node_executable_path" --import tsx "$worker_executable_path" "$@"
`;

/** Creates one coalesced detached successor that waits outside Pi for worker ownership. */
export function createRecallDetachedWorkerSignal(
  workerOwnershipLockPath: string,
  options: CreateRecallDetachedWorkerSignalOptions = {},
): RecallDetachedWorkerSignal {
  return {
    signalDetachedWorker() {
      const workerExecutablePath =
        options.workerExecutablePath ??
        fileURLToPath(new URL('./run-recall-incremental-worker.ts', import.meta.url));
      const child = spawn(
        '/bin/sh',
        [
          '-c',
          RECALL_WORKER_SIGNAL_HANDOFF_SCRIPT,
          'recall-worker-signal',
          `${workerOwnershipLockPath}.signal`,
          workerOwnershipLockPath,
          process.execPath,
          workerExecutablePath,
          ...(options.workerArguments ?? []),
        ],
        {
          cwd: options.workingDirectory ?? dirname(workerExecutablePath),
          detached: true,
          stdio: 'ignore',
        },
      );
      child.once('error', (error) => {
        process.emitWarning(
          `Recall marker worker signal failed [${readNodeErrorCode(error) ?? 'UNKNOWN'}]`,
        );
      });
      child.unref();
    },
  };
}
