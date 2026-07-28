import { randomUUID } from 'node:crypto';
import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';

/** One crash-released kernel lock proving a replacement build is still live. */
export interface HeldRecallRebuildOwnershipLock {
  release(): Promise<void>;
}

/** Derives the rebuild-ownership lock beside the bounded write-window lock. */
export function recallRebuildOwnershipLockPath(writeWindowLockPath: string): string {
  return `${writeWindowLockPath}.rebuild-owner`;
}

function waitForRecallRebuildOwnership(
  child: ChildProcessWithoutNullStreams,
  token: string,
): Promise<boolean> {
  return new Promise((resolve, reject) => {
    let output = '';
    let settled = false;
    const cleanup = (): void => {
      child.stdout.off('data', onData);
      child.off('error', onError);
      child.off('exit', onExit);
    };
    const finish = (acquired: boolean, error?: Error): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      if (error) {
        reject(error);
      } else {
        resolve(acquired);
      }
    };
    const onData = (chunk: Buffer): void => {
      output += chunk.toString('utf8');
      if (output.includes(token)) {
        finish(true);
      }
    };
    const onError = (error: Error): void => {
      finish(false, new Error('Recall rebuild ownership lock failed', { cause: error }));
    };
    const onExit = (code: number | null): void => {
      if (code === 1) {
        finish(false);
      } else {
        finish(
          false,
          new Error(`Recall rebuild ownership lock exited before acquisition: ${code}`),
        );
      }
    };
    child.stdout.on('data', onData);
    child.once('error', onError);
    child.once('exit', onExit);
  });
}

/** Acquires rebuild ownership without waiting, or returns null while another build owns it. */
export async function tryAcquireRecallRebuildOwnershipLock(
  lockPath: string,
): Promise<HeldRecallRebuildOwnershipLock | null> {
  await mkdir(dirname(lockPath), { recursive: true });
  const token = `recall-rebuild-owner-${randomUUID()}`;
  const child = spawn('/usr/bin/flock', ['--exclusive', '--nonblock', lockPath, '/bin/cat'], {
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  // A losing nonblocking flock can close stdin before the token write reaches its pipe.
  child.stdin.on('error', () => undefined);
  child.stdin.write(`${token}\n`);
  if (!(await waitForRecallRebuildOwnership(child, token))) {
    child.stdin.end();
    return null;
  }
  return {
    async release() {
      if (child.exitCode !== null || child.signalCode !== null) {
        return;
      }
      const exited = new Promise<void>((resolve, reject) => {
        child.once('error', reject);
        child.once('exit', () => resolve());
      });
      child.stdin.end();
      await exited;
    },
  };
}
