import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { RecallProjectIdentitySource } from './enums.js';
import { resolveGitProjectIdentity } from './resolve-git-project-identity.js';

const execFileAsync = promisify(execFile);

void test('Git project identity canonicalizes equivalent credentialed SSH and HTTPS origins', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-git-origin-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sshClone = join(directory, 'ssh-clone');
  const httpsClone = join(directory, 'https-clone');
  await Promise.all([mkdir(sshClone), mkdir(httpsClone)]);
  await execFileAsync('git', ['init'], { cwd: sshClone });
  await execFileAsync(
    'git',
    ['remote', 'add', 'origin', 'git@GitHub.com:Whamp/pi-session-recall.git'],
    { cwd: sshClone },
  );
  await execFileAsync('git', ['init'], { cwd: httpsClone });
  await execFileAsync(
    'git',
    ['remote', 'add', 'origin', 'https://access-token@github.com/Whamp/pi-session-recall.git/'],
    { cwd: httpsClone },
  );

  const sshIdentity = await resolveGitProjectIdentity(sshClone);
  const httpsIdentity = await resolveGitProjectIdentity(httpsClone);

  assert.deepEqual(sshIdentity, {
    projectIdentity: 'git-origin:github.com/Whamp/pi-session-recall',
    identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
  });
  assert.deepEqual(httpsIdentity, sshIdentity);
  assert.ok(!httpsIdentity?.projectIdentity.includes('access-token'));
});

void test('Git project identity uses one real common directory for a main checkout and worktree', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-git-common-directory-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const mainCheckout = join(directory, 'main-checkout');
  const worktreeCheckout = join(directory, 'feature-worktree');
  await mkdir(mainCheckout);
  await execFileAsync('git', ['init'], { cwd: mainCheckout });
  await execFileAsync(
    'git',
    [
      '-c',
      'user.name=Recall Test',
      '-c',
      'user.email=recall@example.test',
      'commit',
      '--allow-empty',
      '-m',
      'initial',
    ],
    { cwd: mainCheckout },
  );
  await execFileAsync('git', ['remote', 'add', 'origin', 'not a usable hosted remote'], {
    cwd: mainCheckout,
  });
  await execFileAsync('git', ['worktree', 'add', worktreeCheckout, '-b', 'feature'], {
    cwd: mainCheckout,
  });

  const mainIdentity = await resolveGitProjectIdentity(mainCheckout);
  const worktreeIdentity = await resolveGitProjectIdentity(worktreeCheckout);

  assert.equal(mainIdentity?.identitySource, RecallProjectIdentitySource.GIT_COMMON_DIRECTORY);
  assert.equal(mainIdentity?.projectIdentity, `git-common-directory:${join(mainCheckout, '.git')}`);
  assert.deepEqual(worktreeIdentity, mainIdentity);
});

void test('Git project identity reports missing and non-Git historical origins as unresolved', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-git-unresolved-'));
  t.after(() => rm(directory, { recursive: true, force: true }));

  assert.equal(await resolveGitProjectIdentity(directory), null);
  assert.equal(await resolveGitProjectIdentity(join(directory, 'deleted')), null);
});
