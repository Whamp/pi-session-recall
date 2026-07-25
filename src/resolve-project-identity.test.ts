import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { RecallProjectIdentitySource } from './enums.js';
import { resolveProjectIdentity } from './resolve-project-identity.js';

const EXEC_FILE_ASYNC = promisify(execFile);

void test('Git project identity canonicalizes equivalent credentialed SSH and HTTPS origins', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-git-origin-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sshClone = join(directory, 'ssh-clone');
  const httpsClone = join(directory, 'https-clone');
  await Promise.all([mkdir(sshClone), mkdir(httpsClone)]);
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: sshClone });
  await EXEC_FILE_ASYNC(
    'git',
    ['remote', 'add', 'origin', 'git@GitHub.com:Whamp/pi-session-recall.git'],
    { cwd: sshClone },
  );
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: httpsClone });
  await EXEC_FILE_ASYNC(
    'git',
    ['remote', 'add', 'origin', 'https://access-token@github.com/Whamp/pi-session-recall.git/'],
    { cwd: httpsClone },
  );

  const sshIdentity = await resolveProjectIdentity(sshClone);
  const httpsIdentity = await resolveProjectIdentity(httpsClone);

  assert.deepEqual(sshIdentity, {
    projectIdentity: 'git-origin:github.com/Whamp/pi-session-recall',
    identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
  });
  assert.deepEqual(httpsIdentity, sshIdentity);
  assert.ok(!httpsIdentity?.projectIdentity.includes('access-token'));
});

void test('Git project identity keeps non-default origin ports distinct', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-git-origin-port-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const firstClone = join(directory, 'first-clone');
  const secondClone = join(directory, 'second-clone');
  await Promise.all([mkdir(firstClone), mkdir(secondClone)]);
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: firstClone });
  await EXEC_FILE_ASYNC(
    'git',
    ['remote', 'add', 'origin', 'http://gitea.example.test:3000/acme/web.git'],
    { cwd: firstClone },
  );
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: secondClone });
  await EXEC_FILE_ASYNC(
    'git',
    ['remote', 'add', 'origin', 'http://gitea.example.test:3001/acme/web.git'],
    { cwd: secondClone },
  );

  const firstIdentity = await resolveProjectIdentity(firstClone);
  const secondIdentity = await resolveProjectIdentity(secondClone);

  assert.equal(firstIdentity?.projectIdentity, 'git-origin:gitea.example.test:3000/acme/web');
  assert.equal(secondIdentity?.projectIdentity, 'git-origin:gitea.example.test:3001/acme/web');
  assert.notDeepEqual(firstIdentity, secondIdentity);
});

void test('Git project identity uses one real common directory for a main checkout and worktree', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-git-common-directory-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const mainCheckout = join(directory, 'main-checkout');
  const worktreeCheckout = join(directory, 'feature-worktree');
  await mkdir(mainCheckout);
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: mainCheckout });
  await EXEC_FILE_ASYNC(
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
  await EXEC_FILE_ASYNC('git', ['remote', 'add', 'origin', 'not a usable hosted remote'], {
    cwd: mainCheckout,
  });
  await EXEC_FILE_ASYNC('git', ['worktree', 'add', worktreeCheckout, '-b', 'feature'], {
    cwd: mainCheckout,
  });

  const mainIdentity = await resolveProjectIdentity(mainCheckout);
  const worktreeIdentity = await resolveProjectIdentity(worktreeCheckout);

  assert.equal(mainIdentity?.identitySource, RecallProjectIdentitySource.GIT_COMMON_DIRECTORY);
  assert.equal(mainIdentity?.projectIdentity, `git-common-directory:${join(mainCheckout, '.git')}`);
  assert.deepEqual(worktreeIdentity, mainIdentity);
});

void test('project identity uses an exact existing non-Git session origin and rejects missing origins', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-non-git-origin-'));
  t.after(() => rm(directory, { recursive: true, force: true }));

  assert.deepEqual(await resolveProjectIdentity(directory), {
    projectIdentity: `non-git-session-origin:${directory}`,
    identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
  });
  assert.equal(await resolveProjectIdentity(join(directory, 'deleted')), null);
});
