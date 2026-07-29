# Validation

Validate every pull request against the current integration branch before marking it ready or merging it.

## Use the current base

Fetch `origin` before final validation. The branch merge base must match `origin/master`:

```bash
git fetch origin
test "$(git merge-base HEAD origin/master)" = "$(git rev-parse origin/master)"
```

If `master` advances, rebase the branch and rerun the full validation. Results from an older base are stale.

Confirm that the tested local head matches the remote pull-request head before merging.

## Run the full suite serially

Targeted tests support red-green development, but they do not replace the full suite. Run one full suite at a time:

```bash
npm test

# Include private replay evidence when the corpus is available.
PI_SESSION_IMPORT_CORPUS_ROOT=/path/to/private/session-corpus npm test
```

Do not run full suites concurrently across worktrees. The suites share native dependencies and process-level resources; concurrent runs can create failures that do not reproduce serially.

When the private replay corpus is available, include it in the final run. Record every skipped test. Only tests intentionally gated by opt-in data or host expense may remain skipped.

If a frozen replay digest changes, do not regenerate the expectation immediately. Identify the first behavior-changing commit and explain why the new digest is correct. Then update the expectation and its schema version when the evidence format or identity changed.

## Run repository gates

Run these gates after the final rebase and final changes:

```bash
npm run typecheck
npm run lint
npm run format:check
npm audit --omit=dev
git diff --check
```

For nontrivial TypeScript changes, also run CodeGraph impact and cycle checks and an exact-base slop scan as required by the workspace coding policy.

## Investigate ownership before masking a flake

Do not hide a timeout or cleanup race by increasing its wait first. Reproduce the smallest failing sequence, inspect terminal failure state instead of polling until timeout, and identify which process, file handle, native resource, or temporary directory remains owned. Add a deterministic regression at the public seam before changing production behavior.

A pull request is ready only when the complete serial suite and all required gates pass on its exact merge base.
