## Agent skills

### Issue tracker

Issues are tracked in `Whamp/pi-session-recall` on GitHub. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical triage labels without aliases. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout. See `docs/agents/domain.md`.

## Cursor Cloud specific instructions

The bundled `node:sqlite` must include FTS5, so this project requires Node.js 24+.
The default cloud toolchain's Node 22 ships a SQLite without FTS5 and fails the recall
tests with `no such module: fts5`. CI pins Node 24. In Cloud Agents, Node 24 is
installed via nvm and preferred on `PATH` (ahead of the platform default) so `node`,
`npm`, and the `psr` CLI resolve to it.

Development validation (see the README "Development validation" section):

- `npm test`
- `npm run typecheck`
- `npm run lint`
- `npm run format:check`
