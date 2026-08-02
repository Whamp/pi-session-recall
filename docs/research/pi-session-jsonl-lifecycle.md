# How Pi creates and manages session JSONL files

## Scope and source baseline

This document describes the session files created by the Pi coding-agent CLI in `@earendil-works/pi-coding-agent` 0.83.0. That is the version installed during this investigation. npm reports that package as built from commit [`845d6ff1f6643aba440341cce877ce1c43ebbc39`](https://github.com/earendil-works/pi/commit/845d6ff1f6643aba440341cce877ce1c43ebbc39), the `v0.83.0` release commit.

The main finding is simple:

> Pi keeps the current session as an in-memory append-only tree. For a new persistent session, it allocates the session ID, header, and future pathname immediately, but it normally does not create the JSONL file until the first finalized assistant message arrives. At that point it creates the file exclusively and writes the complete in-memory history. Every later entry is appended as one JSON object plus `\n`.

The coding-agent CLI uses `packages/coding-agent/src/core/session-manager.ts`, not the newer generic `JsonlSessionRepo` under `packages/agent/src/harness/session/`. The two implementations share a broad format but differ in creation timing, validation, entry IDs, and leaf persistence. This document treats the CLI implementation as authoritative for files produced by `pi`, then covers the generic harness separately.

## Ownership and call path

The CLI chooses or creates a `SessionManager` in `main.ts`. It passes that manager through runtime construction into `createAgentSession()`, which restores any existing context and records initial model and reasoning settings. `AgentSession` subscribes to agent events and persists finalized messages on `message_end`.

```text
CLI arguments/settings
        │
        ▼
main.createSessionManager()
        │  create / open / continue / fork / in-memory
        ▼
SessionManager
        │
        ├── createAgentSession(): restore context; append initial model/thinking
        │
        └── AgentSession._handleAgentEvent()
                 │
                 └── message_end → SessionManager.appendMessage()
                                      │
                                      └── _appendEntry() → _persist()
```

Primary source path:

1. [`main.ts:312-402`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/main.ts#L312-L402) selects `inMemory`, `forkFrom`, `open`, `continueRecent`, or `create` from CLI options.
2. [`sdk.ts:170-179`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/sdk.ts#L170-L179) creates a default `SessionManager` when an SDK caller did not supply one.
3. [`sdk.ts:187-238`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/sdk.ts#L187-L238) rebuilds existing context and restores model and thinking state.
4. [`sdk.ts:349-376`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/sdk.ts#L349-L376) gives the agent the session ID and appends initial model/thinking entries for a new session.
5. [`agent-session.ts:611-648`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L611-L648) persists only finalized user, assistant, tool-result, and custom messages from `message_end` events.

## Storage root and pathname

Pi's default agent directory is `~/.pi/agent`. `PI_CODING_AGENT_DIR` can replace that directory. The default sessions root is `<agent-dir>/sessions` ([`config.ts:510-554`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/config.ts#L510-L554)).

At CLI startup, session-directory precedence is:

1. `--session-dir`
2. `PI_CODING_AGENT_SESSION_DIR`
3. the `sessionDir` setting
4. the per-working-directory default

The first three are selected in [`main.ts:621-630`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/main.ts#L621-L630). If none is set, `getDefaultSessionDir()` resolves the working directory and encodes it as:

```text
~/.pi/agent/sessions/--ENCODED-CWD--/
```

Pi forms `ENCODED-CWD` from the absolute cwd by removing its leading separator and replacing each `/`, `\`, or `:` with `-`.

For example:

```text
/home/will/projects/pi-session-recall
→ ~/.pi/agent/sessions/--home-will-projects-pi-session-recall--/
```

The encoding and recursive directory creation are in [`session-manager.ts:476-489`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L476-L489).

A normal filename is:

```text
<ISO timestamp with ':' and '.' replaced by '-'>_<session ID>.jsonl
```

For example:

```text
2026-08-02T13-25-24-424Z_019fc2a6-9348-7c24-88e9-5581c0b55746.jsonl
```

`newSession()` creates the timestamp and path in [`session-manager.ts:930-955`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L930-L955). Generated session IDs are UUIDv7 values. A caller may supply an ID containing alphanumerics and interior `.`, `_`, or `-`; the ID must start and end with an alphanumeric character ([`session-manager.ts:208-219`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L208-L219)).

`PI_SESSION_FILE` is the allocated absolute pathname, not proof that the file already exists. Pi injects the current manager's path into bash-tool commands whenever the session is persistent ([`bash.ts:158-181`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/tools/bash.ts#L158-L181)).

## The in-memory model

`SessionManager` holds six pieces of session state:

- `fileEntries`: the header and every parsed/appended entry in physical order;
- `byId`: the latest entry object for each entry ID;
- `labelsById`: the latest non-empty label for each target;
- `labelTimestampsById`: the timestamp of that latest label change;
- `leafId`: the current tree position;
- `flushed`: whether the in-memory history has been written to the selected path.

These fields appear in [`session-manager.ts:855-866`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L855-L866).

The first physical record is a header, not a tree entry:

```json
{
  "type": "session",
  "version": 3,
  "id": "<session ID>",
  "timestamp": "<ISO timestamp>",
  "cwd": "<resolved absolute working directory>",
  "parentSession": "<optional source session path>"
}
```

Every later record is a tree entry with an entry `id`, `parentId`, and ISO `timestamp`. The header and entry interfaces are defined in [`session-manager.ts:30-156`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L30-L156).

Pi generates an entry ID by taking the first eight hexadecimal characters of a random UUID, retrying up to 100 times against the manager's current `byId` map. It falls back to a full UUID after 100 collisions ([`session-manager.ts:221-228`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L221-L228)).

## New-session creation, step by step

`SessionManager.create(cwd, sessionDir, options)` does the following:

1. Resolves the session directory and creates it recursively if needed.
2. Resolves the working directory.
3. Calls `newSession()` because no existing file was supplied.
4. Generates or validates the session ID.
5. Builds the version-3 header in memory.
6. Clears entry, index, label, and leaf state.
7. Sets `flushed = false`.
8. Computes and returns the future JSONL pathname.
9. Does **not** create that file yet.

The constructor and `newSession()` implement these steps in [`session-manager.ts:868-955`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L868-L955). The release test explicitly asserts that the pathname exists in manager state while no file exists on disk ([`custom-session-id.test.ts:73-83`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/test/session-manager/custom-session-id.test.ts#L73-L83)).

`createAgentSession()` then appends initial `model_change` and `thinking_level_change` entries for a new session. These remain in memory because no assistant entry exists yet ([`sdk.ts:362-374`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/sdk.ts#L362-L374)). Startup `--name` and extension state can also enter the in-memory chain before the first assistant response.

## The exact persistence algorithm

All entry append methods call `_appendEntry()`. It pushes the entry into `fileEntries`, inserts it into `byId`, moves `leafId` to the new entry, and calls `_persist()` ([`session-manager.ts:1044-1049`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1044-L1049)).

`_persist()` has three states ([`session-manager.ts:1015-1042`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1015-L1042)):

| Manager state                                                         | Disk action                                                                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| In-memory session, or no selected path                                | Return without file I/O.                                                                               |
| No assistant message exists and the file has never been flushed       | Keep all records in memory; leave the path absent.                                                     |
| No assistant exists but an opened/initialized file is already flushed | Append the new record.                                                                                 |
| An assistant exists and the file has never been flushed               | Open with `"wx"`; write the header and **all** in-memory entries, one line each; set `flushed = true`. |
| File is flushed                                                       | Append only the new record with `appendFileSync`.                                                      |

This design suppresses abandoned sessions that never receive an assistant response. It also means model changes, reasoning changes, names, extension entries, and the initial user message become durable together when the first assistant message is finalized.

The initial create uses exclusive mode (`wx`), so it refuses to overwrite an existing pathname. Subsequent writes use `appendFileSync`. There is no buffered close-time flush: append methods perform synchronous file I/O before they return.

The published 0.83.0 build behaved exactly this way in a temporary-directory probe:

```text
after create                     exists=false
after model + thinking + user    exists=false
after assistant                  exists=true
on-disk types                    session, model_change, thinking_level_change,
                                 message(user), message(assistant)
after custom entry               one custom line appended
```

## What becomes a record

The CLI `SessionManager` writes these top-level record types:

| `type`                  | Meaning                                                              | Context effect                                            |
| ----------------------- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| `session`               | File header and provenance                                           | Never a tree node or LLM message                          |
| `message`               | User, assistant, tool-result, custom-role, or bash-execution message | Usually participates in context according to message role |
| `thinking_level_change` | Effective reasoning level                                            | Restores session state                                    |
| `model_change`          | Provider and model ID                                                | Restores session state                                    |
| `compaction`            | Summary and kept-entry boundary                                      | Replaces old context logically; history stays on disk     |
| `branch_summary`        | Summary of an abandoned branch                                       | Injects summary context on the new path                   |
| `custom`                | Extension state                                                      | Omitted from default LLM context                          |
| `custom_message`        | Extension-injected content                                           | Included in LLM context                                   |
| `label`                 | Set or clear a label on another entry                                | Navigation metadata only                                  |
| `session_info`          | Current display name                                                 | Selector metadata only                                    |

The union is defined in [`session-manager.ts:53-156`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L53-L156). Each append method builds one typed object and delegates to `_appendEntry()` ([`session-manager.ts:1051-1191`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1051-L1191)).

Important message details:

- A tool call is a content block inside an assistant `message` record, not its own top-level record.
- A tool result is a later `message` record whose nested role is `toolResult`.
- Pi writes user, assistant, and tool-result messages only after the agent emits `message_end`; streaming deltas are not session records ([`agent-session.ts:611-648`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L611-L648)).
- A user-entered bash execution is wrapped as a `BashExecutionMessage` and stored in a top-level `message` record. If the agent is streaming, Pi queues that bash message and appends it after the turn to preserve tool-call/tool-result ordering ([`agent-session.ts:2807-2866`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L2807-L2866)).
- A name change sanitizes CR/LF to spaces and appends `session_info`; it does not rename the physical file ([`session-manager.ts:1136-1164`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1136-L1164)).

## Physical order and tree order

The file is append-only during ordinary conversation, but the conversation is not necessarily linear. Every new entry uses the current in-memory `leafId` as its `parentId`, then becomes the new leaf. Moving the leaf to an older entry makes the next append a second child of that entry; no old line changes ([`session-manager.ts:1044-1048`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1044-L1048), [`session-manager.ts:1360-1374`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1360-L1374)).

One consequence follows:

> The CLI format has no persisted leaf/cursor record. `branch()` and `resetLeaf()` only mutate memory. If the user navigates to an old entry and quits before Pi appends another entry, that navigation is lost. On reopen, Pi treats the last physical non-header record as the leaf.

`_buildIndex()` proves the reload behavior: it walks physical records in order and assigns `leafId = entry.id` for every non-header record ([`session-manager.ts:958-977`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L958-L977)). A runtime probe moved the in-memory leaf backward, reopened the file, and observed the leaf return to the last physical record.

Navigation becomes durable when it causes an append. A branch summary appends a `branch_summary` child at the selected position. A label also appends a record. Resubmitting an edited user prompt appends the new branch's first message. `AgentSession.navigateTree()` shows these cases in [`agent-session.ts:3025-3075`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L3025-L3075).

## Context reconstruction and compaction

Pi reconstructs the active branch by walking `parentId` from the leaf to the root, then reversing the result. A missing requested leaf falls back to the last physical entry. A missing parent silently ends the walk ([`session-manager.ts:325-360`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L325-L360)).

For the CLI manager, compaction never deletes or rewrites old conversation lines. It appends a `compaction` entry containing:

- the generated summary;
- `firstKeptEntryId`;
- `tokensBefore`;
- optional extension details, usage, and `fromHook`.

When rebuilding context, Pi finds the latest compaction on the active path and returns:

1. the compaction entry;
2. pre-compaction path entries starting at `firstKeptEntryId`;
3. every active-path entry appended after the compaction.

Older summarized records stay in the file but leave the LLM context. The transform is in [`session-manager.ts:411-455`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L411-L455); projection into runtime messages is in [`session-manager.ts:362-469`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L362-L469).

The CLI manager in 0.83.0 does **not** write `retainedTail` on compaction entries. That field belongs to the newer generic harness implementation described below. Pi's combined format documentation mentions both forms, so consumers should accept both even though current CLI-created compactions use `firstKeptEntryId`.

## Opening and parsing an existing file

`SessionManager.open()` resolves the explicit path, tries to read a header to recover `cwd`, derives the session directory from the file's parent when none was supplied, then constructs a manager around that path ([`session-manager.ts:1524-1550`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1524-L1550)).

The full loader:

1. Opens the file synchronously.
2. Reads 1 MiB byte chunks.
3. Uses `StringDecoder("utf8")` so a multibyte UTF-8 character can cross a chunk boundary.
4. Splits only on LF (`\n`). `JSON.parse` accepts a trailing CR as whitespace.
5. Attempts to parse the unterminated final record.
6. Skips blank lines and malformed JSON lines.
7. Accepts the file only when the first successfully parsed object is a `session` record with a string `id`.

See [`session-manager.ts:491-556`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L491-L556). Tests cover missing files, empty files, malformed lines, mixed valid/invalid lines, multibuffer headers, and files larger than Node's maximum string length ([`file-operations.test.ts:9-163`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/test/session-manager/file-operations.test.ts#L9-L163)).

The loader is intentionally permissive, not a graph validator. It does not reject:

- duplicate entry IDs;
- missing parents;
- cycles;
- multiple roots;
- multiple session headers after the first parsed header;
- unknown entry types;
- future session versions;
- invalid nested message shapes.

The source builds `byId` with last-write-wins map insertion and treats orphans as roots in `getTree()` ([`session-manager.ts:958-977`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L958-L977), [`session-manager.ts:1310-1352`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1310-L1352)). Consumers that need trustworthy branch semantics must validate more strictly than Pi's own loader.

### Header discovery

Listing and explicit open first use a faster header scanner. It reads 4 KiB chunks and stops after 1 MiB. It skips blank and malformed lines before the first parsed object. A parsed non-header object rejects the candidate. Discovery catches all scanner errors and ignores that file; explicit open falls back to the full loader when only the scan-size limit was exceeded ([`session-manager.ts:563-623`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L563-L623), [`session-manager.ts:1530-1549`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1530-L1549)).

### Empty, missing, and invalid explicit paths

Opening an explicit path has three different outcomes ([`session-manager.ts:891-928`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L891-L928)):

| Path state                                              | Result                                                                                                       |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Missing                                                 | Build a new in-memory header, preserve the explicit pathname, and defer creation until an assistant arrives. |
| Existing and empty                                      | Build a new header, immediately rewrite the empty file with that header, and mark it flushed.                |
| Existing, non-empty, and not recognized as a Pi session | Throw without modifying the file.                                                                            |

Tests verify that invalid non-empty files remain byte-for-byte unchanged and that empty files become one-header sessions ([`file-operations.test.ts:300-375`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/test/session-manager/file-operations.test.ts#L300-L375)).

## Migration and whole-file rewrites

The current CLI session version is 3.

- Version 1 has no header `version` and no entry IDs. Migration assigns random entry IDs, chains each physical entry to the previous one, converts compaction `firstKeptEntryIndex` to `firstKeptEntryId`, and marks the header version 2.
- Version 2 migration changes nested message role `hookMessage` to `custom` and marks the header version 3.
- Any header version greater than or equal to 3 passes through without a version check.

Migration code is in [`session-manager.ts:231-294`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L231-L294).

Opening an old session migrates it **in place** by truncating the original path with `openSync(path, "w")` and serializing the parsed in-memory records back to the file ([`session-manager.ts:895-921`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L895-L921), [`session-manager.ts:979-989`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L979-L989)). The rewrite is neither temp-file-plus-rename nor fsynced.

Because parsing skips malformed lines, migrating a v1 or v2 file drops those malformed lines permanently. A runtime probe opened a v2 file containing one malformed line; the rewritten v3 file contained only the header and valid entry.

Ordinary v3 resume does not rewrite the file. It leaves malformed physical lines in place while ignoring them in memory.

## Resume, continue, listing, and ordering

`--continue` calls `continueRecent()`. That function scans `.jsonl` files in the selected directory, keeps files with discoverable session headers, optionally filters by normalized header `cwd`, and chooses the greatest filesystem `mtime` ([`session-manager.ts:635-656`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L635-L656), [`session-manager.ts:1552-1565`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1552-L1565)).

`/resume`, `--resume`, and ID lookup use `SessionManager.list()` or `listAll()`. Listing reads up to ten files concurrently. For each file it counts every top-level `message` record, extracts user/assistant text for search, resolves the latest `session_info` name, and computes display `modified` as:

1. the greatest nested user/assistant message timestamp;
2. otherwise the header timestamp;
3. otherwise filesystem `mtime`.

It does not use tool-result, bash, label, custom, name, compaction, or branch-summary timestamps as activity. See [`session-manager.ts:658-765`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L658-L765).

This creates a real distinction:

- `--continue` chooses by filesystem `mtime`.
- session-list sorting chooses by logical user/assistant activity time when available.

A rename appends a line and updates filesystem `mtime`, so it can affect `--continue` selection while leaving the resume list's logical `modified` value unchanged.

For custom flat session directories, Pi filters current-project operations by header `cwd`; `listAll()` remains unfiltered ([`session-manager.ts:1632-1711`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1632-L1711)).

## `/new`, `/tree`, `/fork`, `/clone`, and cross-project fork

### `/new`

The runtime creates a fresh `SessionManager`, with a fresh ID and pathname, before tearing down the old runtime. It never reuses the outgoing session's physical file ([`agent-session-runtime.ts:226-259`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L226-L259)). The new file still follows delayed creation.

### `/tree`

`/tree` changes the in-memory leaf in the same file. Existing records remain untouched. A generated summary or subsequent message makes the branch durable by appending a child at the selected position. Leaf movement alone is not durable, as described above.

### `/fork` and `/clone`

Both commands route through `AgentSessionRuntime.fork()`:

- `/fork` selects a user entry and forks from its parent, placing the selected text back in the editor.
- `/clone` forks “at” the current leaf and copies the whole active branch.

The runtime behavior is in [`agent-session-runtime.ts:262-352`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L262-L352). The interactive `/clone` call passes `{ position: "at" }` in [`interactive-mode.ts:4620-4635`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/modes/interactive/interactive-mode.ts#L4620-L4635).

`createBranchedSession(leafId)` creates a new header and copies only the root-to-leaf path. It:

1. removes label records from that path;
2. re-chains retained non-label entries because labels may have been parents;
3. preserves retained entry IDs and payloads;
4. appends fresh label records for labels whose targets survived;
5. gives the new session a UUIDv7 ID and `parentSession` path;
6. replaces the manager's in-memory state with the extracted session.

If the retained path contains an assistant message, Pi writes the new file immediately. If not, it defers file creation until the new branch gets its first assistant response. Source: [`session-manager.ts:1412-1512`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1412-L1512). Regression tests cover both cases and assert one header with no duplicate IDs ([`tree-traversal.test.ts:480-597`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/test/session-manager/tree-traversal.test.ts#L480-L597)).

A persisted session cannot be forked or cloned before its first assistant response because its allocated path does not exist yet. The runtime reports that exact condition ([`agent-session-runtime.ts:290-318`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L290-L318)). A root fork is the exception: it can start another empty delayed session with `parentSession` set.

### Cross-project `--fork`

`SessionManager.forkFrom()` behaves differently from active-branch extraction. It creates the target file immediately with `wx`, writes a new header with the target cwd and source path, then copies every non-header parsed source record with its existing IDs and parent links. It finally reopens that new file ([`session-manager.ts:1572-1630`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1572-L1630)).

### JSONL import

Runtime import copies the selected file by basename into the current session directory unless source and destination are the same path, opens the destination, then replaces the runtime. `copyFileSync` uses its default overwrite behavior, so a same-named destination is replaced ([`agent-session-runtime.ts:361-395`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L361-L395)).

## Rename and deletion

Renaming a session does not rename its file or change its header. The selector opens the session and appends a `session_info` record; the latest such record supplies the display name ([`interactive-mode.ts:4804-4809`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/modes/interactive/interactive-mode.ts#L4804-L4809), [`session-manager.ts:1136-1164`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1136-L1164)).

Interactive deletion tries the external `trash` command first. A zero exit status or an already-absent path counts as success. Otherwise Pi falls back to permanent `unlink`. It then removes the item from both in-memory selector lists and refreshes them ([`session-selector.ts:640-680`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/modes/interactive/components/session-selector.ts#L640-L680), [`session-selector.ts:831-855`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/modes/interactive/components/session-selector.ts#L831-L855)). There is no tombstone or session index to update; deleting the file deletes the session.

## Shutdown and durability boundaries

Ordinary entries are synchronously written when appended, so the session manager has no close, save, or final-flush method. Runtime shutdown emits extension lifecycle events and disposes the agent, but it does not rewrite the session file ([`agent-session-runtime.ts:398-405`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L398-L405)).

Before session replacement (`new`, `resume`, or `fork`), the runtime aborts and waits for the active response to settle so the finalized aborted assistant message and tool results can reach normal `message_end` persistence before the old session is discarded ([`agent-session-runtime.ts:167-178`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L167-L178)).

On ordinary interactive quit, `dispose()` aborts the agent but does not await a final message before process exit. Entries already delivered to `SessionManager` are on disk; an in-flight stream that has not produced `message_end` is outside the session manager's durability boundary ([`interactive-mode.ts:3539-3603`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/modes/interactive/interactive-mode.ts#L3539-L3603), [`agent-session.ts:833-852`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L833-L852)).

A new session that exits before any assistant `message` record is finalized normally leaves no JSONL file. That includes sessions containing only initial model/thinking entries, a name, extension state, user input, or bash output.

## Atomicity, locking, and concurrent writers

The CLI implementation provides these write properties:

- first normal flush: exclusive `openSync(path, "wx")`;
- later entries: `appendFileSync(path, line)`;
- fork-from another project: exclusive header create followed by appends;
- migration, empty-file initialization, and immediate branch extraction: truncate-and-rewrite with `openSync(path, "w")`;
- no temp-file rename;
- no `fsync` or `fdatasync`;
- no session-file lock.

All session-manager file operations are visible in [`session-manager.ts:979-1042`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L979-L1042) and [`session-manager.ts:1412-1629`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1412-L1629).

The design assumes one writer per session file. Two processes that open the same session each build a private `byId` map and leaf from the same snapshot. Neither observes the other's later appends, and no lock serializes their parent selection or whole-file rewrites. Consumers should not treat concurrent writes to one session as supported.

A crash can therefore leave:

- no file for a pre-assistant session;
- a truncated final line during append;
- a partly rewritten file during migration or another whole-file rewrite;
- structurally competing branches after unsupported concurrent writers.

Pi's own loader recovers from some of these by skipping malformed lines and parsing an unterminated final record when complete, but it does not repair graph damage.

## The second implementation: generic harness JSONL storage

Pi 0.83.0 also exports `JsonlSessionRepo` and `JsonlSessionStorage` from `@earendil-works/pi-agent-core`. The coding-agent CLI does not import these classes; its source imports only its local `SessionManager`. These harness classes are a separate storage API, not the owner of normal `pi` CLI files.

The differences matter to parsers because harness-created files can live in the same broad format:

| Behavior         | Coding-agent CLI `SessionManager`                     | Generic harness `JsonlSessionRepo`                                                       |
| ---------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| File creation    | Delayed until first assistant, except special cases   | Header written immediately                                                               |
| I/O              | Synchronous Node `fs`                                 | Async injected `FileSystem`                                                              |
| Entry ID         | First 8 chars of random UUID                          | Last 8 chars of UUIDv7, collision checked                                                |
| Leaf persistence | No leaf record; inferred from last physical entry     | Appends explicit `leaf` records                                                          |
| Additional state | No `active_tools_change`                              | Supports `active_tools_change`                                                           |
| Compaction       | Required `firstKeptEntryId`; no `retainedTail` writer | Optional `firstKeptEntryId`; optional materialized `retainedTail`                        |
| Header metadata  | No generic metadata object                            | Optional `metadata` object                                                               |
| Parser           | Skips malformed lines; migrates v1/v2                 | Requires the first nonblank line to be a strict v3 header; rejects malformed later lines |
| Delete           | CLI selector: trash then unlink                       | Repository `remove(..., { force: true })`                                                |

Harness evidence:

- [`jsonl-repo.ts:38-99`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/harness/session/jsonl-repo.ts#L38-L99) creates directories, computes the same style of pathname, and creates storage immediately.
- [`jsonl-storage.ts:36-136`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/harness/session/jsonl-storage.ts#L36-L136) strictly parses header and entry framing.
- [`jsonl-storage.ts:187-282`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/harness/session/jsonl-storage.ts#L187-L282) writes the header immediately and appends entry or leaf records.
- [`types.ts:375-429`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/harness/types.ts#L375-L429) defines `active_tools_change` and `retainedTail`.
- [`session.ts:60-136`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/harness/session/session.ts#L60-L136) rebuilds compaction-aware context, including `retainedTail`.

The harness parser is stricter about physical JSONL syntax and required base fields, but it still does not perform a complete graph-validation pass at load. Parent existence is checked when traversing a branch, and a persisted leaf target is checked when read.

## Building a SQLite persistence extension

The lifecycle above is necessary but not sufficient to build a lossless SQLite mirror. Pi 0.83.0 has no public extension event that fires after every `SessionManager` append. Some useful events run after specific entries are saved, but messages expose a pre-persistence event, and several append paths have no extension event at all.

> No set of Pi extension hooks can, by itself, persist every session mutation. A complete design needs two layers: hook-driven snapshots of the current `SessionManager`, plus append-aware file reconciliation for session files and mutations that occur outside the active extension context.

This section defines “persist all sessions” as preserving every finalized Pi session record, including sessions resumed from existing files, branches, compactions, extension records, names, labels, custom session directories, and ephemeral `--no-session` runs observed by the extension. Streaming deltas are not session records and do not belong in the session mirror.

### Install the extension globally

Place the extension under `~/.pi/agent/extensions/`, not a project's `.pi/extensions/`. Project-local extensions run only in trusted projects and cannot cover other projects. A global extension still cannot observe a Pi process started with `--no-extensions` unless the extension is passed explicitly with `-e`; that process must be recovered later by scanning its JSONL files ([`args.ts:152-153`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/cli/args.ts#L152-L153)).

The extension factory may run for commands that never start a session. Pi's extension contract therefore says to open long-lived resources from `session_start`, not from the factory, and to close them idempotently from `session_shutdown`. Session replacement and `/reload` tear down the old extension instance and bind a new one ([`extensions.md`, “Long-lived resources and shutdown”](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/docs/extensions.md#long-lived-resources-and-shutdown)).

### Recommended hook set

Use each post-mutation hook as a request to reconcile `ctx.sessionManager`; do not translate the event payload directly into a synthetic session entry. The manager snapshot contains Pi's actual generated entry ID, parent ID, timestamp, normalized payload, and physical order.

| Hook                    | Timing in 0.83.0                                                                                                                                    | SQLite action                                                                                                                                                                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `session_start`         | After the manager, initial model/thinking entries, startup name, resumed history, and `ctx.newSession({ setup })` work are bound to the new runtime | Open the database and worker, register `getSessionDir()`, take a full header/entry/leaf baseline, and start current-directory watch/poll resources. Treat `getSessionFile()` as an allocated path that may not exist yet.                                         |
| `turn_end`              | After all `message_end` handlers and `SessionManager` appends for that LLM turn                                                                     | Request an incremental snapshot for lower latency. This catches user, assistant, and tool-result records for the completed turn.                                                                                                                                  |
| `agent_settled`         | After retries, automatic compaction, queued continuations, and pending bash-message flushes                                                         | Request the authoritative end-of-run snapshot. Prefer this to `agent_end` as the final agent boundary.                                                                                                                                                            |
| `session_compact`       | After Pi appends the `compaction` entry and rebuilds context                                                                                        | Reconcile immediately. This is required for manual compaction while no agent run is active. The event also supplies the saved entry.                                                                                                                              |
| `session_tree`          | After leaf movement and any branch summary or navigation label append                                                                               | Reconcile entries and record the current runtime leaf. A leaf-only move is runtime state, not durable CLI JSONL state.                                                                                                                                            |
| `session_info_changed`  | After the active manager appends `session_info`                                                                                                     | Reconcile the current name entry. This event does not cover startup `--name` or renaming an inactive session in the selector; startup baseline and file scanning cover those.                                                                                     |
| `model_select`          | After `model_change` and any resulting thinking-level append for direct model selection/cycling                                                     | Reconcile model and reasoning records. In current source, the event is awaited after the append.                                                                                                                                                                  |
| `thinking_level_select` | Triggered after `thinking_level_change` is appended                                                                                                 | Reconcile reasoning records. The call site launches this event without awaiting it, so serialize all database work through the extension's own queue.                                                                                                             |
| `session_shutdown`      | Before the old extension runtime is invalidated for quit, reload, new, resume, or fork                                                              | Stop new watch callbacks, take one final manager snapshot, await the write queue, checkpoint only if operational policy requires it, then close session-scoped resources. Use `event.reason` and `targetSessionFile` as transition metadata, not session records. |

The startup baseline includes state written before extension dispatch. Main appends `--name` before creating the agent session; the SDK appends initial model/thinking entries before constructing `AgentSession`; replacement-session `setup` runs before `finishSessionReplacement()` rebinds extensions; and `bindExtensions()` then emits `session_start` ([`main.ts:641-651`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/main.ts#L641-L651), [`sdk.ts:349-389`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/sdk.ts#L349-L389), [`agent-session-runtime.ts:226-259`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L226-L259), [`agent-session.ts:2234-2252`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L2234-L2252)).

Pi awaits low-level agent events in order. Each `message_end` listener completes, including the later `SessionManager` append, before `turn_end` begins ([`agent.ts:233-247`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/agent.ts#L233-L247), [`agent.ts:493-536`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/agent.ts#L493-L536)). `agent_settled` is later still: `_runAgentPrompt()` handles retry and compaction loops, flushes queued bash messages in `finally`, then emits `agent_settled` ([`agent-session.ts:581-590`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L581-L590), [`agent-session.ts:1061-1103`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L1061-L1103)).

The post-session hooks have the expected ordering in source:

- compaction appends first and then awaits `session_compact` ([`agent-session.ts:1872-1893`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L1872-L1893), [`agent-session.ts:2153-2174`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L2153-L2174));
- tree navigation applies the branch, summary, and label changes before awaiting `session_tree` ([`agent-session.ts:3035-3078`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L3035-L3078));
- model selection appends before awaiting `model_select`, while thinking selection appends before launching `thinking_level_select` ([`agent-session.ts:1578-1593`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L1578-L1593), [`agent-session.ts:1676-1698`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L1676-L1698));
- name changes append before launching `session_info_changed` ([`agent-session.ts:2869-2877`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L2869-L2877)).

`session_before_switch`, `session_before_fork`, `session_before_compact`, and `session_before_tree` are cancellable pre-mutation events. They are useful for policy gates, not as evidence that a mutation completed. Pi 0.83.0 has no post-events named `session_switch` or `session_fork`; successful replacement is represented by `session_shutdown` on the old runtime followed by `session_start` on the new one ([`types.ts:558-668`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/extensions/types.ts#L558-L668)).

### Hooks that are not persistence boundaries

Do not write a final session record from these event payloads:

- `message_end` runs **before** `_handleAgentEvent()` calls `appendMessage()` or `appendCustomMessageEntry()`. Its `ctx.sessionManager` does not yet contain that message, and a later-loaded extension may still replace the message before Pi persists it ([`agent-session.ts:611-648`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L611-L648), [`agent-session.ts:712-781`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L712-L781)). It can mark the mirror dirty, but `turn_end`, `agent_settled`, or polling must read the generated entry afterward.
- `message_start` and `message_update` describe transient agent state. Pi never writes streaming deltas to session JSONL.
- `agent_end` follows the low-level run's messages, but precedes post-run retry, automatic compaction, queued continuation, and final pending-bash flush. Use it only as an optional early trigger.
- `tool_execution_end` and `tool_result` precede the final `toolResult` message event. They do not expose the session entry ID or parent ID.
- `model_select`, `thinking_level_select`, and `session_info_changed` report normalized state but not the generated session entry. Snapshot the manager instead of synthesizing one.

### Mutations with no public extension event

Periodic current-manager reconciliation is required because these paths append records without a corresponding public extension hook:

- `pi.appendEntry()` appends a `custom` entry. Pi emits an internal `entry_appended` event to its own UI listeners, but `entry_appended` is not part of `ExtensionAPI.on()` ([`agent-session.ts:2370-2393`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L2370-L2393), [`types.ts:1190-1230`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/extensions/types.ts#L1190-L1230)).
- `pi.setLabel()` appends a label without an extension event ([`agent-session.ts:2388-2392`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L2388-L2392)). Interactive label changes inside the tree selector also append directly.
- `pi.sendMessage()` with no triggered turn appends a `custom_message` directly and emits only internal UI events ([`agent-session.ts:1417-1461`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L1417-L1461)).
- An idle `!` or `!!` command appends a `bashExecution` message directly. Bash issued during streaming is queued and later covered by `agent_settled`, but idle bash has no post-persistence extension event ([`agent-session.ts:2807-2863`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts#L2807-L2863)).
- Startup `--name` appends before extension binding. `session_start` sees it, but no `session_info_changed` event fires ([`main.ts:641-651`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/main.ts#L641-L651)).
- The session selector can rename or delete an inactive session through a separately opened manager. The active extension context receives no event.
- Another Pi process, the generic harness repository, or an external tool can create, append, rewrite, move, or delete session files without touching this runtime.

The read-only extension context exposes `getHeader()`, `getEntries()`, `getLeafId()`, `getSessionName()`, `getSessionId()`, `getSessionFile()`, `getSessionDir()`, and tree/context readers. It does not expose an append callback, raw physical lines, byte offsets, file generation, or `isPersisted()` ([`session-manager.ts:190-207`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L190-L207)). Do not cast away the read-only type or monkey-patch append methods; that is an unsupported implementation dependency and misses other managers in the same process.

### Reconciliation design

Use one serialized, coalescing write queue per extension instance. Every hook above should mark the current source dirty. `session_start` must await database/worker initialization and durable source registration, but it should not parse and rewrite a large resumed session on Pi's thread. Queue the persisted-file baseline for the worker; synchronously capture only the bounded metadata and pending or ephemeral entries that do not yet exist on disk. Ordinary post-mutation hooks may enqueue lightweight work, but `session_shutdown` must stop producers and await the queue before closing the connection. This serialization also handles `thinking_level_select` and `session_info_changed`, whose call sites do not await extension completion.

For the active manager, a reconciliation pass should capture:

```text
getHeader()          session id, version, timestamp, cwd, parentSession
getEntries()         every parsed entry in physical/in-memory order
getLeafId()          current runtime leaf, including non-durable leaf-only navigation
getSessionName()     latest resolved name
getSessionFile()     allocated path, or undefined for an ephemeral session
getSessionDir()      current configured session directory
```

`getEntries()` returns a new array but only shallow-copies entry objects ([`session-manager.ts:1296-1303`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts#L1296-L1303)). A worker thread cannot read that live manager. Capture an immutable metadata or entry delta on Pi's thread, then transfer plain data to the worker. For an existing persisted file, let the file reconciler read bulk history instead of repeatedly serializing the whole manager. Capture or serialize any remaining in-memory state before allowing the old runtime to shut down. Never retain `ctx` or its manager across session replacement; Pi invalidates the old extension instance.

Make the active-manager importer append-aware:

1. Store the last mirrored entry count and a stable fingerprint of the mirrored prefix.
2. When the count grows and the prefix still matches, insert only the new ordinal suffix.
3. When the count shrinks or the prefix changes, reconcile the whole source generation. This handles migration, import overwrite, explicit rewrites, and future format changes.
4. Store the header independently because `getEntries()` excludes it.
5. Update the runtime leaf separately. For CLI files, a leaf-only `/tree` move is not a new physical record and will disappear after reopen.
6. For a persistent manager whose allocated path is absent, create a pending source keyed by that path. When the first assistant causes Pi's initial flush, attach the new file generation to the existing source instead of creating a duplicate session.
7. Give an ephemeral session a process-instance-qualified source key such as `ephemeral:<sessionId>:<runtimeInstanceId>`. Session ID alone is not a safe physical-occurrence key.

For physical files, run a second append-aware reconciler:

1. Register the default sessions root and every `ctx.sessionManager.getSessionDir()` observed at `session_start`. Arbitrary `--session-dir` values are not globally enumerable before a process using them starts.
2. Watch directories for low latency, but also poll and rescan. `fs.watch` can coalesce or lose events, and inode replacement can detach a file watch.
3. Track path, device/inode where available, size, modification time, generation fingerprint, last complete LF offset, and buffered tail bytes.
4. Parse new bytes only after LF. On quiescence, shutdown, or later recovery, attempt a complete unterminated final JSON record exactly as Pi and this project's canonical importer do.
5. If size decreases, inode changes, or the mirrored prefix changes, treat the file as rewritten and reconcile a new generation rather than appending blindly.
6. Mark missing paths as deleted or moved according to an explicit retention policy. Do not erase archived SQLite evidence merely because Pi moved a file to trash unless that is the product requirement.
7. Periodically scan the full registered roots so inactive-session rename/delete, processes with missed watch events, and hard-crash leftovers converge.

Do not call the existing whole-session `psr index` operation from an awaited Pi hook. This project already measured full parse/tokenize/index work blocking interactive Pi and deliberately moved it to an explicit CLI ([ADR-0003](../adr/0003-keep-whole-session-recall-maintenance-explicit.md)). A SQLite extension must do bounded append capture in the foreground and move database work, full-file reconciliation, tokenization, embeddings, and corpus repair to a worker or separate process.

### Minimum SQLite identity and schema rules

A database keyed only by `session.id` or `entry.id` will lose evidence:

- JSONL import can copy the same header ID to another path.
- Historical file reuse can place several logical session headers in one physical file.
- Pi's permissive loader can retain duplicate entry IDs.
- Forks preserve copied entry IDs under a new session header.
- Ephemeral sessions have no physical path.

At minimum, keep these identities separate:

| Entity          | Required identity                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| Observed source | Canonical physical path plus file generation, or a process-qualified ephemeral key                               |
| Logical session | Observed source plus header occurrence/segment; never header ID alone                                            |
| Physical record | Observed source generation plus physical ordinal or byte range                                                   |
| Semantic entry  | Logical session plus physical ordinal; index `entry.id` for graph lookup but do not make it the sole primary key |
| Runtime cursor  | Runtime instance plus logical session; store `leafId` separately from durable records                            |

A practical schema needs tables equivalent to:

- `session_sources`: path or ephemeral key, generation fingerprint, observed session directory, current/deleted state, and scan cursor;
- `logical_sessions`: source key, header occurrence, raw header JSON, session ID, version, cwd, timestamp, and parent-session path;
- `session_records`: logical-session key, physical ordinal, byte bounds when available, raw JSON, parsed type, entry ID, parent ID, and timestamp;
- `runtime_observations`: process/runtime identity, current source, effective leaf, name, and last reconciliation time;
- `session_roots`: every default or custom directory that future scans must revisit.

Store raw record JSON or bytes in addition to normalized columns if “persist” means archival fidelity. `ctx.sessionManager` exposes parsed objects only; it has already discarded malformed lines and cannot reproduce original whitespace, CRLF framing, duplicate headers excluded by `getEntries()`, or exact byte offsets. Raw fidelity therefore requires reading the file with the canonical byte-framing rules, not only manager snapshots.

Use WAL mode, a nonzero busy timeout, short transactions, and idempotent upserts if several Pi processes share one database. Serialize writes inside each process and test cross-process contention. Decide whether the database is an archive, a mirror that propagates deletion, or both; the session lifecycle has no natural “completed forever” event because any session can be resumed later.

Session data can contain complete prompts, assistant thinking, source files, shell output, images, credentials accidentally printed by tools, and extension-private data. Set restrictive file permissions and define encryption, retention, redaction, backup, and deletion behavior before treating the SQLite file as a durable archive.

### Decisions Pi does not make for the extension

The source and hooks cannot answer these product and implementation questions. Resolve them before implementation:

- **Fidelity:** Is SQLite a semantic mirror of parsed Pi entries, a byte-faithful archive of JSONL, or both?
- **Deletion:** Does file deletion tombstone the source while retaining history, purge SQLite rows, or move them to a separate archive?
- **Ephemeral sessions:** Should `--no-session` runs enter SQLite, and how long should they remain?
- **Coverage:** Must the system recover sessions created with `--no-extensions`, in arbitrary custom directories, on other machines, or by the generic harness?
- **Execution model:** Which SQLite binding, worker/daemon boundary, queue durability mechanism, migration framework, and database path will the extension use?
- **Concurrency:** Will every Pi process write one shared database, per-process databases merged later, or a single-writer daemon?
- **Failure policy:** Should database failure notify only, retry in memory, spool to another file, block shutdown, or fail Pi? Extension exceptions are logged and Pi continues.
- **Security:** What file permissions, encryption, redaction, retention, backup, and user deletion controls apply?
- **Upgrade policy:** Which Pi versions and session formats are supported, and what compatibility test gates an upgrade?

### Extension control skeleton

The hook wiring should have this shape. `requestReconcile()` must synchronously capture a bounded immutable delta, coalesce requests, and transfer plain data through a worker-backed queue. It should read manager state, not manufacture entries from event payloads.

```typescript
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';

export default function sqliteSessionMirror(pi: ExtensionAPI) {
  let mirror: SessionMirror | undefined;

  const reconcile = (ctx: ExtensionContext, reason: string) => {
    mirror?.requestReconcile(ctx.sessionManager, reason);
  };

  pi.on('session_start', async (event, ctx) => {
    mirror = await SessionMirror.open({
      sessionDir: ctx.sessionManager.getSessionDir(),
    });
    await mirror.reconcileNow(ctx.sessionManager, `session_start:${event.reason}`);
    mirror.startPolling(ctx.sessionManager);
  });

  pi.on('turn_end', (_event, ctx) => reconcile(ctx, 'turn_end'));
  pi.on('agent_settled', (_event, ctx) => reconcile(ctx, 'agent_settled'));
  pi.on('session_compact', (_event, ctx) => reconcile(ctx, 'session_compact'));
  pi.on('session_tree', (_event, ctx) => reconcile(ctx, 'session_tree'));
  pi.on('session_info_changed', (_event, ctx) => reconcile(ctx, 'session_info_changed'));
  pi.on('model_select', (_event, ctx) => reconcile(ctx, 'model_select'));
  pi.on('thinking_level_select', (_event, ctx) => reconcile(ctx, 'thinking_level_select'));

  pi.on('message_end', () => {
    // Pre-persistence only: mark dirty; never insert a final entry here.
    mirror?.markDirty();
  });

  pi.on('session_shutdown', async (event, ctx) => {
    if (!mirror) return;
    mirror.stopPolling();
    await mirror.reconcileNow(ctx.sessionManager, `session_shutdown:${event.reason}`);
    await mirror.flush();
    await mirror.close();
    mirror = undefined;
  });
}
```

The skeleton intentionally omits a SQLite library and watcher implementation. Those choices depend on whether writes run in a worker thread, a separate daemon, or an asynchronous native binding. A synchronous SQLite binding used for full snapshots inside awaited event handlers would recreate the interaction latency that ADR-0003 removed.

### Required validation scenarios

Before calling the extension complete, test at least these cases against the published Pi build:

1. New persistent session before and after the delayed first-assistant file creation.
2. A no-tool turn, a multi-tool turn, parallel tools, steering, follow-up messages, retries, aborted messages, and `agent_settled` ordering.
3. Idle and streaming `!`/`!!` bash commands.
4. `pi.appendEntry()`, `pi.sendMessage()` with no triggered turn, and `pi.setLabel()` from another test extension.
5. Startup `--name`, active `/name`, inactive selector rename, and name clearing.
6. Model and thinking changes, including a model change that clamps thinking.
7. Manual, threshold, and overflow compaction.
8. `/tree` with and without summary, root navigation, labels, and a leaf-only move followed by quit/reopen.
9. `/new`, `/resume`, `/fork`, `/clone`, JSONL import overwrite, and `/reload`.
10. Default, custom `--session-dir`, and `--no-session` sessions.
11. v1/v2 migration rewrite, malformed lines, an unterminated final record, truncation, inode replacement, and historical multiple-header files.
12. Inactive-file deletion, trash moves, and changes from another Pi process.
13. Two Pi processes writing one SQLite database under lock contention.
14. Graceful shutdown, `SIGTERM`, and hard kill followed by restart reconciliation.
15. A process launched with `--no-extensions`, recovered by the next corpus scan.

Pin these tests to Pi 0.83.0. The event names are public API, but several ordering and bypass conclusions above come from current implementation source. Re-audit the dispatch sites when upgrading Pi.

## Building a Zvec incremental recall index

Zvec should be a derived retrieval index over canonical session evidence, not the sole archive. Its collection model stores strongly typed documents with scalar fields and vectors; it does not preserve malformed JSONL lines, original whitespace, byte framing, or runtime state unless the application stores those facts explicitly. Keep JSONL or the SQLite/raw-record layer authoritative and make Zvec disposable and rebuildable.

The source index for this section is Zvec's official [`llms.txt`](https://zvec.org/llms.txt), accessed on 2026-08-02. It links the rolling official documentation cited below. This project pins `@zvec/zvec` 0.6.0, so implementation must also compile and test every documented call against that installed package; the Zvec website does not version these pages to the package pin.

### Storage role and process boundary

Zvec runs in-process and persists each collection in a self-contained directory. Documents in one collection share a schema, and Zvec does not support joins, unions, or cross-collection searches ([data modeling](https://zvec.org/en/docs/db/concepts/data-modeling/)). Use one recall collection unless the application deliberately fans queries across collections and merges the results itself.

Do not open one writable collection from every Pi process. The official open-collection guide recommends read-only mode when several processes share a collection ([open a collection](https://zvec.org/en/docs/db/collections/open/)). It does not document safe multi-process writers. Use one of these ownership models:

1. **Preferred:** Pi extensions append small dirty-source notices or pending-session snapshots to a durable queue; one daemon owns the writable Zvec collection.
2. **Compatible with current project policy:** `psr index` acquires the existing operation lock and remains the only writer. Extensions and search open Zvec read-only.
3. **Not recommended:** Every Pi extension opens the collection for writes and relies on timing. Zvec's official docs do not establish this as safe.

The first model preserves [ADR-0003](../adr/0003-keep-whole-session-recall-maintenance-explicit.md): awaited Pi hooks do bounded capture only. Canonical parsing, tokenization, embedding, Zvec upserts, deletion, and optimization run outside the interactive process. A Zvec-writing lifecycle extension without that process boundary would supersede ADR-0003 and needs a new architectural decision.

### Chunk contract: 512 tokens with overlap

Use the project's accepted chunk policy as the indexing assumption:

```text
maximum chunk size: 512 embedding-model tokens
overlap ceiling:     64 tokens between adjacent conversational chunks
stride:              variable, because chunks prefer natural text boundaries
```

The existing manifest freezes this as `{ maxTokens: 512, overlapTokens: 64 }` ([`recall-index-manifest.ts`](../../src/recall-index-manifest.ts)). The current chunker chooses a natural boundary at or before the 512-token hard limit, starts the next conversational chunk with at most 64 tokens from the previous chunk, and records token and character spans ([`session-conversation-index.ts`](../../src/session-conversation-index.ts)). Verbatim tool evidence remains token-bounded and atomic rather than adding conversational overlap.

“512-token chunks” means 512 tokens under the pinned embedding-model tokenizer, not 512 words, characters, bytes, JSON tokens, or Zvec full-text tokens. Persist these compatibility inputs in the index manifest:

- embedding request model, served model ID, native and stored dimensions, and vector transformation;
- tokenizer model, revision, library version, options, and asset hashes;
- `maxTokens: 512`, `overlapTokens: 64`, and boundary-algorithm version;
- canonical JSONL import policy and conversation-document schema versions;
- Zvec scalar/vector schema, full-text configuration, metric, index type, and index parameters.

Any change that can alter chunk boundaries, document IDs, vector dimensions, or similarity semantics requires a rebuild. The official docs require every dense vector to match the schema's exact dimension and recommend choosing a metric that matches the embedding model ([schema](https://zvec.org/en/docs/db/collections/create/schema/), [data modeling](https://zvec.org/en/docs/db/concepts/data-modeling/)).

### Collection schema

Install the Node package as `@zvec/zvec`. Create a collection with `ZVecCreateAndOpen()` and reopen it with `ZVecOpen()` ([quickstart](https://zvec.org/en/docs/db/quickstart/), [open a collection](https://zvec.org/en/docs/db/collections/open/)). Define one document per searchable chunk.

| Document component         | Required contents                                                                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                       | Deterministic hash of logical-session occurrence, document kind, contributing entry identity, text-run identity, and chunk span. Never use entry ID alone. Supported imports must first receive a fresh logical session ID. |
| `fields.content`           | Exact chunk text, indexed with Zvec FTS when lexical recall is required.                                                                                                                                                    |
| `fields.identifierContent` | Optional case-preserving FTS projection for paths, symbols, hashes, and other identifiers.                                                                                                                                  |
| Source identity            | Canonical path or ephemeral source key, source generation, logical header occurrence, session ID, parent-session path, cwd, and project identity.                                                                           |
| Graph provenance           | Entry/parent/contributing IDs, branch/compaction provenance, current leaf observation, active-branch/context flags, and source line/block/span coordinates.                                                                 |
| Chunk geometry             | Tokenizer identity reference, token/character bounds, token count, overlap count, text-run ID, chunk index/count, and sibling IDs.                                                                                          |
| Change detection           | `contentChecksum` for embedding reuse and `documentChecksum` over every persisted scalar field that affects retrieval or provenance.                                                                                        |
| Vector                     | Dense FP32 embedding under a fixed field name and dimension. Lexical-only documents need an explicit policy, such as a zero vector plus `isDenseSearchable = false`, if the schema requires a vector.                       |

Zvec enforces field types and vector dimensions at ingestion. Scalar fields used frequently in filters should have inverted indexes; unindexed fields can still be filtered but more slowly ([data modeling](https://zvec.org/en/docs/db/concepts/data-modeling/), [conditional filtering](https://zvec.org/en/docs/db/data-operations/query/filter/)). Index at least the fields used to scope recall, such as project-identity digest, source generation, document kind, and dense-searchability flag. Use a full-text index for chunk content; FTS tokenizes text and ranks results with BM25 ([full-text index](https://zvec.org/en/docs/db/concepts/fts-index/)).

The Node examples accept vectors generated elsewhere. Zvec's official embedding-function page currently documents Python implementations, not a Node embedding API ([embedding models](https://zvec.org/en/docs/ai/embedding/)). A TypeScript Pi extension therefore needs its own embedding provider. This project already uses a direct Octen HTTP provider and stores normalized FP32 prefixes with inner-product search under [ADR-0005](../adr/0005-store-normalized-octen-prefixes-with-inner-product.md).

### Changed-file incremental indexing

For this design, **incremental indexing means a changed physical JSONL file produces only the required Zvec document deletes, metadata upserts, and new embeddings**. It does not require a tail-only semantic parser. Re-parsing one changed file is acceptable and is currently safer because one appended record can change graph-derived metadata on older chunks: the current leaf, child IDs, active branch, visible context, compaction coverage, branch paths, or session name.

Use this algorithm for every scan or dirty-path batch:

1. **Discover candidates.** Coalesce watcher/hook notices by canonical path, then perform a periodic directory scan to recover missed events, inactive-session changes, and `--no-extensions` processes.
2. **Classify the file.** Compare the recorded source generation, device/inode where available, size, modification time, and a prefix fingerprint. An unchanged file does no parse, embedding, or Zvec write.
3. **Frame canonical records.** Read complete LF-framed records and follow the project's import policy for CRLF, malformed lines, unterminated final records, migrations, and multiple-header histories. Do not advance the durable cursor past an incomplete tail.
4. **Rebuild the changed file's semantic projection.** Validate each logical session, graph, branch/compaction links, and current CLI or harness leaf. A future append-aware graph parser may resume from a checkpoint, but it must prove the same output as full changed-file parsing.
5. **Chunk deterministically.** Produce token-bounded documents under the 512/64 policy. A changed seam can replace the previous final chunk and create new siblings; deterministic IDs and checksums expose that delta.
6. **Fetch existing rows and vectors by candidate IDs.** Zvec supports direct fetch by document ID ([fetch](https://zvec.org/en/docs/db/data-operations/fetch/)). Batch requests to bound memory and native-call overhead.
7. **Classify each current chunk.** Use both checksums:
   - same `documentChecksum` and required vector present: no-op;
   - changed `documentChecksum`, same `contentChecksum`: reuse the stored vector and upsert metadata;
   - new ID, changed content, or missing vector: embed content and upsert the complete document;
   - lexical-only row: upsert only when its document checksum changes.
8. **Find stale IDs.** Compare the previous file manifest with the current deterministic IDs. Stale chunks include deleted records, changed chunk boundaries, removed logical sessions, and superseded file generations.
9. **Upsert before delete.** Batch `upsertSync()` or `upsert()` calls, verify every returned status, and only then delete stale IDs. Zvec upsert overwrites a document with the same ID; batch validation errors reject the whole batch, but operational failures can affect individual documents, so every status matters ([upsert](https://zvec.org/en/docs/db/data-operations/upsert/)).
10. **Commit the sidecar checkpoint last.** Record source generation, file observations, canonical parse boundary, chunk IDs, both checksums, and embedding identity only after all required Zvec operations succeed. A retry must be idempotent.
11. **Schedule optimization separately.** Newly written vectors are queryable immediately but first accumulate in a flat buffer. Optimize by policy, not once per changed file.

This is incrementality at three levels:

- unchanged files are skipped;
- unchanged documents in a changed file are not rewritten;
- unchanged content keeps its embedding even when provenance metadata changes.

A pure “append new chunks after the old byte offset” algorithm is incorrect without dependency invalidation. At minimum, invalidate the prior open text run and its overlap seam. Also update older chunks whose graph-derived metadata changed. A whole changed-file semantic pass with document-level Zvec deltas is the safe baseline; optimize parsing only after equivalence tests cover branches, compactions, names, tool pairing, multi-header files, and rewrites.

### Rewrite, deletion, and failure behavior

Handle source changes by class, not only by `mtime`:

| Source event                                                                           | Zvec action                                                                                                                                                                          |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Append-only growth                                                                     | Reconcile the changed file; embed only new or content-changed 512-token chunks and upsert affected metadata.                                                                         |
| Truncation, inode replacement, prefix mismatch, import overwrite, or migration rewrite | Start a new source generation and perform a full semantic diff.                                                                                                                      |
| Temporary malformed/incomplete tail                                                    | Keep the previous indexed generation queryable, record the error, and retry after quiescence. Do not delete good evidence merely because the active file is between writes.          |
| File removed                                                                           | Delete its chunk IDs only if product policy defines Zvec as a live mirror; otherwise tombstone the source in the authoritative ledger and retain or archive its retrieval documents. |
| Path move with same content                                                            | Either preserve source-occurrence identity and update path metadata, or model delete-plus-add. Decide this once and include it in the manifest.                                      |
| Ephemeral `--no-session` run                                                           | Persist a synthetic source occurrence from the manager snapshot into the durable queue, then chunk and index it. It cannot be recovered from a later filesystem scan.                |

Zvec deletes are immediate and irreversible ([delete documents](https://zvec.org/en/docs/db/data-operations/delete/)). Keep the previous source manifest until deletion succeeds, and never use `deleteByFilter()` with unescaped external values. Exact ID deletion is safer when the sidecar already records every chunk ID.

The official upsert API does not document a transaction spanning upsert, delete, and checkpoint updates. A crash can therefore expose a temporarily mixed projection. The sidecar must remain behind Zvec, retry idempotently, and reconcile to one state on restart. If atomic source-generation visibility is mandatory, this design needs an additional publication protocol or shadow-collection swap; Zvec's documented API does not supply that guarantee.

### Hook integration

Use the same Pi hook set documented in [Recommended hook set](#recommended-hook-set), but change the action from “write Zvec now” to “capture a dirty-source notice now.”

| Boundary                                                                             | Zvec pipeline action                                                                                                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `session_start`                                                                      | Register the session directory and source path; enqueue a baseline notice. Capture pending or ephemeral entries that do not exist on disk.   |
| `turn_end`, `agent_settled`, `session_compact`, `session_tree`, metadata/model hooks | Coalesce the current source path and latest complete offset into the durable queue. Do not tokenize or embed in the handler.                 |
| Poll/watch reconciliation                                                            | Enqueue changed paths that hooks cannot observe, including labels, custom entries, idle bash, inactive sessions, and other processes.        |
| `session_shutdown`                                                                   | Flush the bounded source notice or pending-session snapshot, not the full Zvec maintenance job. The single writer may finish indexing later. |

A dirty notice needs source identity and observations, not a claim that a specific entry was saved. The worker must reopen and reconcile the file because message hooks can be pre-persistence, watch events can coalesce, and a file may be rewritten before the worker reads it.

For persistent sessions before first-assistant flush, capture header and entries in the durable queue under the allocated path. Replace that pending snapshot with file reconciliation when the JSONL appears. For ephemeral sessions, keep the process-qualified source key permanently because no file will appear.

### Zvec writes and optimization

Zvec's Node quickstart uses `ZVecCreateAndOpen`, `ZVecOpen`, `collection.upsertSync()`, `collection.fetchSync()`, `collection.deleteSync()`, and vector or filtered queries ([quickstart](https://zvec.org/en/docs/db/quickstart/)). Keep these native calls inside the writer process. Check every status rather than assuming a batch succeeded.

New vectors first enter a temporary flat buffer. `optimize()` builds or refreshes the configured vector index in the background without blocking reads or writes. `collection.stats.indexCompleteness` reports the indexed fraction; the docs recommend balancing ingestion rate and search latency rather than optimizing small batches repeatedly ([optimize a collection](https://zvec.org/en/docs/db/collections/optimize/)). Trigger optimization from document-count or completeness thresholds, a maintenance timer, or explicit `psr index --optimize`. Do not call it from every Pi lifecycle event.

Use read-only `ZVecOpen(path, { readOnly: true })` in search processes. Reopen after writer publication if the installed package requires that for visibility; verify this behavior against 0.6.0 because the rolling docs do not state cross-process refresh semantics.

### Query implications

Zvec supports vector search with scalar filters, so project- or source-scoped dense recall can filter before ranking ([vector plus filter](https://zvec.org/en/docs/db/data-operations/query/hybrid/)). It also supports BM25 full-text search. One query route cannot combine FTS and vector input; the official FTS guide says to run separate routes and merge or rerank in the application ([full-text index](https://zvec.org/en/docs/db/concepts/fts-index/)).

That matches this project's accepted approach in [ADR-0002](../adr/0002-fuse-hybrid-retrieval-in-application-code.md): run bounded dense, ordinary FTS, and case-preserving identifier searches separately, retain component evidence, then fuse deterministically. Do not describe Zvec's “hybrid” vector-plus-filter route as dense-plus-keyword fusion.

### Fit with the current project

The repository already implements most of this derived-index shape:

- `src/incremental-session-indexer.ts` scans JSONL paths, skips unchanged size/mtime pairs, reparses changed files, diffs deterministic chunk IDs, fetches existing documents and vectors, embeds changed chunks, batches upserts, deletes stale IDs, and writes a sidecar state file.
- `src/session-conversation-index.ts` performs canonical import, graph validation, deterministic document creation, and 512/64 token-aware chunking.
- `src/zvec-conversation-store.ts` owns the typed Zvec schema, FTS fields, HNSW vector field, fetch/upsert/delete operations, filtered dense search, lexical search, and optimization.
- `src/recall-index-manifest.ts` binds import, embedding, tokenizer, chunking, schema, project identity, and Zvec index semantics.
- `src/recall-conversation-service.ts` holds the single-writer operation lock and runs Zvec optimization only after changed maintenance when requested.

Before adapting that code to continuous dirty-source processing, close these gaps:

1. **Split content and document checksums.** Current chunk `checksum` hashes content only, and the incremental writer skips rows when that hash matches. A session-name, leaf, branch, context-visibility, child-link, compaction, path, or project-attribution change can therefore leave Zvec scalar metadata stale even though the file was rescanned.
2. **Check Zvec statuses.** The current store discards upsert/delete status results. Official docs require checking each status because operational failures can be partial.
3. **Strengthen file generations.** Current state uses path, size, and `mtimeMs`. Add inode/prefix/generation evidence and a complete-record offset to distinguish append from rewrite and recover from coarse or preserved timestamps.
4. **Retain good evidence on transient parse failure.** Current changed-file failure removes the prior indexed session. Continuous indexing should quarantine the new observation and keep the last good generation until a policy says otherwise.
5. **Move bulk work out of Pi.** Keep the extension read-only apart from durable dirty notices; run parsing, tokenization, embeddings, and Zvec native writes in the single writer.
6. **Define cross-process visibility.** Test how Zvec 0.6.0 read-only handles observe writer updates and whether readers must reopen.
7. **Index frequent filter fields.** The official docs recommend inverted indexes for fields used in filters. Measure project-scoped query latency and add indexes through an explicit schema migration if needed.

These are implementation requirements, not reasons to rebuild every changed file into a fresh collection. The target remains incremental document reconciliation under one collection.

#### Accepted limitation: Pi's built-in `/import`

The supported import path will be a `/psr-import <path.jsonl>` extension command. It should validate the source, create a new session header with a fresh session ID, retain the imported conversation, switch to that new session, and enqueue it for indexing. Because the logical session ID changes, its Zvec document IDs cannot collide with the source session.

Pi extensions cannot replace or wrap the built-in `/import` command. Pi handles built-in commands before extension commands, and the public extension context has no import method. Pi's built-in import copies a JSONL file without changing its session ID or entries ([`agent-session-runtime.ts:361-383`](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts#L361-L383)). If someone uses it on an already indexed Pi session, the two physical files can generate the same Zvec IDs.

That collision is an accepted edge case. Do not add the physical path to every Zvec document ID solely to handle it. When `/psr-import` ships, its README documentation must say that it is the supported import command and that Pi's built-in `/import` bypasses the safeguard.

### Zvec-specific validation scenarios

Add these to the extension validation matrix:

1. An unchanged corpus performs zero parsing, embedding, upsert, delete, and optimization work.
2. Appending a short new record changes only documents that depend on that record or an open turn-context seam; appending a 512-plus-token message creates stable 512/64 siblings while unrelated prior chunks keep their IDs.
3. Unchanged chunk content reuses its vector after session-name, current-leaf, branch, compaction, visibility, project-attribution, or path metadata changes.
4. A middle-file rewrite, truncation, inode replacement, same-size rewrite, migration, and import overwrite produce the same final documents as a clean rebuild.
5. Multiple logical sessions in one physical file keep disjoint source-occurrence identities.
6. `/psr-import` assigns a fresh session ID, preserves the conversation, and produces Zvec document IDs distinct from the source session.
7. Malformed and incomplete tails retain the last good indexed generation and converge after repair.
8. New, changed, and stale IDs survive a crash after partial upsert, after full upsert but before delete, and after Zvec completion but before checkpoint commit.
9. Every batch status is checked; injected per-document disk failures leave the state checkpoint behind and retry cleanly.
10. Deleted-file behavior matches the chosen mirror/archive policy and uses exact stored IDs.
11. Persistent pre-assistant and ephemeral sessions survive extension shutdown through the durable queue.
12. Two Pi processes emit notices concurrently while one writer owns Zvec and several readers open it read-only.
13. Readers observe writer changes according to the tested reopen/refresh contract for `@zvec/zvec` 0.6.0.
14. Dense vectors match the manifest dimension and metric; incompatible model, tokenizer, 512/64 policy, schema, FTS, or HNSW settings demand rebuild.
15. Dense, lexical, and identifier routes return independently attributable candidates and deterministic fused results.
16. Optimization thresholds use `indexCompleteness` and do not run once per changed file.
17. Incremental output is field-for-field and vector-for-vector equivalent to a clean full rebuild for the same corpus and manifest.

## Consequences for session consumers

A correct external indexer, watcher, backup tool, or parser should account for these behaviors:

1. **Watch directories as well as known files.** A valid persistent session path may not exist until the first assistant response.
2. **Treat ordinary growth as append-only, not immutable.** Opening v1/v2 sessions and some explicit-path operations rewrite whole files.
3. **Identify a logical session by source occurrence plus header occurrence.** Keep header ID and pathname as indexed attributes, not sole keys: imports can copy one header ID to several paths, and historical files can contain several headers.
4. **Do not infer the active branch solely from physical order without naming the implementation.** CLI files use the last physical entry as leaf after reload; harness files can contain explicit `leaf` records.
5. **Validate the graph independently.** Pi's CLI loader accepts malformed gaps, duplicate IDs, missing parents, cycles, and unknown shapes.
6. **Frame records on LF bytes and parse a complete unterminated tail.** Pi itself reads this way.
7. **Expect malformed lines to disappear during migration.** A rewritten v1/v2 file contains only records that Pi parsed successfully.
8. **Support both compaction forms.** CLI compactions use `firstKeptEntryId`; generic harness compactions may also carry `retainedTail`.
9. **Do not use `PI_SESSION_FILE` existence as a save signal.** Check the filesystem.
10. **Do not write concurrently with Pi.** The CLI manager has no coordination protocol for external writers.
11. **Expect rename as an appended `session_info`, not a filesystem rename.** Historical names remain in the log.
12. **Expect deletion to remove the whole file.** There is no tombstone in another store.
13. **Keep Zvec derived and rebuildable.** Preserve raw or canonical evidence outside the collection.
14. **Increment changed files at document granularity.** Under the 512/64 policy, reuse unchanged vectors, upsert metadata changes, embed only changed content, and delete stale chunk IDs.
15. **Use one Zvec writer.** Pi hooks should enqueue dirty sources; a worker, daemon, or explicit locked CLI should own native writes and optimization.

## Evidence ledgers

### CodeGraph evidence

A CodeGraph index of commit `845d6ff1` found `SessionManager` in `packages/coding-agent/src/core/session-manager.ts` with direct importers in `main.ts`, `sdk.ts`, `agent-session.ts`, `agent-session-runtime.ts`, interactive mode, RPC mode, compaction, extensions, and tests. It identified `_appendEntry`, `_persist`, `newSession`, `_setSessionFile`, `createBranchedSession`, `open`, `continueRecent`, and `forkFrom` as the persistence spine. It separately found `JsonlSessionRepo` and `JsonlSessionStorage` under `packages/agent/src/harness/session/`; no coding-agent CLI source imports those classes.

The extension audit identified `ExtensionRunner.emit`, `emitMessageEnd`, `AgentSession._emitExtensionEvent`, `_handleAgentEvent`, `_runAgentPrompt`, and `AgentSessionRuntime.teardownCurrent` as the event-ordering spine. It also showed that extension event names are string-dispatched through handler maps, a CodeGraph blind spot that required exact source search and reads.

The project CodeGraph index contained 32 TypeScript files and identified `incremental-session-indexer.ts`, `session-conversation-index.ts`, `zvec-conversation-store.ts`, and `recall-index-manifest.ts` as the changed-file indexing, chunking, storage, and compatibility spine. `recall-conversation-service.ts` is the owning service and operation-lock caller. The graph cannot establish native Zvec status behavior, filesystem generation semantics, or whether scalar-only changes alter current checksums; those conclusions required source reads and official documentation.

CodeGraph is structural evidence only. It narrowed the source reads but did not prove runtime behavior.

### Source-read interpretation

The cited source establishes:

- selection and lifecycle ownership;
- delayed creation and later append behavior;
- permissive parsing and in-place migration;
- tree and compaction reconstruction;
- non-persisted CLI leaf movement;
- branch extraction, full-history fork, import, rename, and delete behavior;
- absence of locking, atomic rename, and fsync in the session write path;
- the separate generic harness format;
- the complete public extension event set;
- pre- versus post-persistence event ordering;
- append paths with no public extension event;
- session replacement, reload, and shutdown ordering;
- current changed-file skip, chunk-diff, vector-reuse, upsert/delete, checkpoint, and locking behavior;
- the 512-token/64-token-overlap chunk geometry and compatibility manifest;
- the content-only current checksum, accepted built-in-import identity collision, and unobserved Zvec status gaps.

The official Zvec documentation establishes:

- one strongly typed schema per self-contained collection and no cross-collection queries;
- Node create/open/fetch/upsert/delete/query/optimize APIs;
- immediate query availability after successful upsert and per-document batch statuses;
- exact vector-dimension requirements, scalar inverted indexes, FTS/BM25, and vector filtering;
- separate FTS and vector query routes;
- staged vector ingestion, `indexCompleteness`, and background optimization;
- read-only mode as the documented sharing mode across processes.

### Proof commands and observations

The investigation used these checks against primary artifacts:

```bash
npm view @earendil-works/pi-coding-agent@0.83.0 version dist.tarball gitHead time --json
git checkout --detach 845d6ff1f6643aba440341cce877ce1c43ebbc39
codegraph build .
codegraph stats -T
codegraph brief packages/coding-agent/src/core/session-manager.ts -T
codegraph deps packages/coding-agent/src/core/session-manager.ts -T --json
codegraph context SessionManager.branch -T --file packages/coding-agent/src/core/session-manager.ts
codegraph brief packages/coding-agent/src/core/extensions/runner.ts -T
codegraph context ExtensionRunner.emit -T --file packages/coding-agent/src/core/extensions/runner.ts
codegraph context AgentSession._handleAgentEvent -T --file packages/coding-agent/src/core/agent-session.ts
rg -n 'PI_SESSION_FILE|appendFileSync|openSync|fsync|lockfile|JsonlSessionRepo' packages
rg -n 'session_start|session_shutdown|message_end|agent_settled|entry_appended' packages/coding-agent/src/core

# Project Zvec/indexing audit
codegraph stats -T
codegraph brief src/incremental-session-indexer.ts -T
codegraph deps src/incremental-session-indexer.ts -T --json
codegraph brief src/recall-chunk-policy.ts -T
codegraph brief src/zvec-conversation-store.ts -T
codegraph brief src/recall-index-manifest.ts -T
rg -n 'maxTokens|overlapTokens|chunkPolicy|optimize\(' src docs
```

The installed source maps contain the original TypeScript source. SHA-256 comparison found byte-for-byte matches between those embedded sources and commit `845d6ff1` for `main.ts`, `session-manager.ts`, `agent-session.ts`, and `agent-session-runtime.ts`.

A Node probe imported the installed 0.83.0 `dist/core/session-manager.js` and verified:

- no file after `create()`;
- no file after initial metadata and user entries;
- complete file creation after the first assistant entry;
- one-line append afterward;
- branch movement disappearing after reopen when no entry followed it;
- malformed lines being removed by a v2-to-v3 migration rewrite.

The live session used for this research also matched the source shape: header, initial model and reasoning records, extension custom records, user message, finalized assistant message, and tool-result records linked by `parentId`.

A focused project probe ran `readSessionConversationChunks()` with the 512/64 policy against an original canonical session file and an unchanged copy at another path, matching Pi's built-in import behavior. The paths differed but every generated chunk ID matched. The project accepts this built-in `/import` edge case and will provide `/psr-import` with a fresh session identity instead. The same probe appended only a `session_info` rename; each chunk's resolved `sessionName` changed while its ID and content checksum did not, confirming the separate metadata-checksum gap.

The focused project test set passed 38 tests covering canonical framing and migration, chunk geometry, branch/compaction provenance, changed-file Zvec reconciliation, vector reuse, manifest compatibility, Zvec persistence, filtered retrieval, lexical-only evidence, and dimension mismatch rejection.

## Source index

- [CLI session manager](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/session-manager.ts)
- [CLI startup and session selection](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/main.ts)
- [SDK session construction](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/sdk.ts)
- [Agent event persistence, compaction, and tree navigation](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session.ts)
- [Runtime new/resume/fork/import lifecycle](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/agent-session-runtime.ts)
- [Session selector rename/delete behavior](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/modes/interactive/components/session-selector.ts)
- [Official session-format documentation](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/docs/session-format.md)
- [Official extension documentation](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/docs/extensions.md)
- [Extension event and context types](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/extensions/types.ts)
- [Extension event runner](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/coding-agent/src/core/extensions/runner.ts)
- [Low-level awaited agent event loop](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/agent.ts)
- [Zvec official documentation index](https://zvec.org/llms.txt)
- [Zvec Node quickstart](https://zvec.org/en/docs/db/quickstart/)
- [Zvec data model](https://zvec.org/en/docs/db/concepts/data-modeling/)
- [Zvec collection schema](https://zvec.org/en/docs/db/collections/create/schema/)
- [Zvec upsert semantics](https://zvec.org/en/docs/db/data-operations/upsert/)
- [Zvec collection optimization](https://zvec.org/en/docs/db/collections/optimize/)
- [Zvec full-text index](https://zvec.org/en/docs/db/concepts/fts-index/)
- [Project incremental session indexer](../../src/incremental-session-indexer.ts)
- [Project session chunker](../../src/session-conversation-index.ts)
- [Project Zvec conversation store](../../src/zvec-conversation-store.ts)
- [Project recall index manifest](../../src/recall-index-manifest.ts)
- [Generic harness repository](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/harness/session/jsonl-repo.ts)
- [Generic harness JSONL storage](https://github.com/earendil-works/pi/blob/845d6ff1f6643aba440341cce877ce1c43ebbc39/packages/agent/src/harness/session/jsonl-storage.ts)
