import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

import { RecallSearchScope } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { listRecallSessionFiles } from './recall-session-files.js';
import type { ProjectIdentity, ResolvedProjectIdentity } from './resolve-project-identity.js';

const SOURCE_SEARCH_EXCERPT_CHARACTER_LIMIT = 2_000;

/** One exact match read directly from a canonical physical session file. */
export interface SessionSourceSearchResult {
  sessionPath: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  entryId: string | null;
  sessionOrigin: string;
  text: string;
}

/** One physical session that could not be read while other source files remained searchable. */
export interface SessionSourceSearchFailure {
  sessionPath: string;
  error: string;
}

/** Read-only source-scan results and failures from the eligible physical session corpus. */
export interface SessionSourceSearch {
  results: SessionSourceSearchResult[];
  failures: SessionSourceSearchFailure[];
  filesScanned: number;
  scope: RecallSearchScope;
  invocationProjectIdentity: ProjectIdentity | null;
}

/** Exact source-scan inputs after trusted invocation scope has been resolved. */
export interface SessionSourceSearchOptions {
  sessionsDirectory: string;
  ignoredPhysicalSessionPaths: ReadonlySet<string>;
  query: string;
  limit: number;
  scope: RecallSearchScope;
  invocationProjectIdentity: ProjectIdentity | null;
  resolveProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>;
  signal?: AbortSignal;
}

/** Injectable file discovery used to certify missing-file behavior without persistent state. */
export interface SessionSourceSearchDependencies {
  listSessionPaths?: (sessionsDirectory: string) => Promise<string[]>;
}

function parseSourceRecord(line: string): Record<string, unknown> | null {
  const parsed: unknown = JSON.parse(line);
  return isUnknownRecord(parsed) ? parsed : null;
}

function readSourceEntryId(record: Record<string, unknown> | null): string | null {
  return record && typeof record.id === 'string' ? record.id : null;
}

function formatSourceSearchExcerpt(line: string, lowerQuery: string): string {
  if (line.length <= SOURCE_SEARCH_EXCERPT_CHARACTER_LIMIT) {
    return line;
  }
  const matchIndex = line.toLowerCase().indexOf(lowerQuery);
  const availableContext = SOURCE_SEARCH_EXCERPT_CHARACTER_LIMIT - 2;
  const excerptStart = Math.max(0, matchIndex - Math.floor(availableContext / 2));
  const excerptEnd = Math.min(line.length, excerptStart + availableContext);
  const prefix = excerptStart > 0 ? '…' : '';
  const suffix = excerptEnd < line.length ? '…' : '';
  return `${prefix}${line.slice(excerptStart, excerptEnd)}${suffix}`;
}

function formatSourceReadFailure(sessionPath: string, error: unknown): SessionSourceSearchFailure {
  if (readNodeErrorCode(error) === 'ENOENT') {
    return {
      sessionPath,
      error: `Source session missing at ${sessionPath}`,
    };
  }
  const message = error instanceof Error ? error.message : String(error);
  return {
    sessionPath,
    error: `Source session unreadable at ${sessionPath}: ${message}`,
  };
}

async function readSessionSourceMatches(
  sessionPath: string,
  lowerQuery: string,
  remainingLimit: number,
  options: SessionSourceSearchOptions,
): Promise<SessionSourceSearchResult[]> {
  const results: SessionSourceSearchResult[] = [];
  const lines = createInterface({
    input: createReadStream(sessionPath),
    crlfDelay: Infinity,
  });
  let sourceLine = 0;
  let sessionOrigin = '';
  let sessionProjectIdentity: ProjectIdentity | null = null;

  for await (const line of lines) {
    options.signal?.throwIfAborted();
    sourceLine += 1;
    const isMatch = line.toLowerCase().includes(lowerQuery);
    const mightBeSessionHeader = line.includes('"session"');
    const record = isMatch || mightBeSessionHeader ? parseSourceRecord(line) : null;

    if (record?.type === 'session') {
      sessionOrigin = typeof record.cwd === 'string' ? record.cwd : '';
      sessionProjectIdentity = null;
      if (options.scope === RecallSearchScope.PROJECT && sessionOrigin) {
        const project = await options.resolveProjectIdentity(sessionOrigin);
        options.signal?.throwIfAborted();
        sessionProjectIdentity = project?.projectIdentity ?? null;
      }
    }

    if (!isMatch) {
      continue;
    }
    const isWithinScope =
      options.scope === RecallSearchScope.GLOBAL ||
      (options.invocationProjectIdentity !== null &&
        sessionProjectIdentity === options.invocationProjectIdentity);
    if (!isWithinScope || results.length >= remainingLimit) {
      continue;
    }
    results.push({
      sessionPath,
      sourceLineStart: sourceLine,
      sourceLineEnd: sourceLine,
      entryId: readSourceEntryId(record),
      sessionOrigin,
      text: formatSourceSearchExcerpt(line, lowerQuery),
    });
  }

  return results;
}

/** Formats exact raw-output matches with physical path, line range, and entry identity. */
export function formatSessionSourceSearchResults(search: SessionSourceSearch): string {
  const scopeDescription =
    search.scope === RecallSearchScope.PROJECT
      ? `project scope for ${search.invocationProjectIdentity ?? 'an unresolved project'}`
      : 'global scope';
  const lines = [
    `Source search scanned ${search.filesScanned.toLocaleString('en-US')} eligible physical session files in ${scopeDescription}.`,
  ];
  if (search.results.length === 0) {
    for (const failure of search.failures) {
      lines.push(`Source warning: ${failure.error}`);
    }
    lines.push('No matching source-backed evidence found.');
    if (search.scope === RecallSearchScope.PROJECT) {
      lines.push(
        'Retry with scope "global" to scan every eligible session; project scope was not broadened automatically.',
      );
    }
    return lines.join('\n');
  }
  for (const [index, result] of search.results.entries()) {
    const entryIdentity = result.entryId ? `#${result.entryId}` : '';
    lines.push(
      '',
      `${index + 1}. ${result.sessionPath}:${result.sourceLineStart}-${result.sourceLineEnd}${entryIdentity}`,
      `session origin ${result.sessionOrigin || 'unknown'}`,
      result.text,
    );
  }
  for (const failure of search.failures) {
    lines.push(`Source warning: ${failure.error}`);
  }
  return lines.join('\n');
}

/** Scans eligible session JSONL on demand without writing an index, cache, or source file. */
export async function searchSessionSourceFiles(
  options: SessionSourceSearchOptions,
  dependencies: SessionSourceSearchDependencies = {},
): Promise<SessionSourceSearch> {
  const query = options.query.trim();
  if (!query) {
    throw new Error('Source search query must not be blank');
  }
  if (!Number.isInteger(options.limit) || options.limit < 1) {
    throw new Error(`Source search limit must be a positive integer, received ${options.limit}`);
  }
  if (options.scope === RecallSearchScope.PROJECT && options.invocationProjectIdentity === null) {
    throw new Error('Project-scoped source search requires a resolved invocation project identity');
  }

  options.signal?.throwIfAborted();
  const listSessionPaths = dependencies.listSessionPaths ?? listRecallSessionFiles;
  const sessionPaths = (await listSessionPaths(options.sessionsDirectory))
    .filter((sessionPath) => !options.ignoredPhysicalSessionPaths.has(sessionPath))
    .toSorted();
  const lowerQuery = query.toLowerCase();
  const projectIdentityResolutions = new Map<string, Promise<ResolvedProjectIdentity | null>>();
  const resolveCachedProjectIdentity = (
    sessionOrigin: string,
  ): Promise<ResolvedProjectIdentity | null> => {
    const existingResolution = projectIdentityResolutions.get(sessionOrigin);
    if (existingResolution) {
      return existingResolution;
    }
    const resolution = options.resolveProjectIdentity(sessionOrigin);
    projectIdentityResolutions.set(sessionOrigin, resolution);
    return resolution;
  };
  const searchOptions: SessionSourceSearchOptions = {
    ...options,
    resolveProjectIdentity: resolveCachedProjectIdentity,
  };
  const results: SessionSourceSearchResult[] = [];
  const failures: SessionSourceSearchFailure[] = [];
  let filesScanned = 0;

  for (const sessionPath of sessionPaths) {
    options.signal?.throwIfAborted();
    try {
      const matches = await readSessionSourceMatches(
        sessionPath,
        lowerQuery,
        options.limit - results.length,
        searchOptions,
      );
      filesScanned += 1;
      results.push(...matches);
    } catch (error) {
      if (options.signal?.aborted) {
        options.signal.throwIfAborted();
      }
      failures.push(formatSourceReadFailure(sessionPath, error));
    }
  }

  return {
    results,
    failures,
    filesScanned,
    scope: options.scope,
    invocationProjectIdentity: options.invocationProjectIdentity,
  };
}
