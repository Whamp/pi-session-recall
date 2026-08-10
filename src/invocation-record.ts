import { isUnknownRecord } from './is-unknown-record.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';

const INVOCATION_SCALAR_CHARACTER_LIMIT = 1_024;
const INVOCATION_SEARCHABLE_CHARACTER_LIMIT = 4_096;
const INVOCATION_OMISSION_MARKER = '<omitted>';
const OMITTED_PAYLOAD_KEY_SUFFIXES = [
  'body',
  'content',
  'filebody',
  'filecontent',
  'newtext',
  'oldtext',
  'prompt',
  'replacement',
  'replacementtext',
  'script',
] as const;
const OMITTED_PAYLOAD_KEYS = new Set([
  'code',
  'data',
  'image',
  'images',
  'instructions',
  'output',
  'prompts',
  'scripts',
  'stderr',
  'stdout',
  'text',
]);

/** One compact, source-backed tool call or direct bash execution prepared for invocation search. */
export interface InvocationRecord {
  kind: 'tool_call' | 'bash_execution';
  toolName: string;
  toolCallId: string | null;
  sessionPath: string;
  sessionId: string;
  entryId: string;
  sourceLineStart: number;
  sourceLineEnd: number;
  sourceBlockIndex: number | null;
  timestamp: string;
  sessionOrigin: string;
  projectAttribution: ResolvedProjectIdentity | null;
  isError: boolean | null;
  searchableText: string;
}

/** Minimal validated session-entry shape consumed by Invocation record projection. */
export interface InvocationSourceEntry {
  id: string;
  timestamp: string;
  lineIndex: number;
  record: Record<string, unknown>;
}

/** Physical and project provenance shared by Invocation records from one logical session. */
export interface SessionInvocationContext {
  sessionPath: string;
  sessionId: string;
  sessionOrigin: string;
  projectAttribution: ResolvedProjectIdentity | null;
}

function sliceUnicodeCharacters(value: string, limit: number): string {
  return Array.from(value).slice(0, limit).join('');
}

function isOmittedInvocationPayloadKey(key: string): boolean {
  const normalized = key.replaceAll(/[^a-z0-9]/giu, '').toLowerCase();
  if (normalized === 'querytext') {
    return false;
  }
  return (
    OMITTED_PAYLOAD_KEYS.has(normalized) ||
    OMITTED_PAYLOAD_KEY_SUFFIXES.some((suffix) => normalized.endsWith(suffix))
  );
}

function appendInvocationSearchLine(lines: string[], line: string): boolean {
  const separatorLength = lines.length === 0 ? 0 : 1;
  const usedCharacters = Array.from(lines.join('\n')).length;
  const lineCharacters = Array.from(line).length;
  if (usedCharacters + separatorLength + lineCharacters > INVOCATION_SEARCHABLE_CHARACTER_LIMIT) {
    return false;
  }
  lines.push(line);
  return true;
}

function renderInvocationScalar(value: string | number | boolean | null): string {
  if (typeof value === 'string') {
    return JSON.stringify(sliceUnicodeCharacters(value, INVOCATION_SCALAR_CHARACTER_LIMIT));
  }
  return JSON.stringify(value);
}

function projectInvocationArgument(
  value: unknown,
  path: string,
  key: string,
  lines: string[],
): void {
  if (isOmittedInvocationPayloadKey(key)) {
    appendInvocationSearchLine(lines, `${path}=${INVOCATION_OMISSION_MARKER}`);
    return;
  }
  if (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    appendInvocationSearchLine(lines, `${path}=${renderInvocationScalar(value)}`);
    return;
  }
  if (Array.isArray(value)) {
    for (const [index, nestedValue] of value.entries()) {
      projectInvocationArgument(nestedValue, `${path}[${index}]`, key, lines);
    }
    return;
  }
  if (!isUnknownRecord(value)) {
    return;
  }
  for (const nestedKey of Object.keys(value).toSorted()) {
    const nestedPath = path ? `${path}.${nestedKey}` : nestedKey;
    projectInvocationArgument(value[nestedKey], nestedPath, nestedKey, lines);
  }
}

function projectInvocationSearchableText(toolName: string, argumentsValue: unknown): string {
  const lines = [
    `tool=${JSON.stringify(sliceUnicodeCharacters(toolName, INVOCATION_SCALAR_CHARACTER_LIMIT))}`,
  ];
  if (isUnknownRecord(argumentsValue)) {
    for (const key of Object.keys(argumentsValue).toSorted()) {
      projectInvocationArgument(argumentsValue[key], key, key, lines);
    }
  } else if (Array.isArray(argumentsValue)) {
    for (const [index, value] of argumentsValue.entries()) {
      projectInvocationArgument(value, `[${index}]`, String(index), lines);
    }
  }
  return lines.join('\n');
}

function readToolResultErrors(
  entries: readonly InvocationSourceEntry[],
): Map<string, boolean | null> {
  const errors = new Map<string, boolean | null>();
  for (const entry of entries) {
    const message = isUnknownRecord(entry.record.message) ? entry.record.message : null;
    if (message?.role !== 'toolResult' || typeof message.toolCallId !== 'string') {
      continue;
    }
    errors.set(message.toolCallId, typeof message.isError === 'boolean' ? message.isError : null);
  }
  return errors;
}

function readBashExecutionError(message: Record<string, unknown>): boolean | null {
  if (message.cancelled === true) {
    return true;
  }
  return typeof message.exitCode === 'number' ? message.exitCode !== 0 : null;
}

function createDirectBashInvocationRecord(
  entry: InvocationSourceEntry,
  message: Record<string, unknown>,
  context: SessionInvocationContext,
): InvocationRecord[] {
  if (typeof message.command !== 'string') {
    return [];
  }
  return [
    {
      kind: 'bash_execution',
      toolName: 'bash',
      toolCallId: null,
      sessionPath: context.sessionPath,
      sessionId: context.sessionId,
      entryId: entry.id,
      sourceLineStart: entry.lineIndex,
      sourceLineEnd: entry.lineIndex,
      sourceBlockIndex: null,
      timestamp: entry.timestamp,
      sessionOrigin: context.sessionOrigin,
      projectAttribution: context.projectAttribution,
      isError: readBashExecutionError(message),
      searchableText: projectInvocationSearchableText('bash', { command: message.command }),
    },
  ];
}

/** Creates deterministic, bounded Invocation records without copying result or payload content. */
export function createSessionInvocationRecords(
  entries: readonly InvocationSourceEntry[],
  context: SessionInvocationContext,
): InvocationRecord[] {
  const toolResultErrors = readToolResultErrors(entries);
  const invocations: InvocationRecord[] = [];
  for (const entry of entries) {
    const message = isUnknownRecord(entry.record.message) ? entry.record.message : null;
    if (!message) {
      continue;
    }
    if (message.role === 'bashExecution') {
      invocations.push(...createDirectBashInvocationRecord(entry, message, context));
      continue;
    }
    if (message.role !== 'assistant' || !Array.isArray(message.content)) {
      continue;
    }
    for (const [blockIndex, block] of message.content.entries()) {
      if (
        !isUnknownRecord(block) ||
        block.type !== 'toolCall' ||
        typeof block.id !== 'string' ||
        !block.id ||
        typeof block.name !== 'string' ||
        !block.name ||
        block.name === 'pi-session-recall'
      ) {
        continue;
      }
      invocations.push({
        kind: 'tool_call',
        toolName: sliceUnicodeCharacters(block.name, INVOCATION_SCALAR_CHARACTER_LIMIT),
        toolCallId: block.id,
        sessionPath: context.sessionPath,
        sessionId: context.sessionId,
        entryId: entry.id,
        sourceLineStart: entry.lineIndex,
        sourceLineEnd: entry.lineIndex,
        sourceBlockIndex: blockIndex,
        timestamp: entry.timestamp,
        sessionOrigin: context.sessionOrigin,
        projectAttribution: context.projectAttribution,
        isError: toolResultErrors.get(block.id) ?? null,
        searchableText: projectInvocationSearchableText(block.name, block.arguments),
      });
    }
  }
  return invocations;
}
