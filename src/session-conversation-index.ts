import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

/** A Pi session ID that cannot be passed where a session entry ID is required. */
export interface PiSessionId {
  readonly value: string;
}

/** A Pi session entry ID that cannot be passed where a session ID is required. */
export interface PiSessionEntryId {
  readonly value: string;
}

/** A searchable conversation segment with exact provenance back to one Pi session entry. */
export interface SessionConversationChunk {
  id: string;
  checksum: string;
  sessionId: PiSessionId;
  sessionPath: string;
  cwd: string;
  sessionName: string;
  entryId: PiSessionEntryId;
  role: 'user' | 'assistant' | 'summary' | 'custom';
  timestamp: string;
  chunkIndex: number;
  content: string;
}

/** Character limits for embedding model input; overlap must stay smaller than the chunk limit. */
export interface SessionConversationChunkOptions {
  maxCharacters?: number;
  overlapCharacters?: number;
}

function isUnknownRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function extractTextContent(content: unknown): string {
  if (typeof content === 'string') {
    return content.trim();
  }
  if (!Array.isArray(content)) {
    return '';
  }
  return content
    .filter((block): block is { type: 'text'; text: string } => {
      if (!isUnknownRecord(block)) {
        return false;
      }
      return block.type === 'text' && typeof block.text === 'string';
    })
    .map((block) => block.text)
    .join('\n')
    .trim();
}

function splitConversationText(
  text: string,
  maxCharacters: number,
  overlapCharacters: number,
): string[] {
  if (text.length <= maxCharacters) {
    return text ? [text] : [];
  }
  const chunks: string[] = [];
  const step = maxCharacters - overlapCharacters;
  for (let start = 0; start < text.length; start += step) {
    const end = Math.min(start + maxCharacters, text.length);
    let boundary = end;
    if (end < text.length) {
      const lastNewline = text.lastIndexOf('\n', end);
      const lastSpace = text.lastIndexOf(' ', end);
      const candidate = Math.max(lastNewline, lastSpace);
      if (candidate > start + Math.floor(maxCharacters / 2)) {
        boundary = candidate;
      }
    }
    const chunk = text.slice(start, boundary).trim();
    if (chunk) {
      chunks.push(chunk);
    }
    if (boundary >= text.length) {
      break;
    }
    start = boundary - step;
  }
  return chunks;
}

function hashConversationValue(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

/** Reads user-visible conversation text from a Pi JSONL session while excluding tools and reasoning. */
export async function readSessionConversationChunks(
  sessionPath: string,
  options: SessionConversationChunkOptions = {},
): Promise<SessionConversationChunk[]> {
  const maxCharacters = options.maxCharacters ?? 6_000;
  const overlapCharacters = options.overlapCharacters ?? 400;
  if (maxCharacters < 1 || overlapCharacters < 0 || overlapCharacters >= maxCharacters) {
    throw new Error(
      'Recall chunk settings invalid: overlapCharacters must be smaller than maxCharacters',
    );
  }

  let sessionIdValue = '';
  let cwd = '';
  let sessionName = '';
  const pending: Array<{
    entryId: string;
    role: SessionConversationChunk['role'];
    timestamp: string;
    content: string;
  }> = [];
  const lines = createInterface({
    input: createReadStream(sessionPath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });

  for await (const line of lines) {
    if (!line.trim()) {
      continue;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      continue;
    }
    if (!isUnknownRecord(parsed)) {
      continue;
    }
    const entry = parsed;
    if (entry.type === 'session') {
      if (typeof entry.id === 'string') {
        sessionIdValue = entry.id;
      }
      if (typeof entry.cwd === 'string') {
        cwd = entry.cwd;
      }
      continue;
    }
    if (entry.type === 'session_info' && typeof entry.name === 'string') {
      sessionName = entry.name;
      continue;
    }

    const entryId = typeof entry.id === 'string' ? entry.id : '';
    const timestamp = typeof entry.timestamp === 'string' ? entry.timestamp : '';
    if (!entryId) {
      continue;
    }

    if (entry.type === 'message' && isUnknownRecord(entry.message)) {
      const message = entry.message;
      if (message.role !== 'user' && message.role !== 'assistant') {
        continue;
      }
      const content = extractTextContent(message.content);
      if (content) {
        pending.push({ entryId, role: message.role, timestamp, content });
      }
      continue;
    }
    if (
      (entry.type === 'compaction' || entry.type === 'branch_summary') &&
      typeof entry.summary === 'string'
    ) {
      const content = entry.summary.trim();
      if (content) {
        pending.push({ entryId, role: 'summary', timestamp, content });
      }
      continue;
    }
    if (entry.type === 'custom_message') {
      const content = extractTextContent(entry.content);
      if (content) {
        pending.push({ entryId, role: 'custom', timestamp, content });
      }
    }
  }

  if (!sessionIdValue) {
    throw new Error(`Recall session header missing id: ${sessionPath}`);
  }
  const sessionId: PiSessionId = { value: sessionIdValue };
  return pending.flatMap((item) =>
    splitConversationText(item.content, maxCharacters, overlapCharacters).map(
      (content, chunkIndex) => {
        const id = hashConversationValue(`${sessionPath}\0${item.entryId}\0${chunkIndex}`).slice(
          0,
          40,
        );
        return {
          id,
          checksum: hashConversationValue(content),
          sessionId,
          sessionPath,
          cwd,
          sessionName,
          entryId: { value: item.entryId },
          role: item.role,
          timestamp: item.timestamp,
          chunkIndex,
          content,
        };
      },
    ),
  );
}
