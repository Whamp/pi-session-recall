import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import test from 'node:test';
import { stripVTControlCharacters } from 'node:util';

import { initTheme } from '@earendil-works/pi-coding-agent';

import { RecallEvidenceRelation, RecallSearchScope } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import recallExtension, {
  createPiRecallToolDefinition,
  createPiRecallToolDetails,
  type PiRecallParameters,
  searchPiRecall,
} from './recall-extension.js';
import type { RecallConversationService } from './recall-conversation-service.js';
import {
  createTestRankedRecallSearchResult,
  createTestRecallSearchResult,
  createTestSessionConversationChunk,
} from './recall-test-utils.js';

interface PiTuiKeybindingDefinition {
  defaultKeys: string;
  description: string;
}

interface PiTuiKeybindingRegistryModule {
  getKeybindings(): unknown;
  setKeybindings(keybindings: unknown): void;
  KeybindingsManager: new (
    definitions: Record<string, PiTuiKeybindingDefinition>,
    userBindings: Record<string, string>,
  ) => unknown;
}

function isPiTuiKeybindingRegistryModule(value: unknown): value is PiTuiKeybindingRegistryModule {
  return (
    isUnknownRecord(value) &&
    typeof value.getKeybindings === 'function' &&
    typeof value.setKeybindings === 'function' &&
    typeof value.KeybindingsManager === 'function'
  );
}

async function usePiToolExpansionKeybinding(key: string): Promise<() => void> {
  const requireFromPi = createRequire(import.meta.resolve('@earendil-works/pi-coding-agent'));
  const piTuiEntry = requireFromPi.resolve('@earendil-works/pi-tui');
  const piTuiModule: unknown = await import(pathToFileURL(piTuiEntry).href);
  if (!isPiTuiKeybindingRegistryModule(piTuiModule)) {
    throw new Error('Pi TUI keybinding registry is unavailable');
  }

  const previousKeybindings = piTuiModule.getKeybindings();
  const configuredKeybindings = new piTuiModule.KeybindingsManager(
    {
      'app.tools.expand': {
        defaultKeys: 'ctrl+o',
        description: 'Toggle tool output',
      },
    },
    { 'app.tools.expand': key },
  );
  piTuiModule.setKeybindings(configuredKeybindings);
  return () => {
    piTuiModule.setKeybindings(previousKeybindings);
  };
}

function createEmptySearch(scope: RecallSearchScope, invocationProjectIdentity: null = null) {
  return {
    results: [],
    totalChunks: 0,
    indexMaintenanceStatus: null,
    searchPolicy: {
      scope,
      invocationProjectIdentity,
      rankingMode: 'hybrid' as const,
      rankFusionVersion: 2,
      reciprocalRankConstant: 60,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    },
  };
}

void test('Pi recall applies trusted cwd and project-default scope without a write path', async () => {
  const calls: unknown[] = [];
  const service = {
    async search(query, limit, options) {
      calls.push({ query, limit, options });
      return createEmptySearch(options?.scope ?? RecallSearchScope.PROJECT);
    },
    async index() {
      throw new Error('search adapter must not index');
    },
  } satisfies RecallConversationService;

  await searchPiRecall(service, { query: '  queue decision  ' }, { cwd: '/trusted/project' });
  await searchPiRecall(
    service,
    { query: 'global decision', limit: 3, scope: 'global' },
    { cwd: '/trusted/project' },
  );

  assert.deepEqual(calls, [
    {
      query: 'queue decision',
      limit: 5,
      options: {
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/trusted/project',
      },
    },
    {
      query: 'global decision',
      limit: 3,
      options: {
        scope: RecallSearchScope.GLOBAL,
        invocationDirectory: '/trusted/project',
      },
    },
  ]);
});

void test('Pi recall call shows a concise exact query without changing retrieval input', async () => {
  const calls: unknown[] = [];
  const service = {
    async search(query, limit, options) {
      calls.push({ query, limit, options });
      return createEmptySearch(options?.scope ?? RecallSearchScope.PROJECT);
    },
    async index() {
      throw new Error('search adapter must not index');
    },
  } satisfies RecallConversationService;
  const tool = createPiRecallToolDefinition(service);
  const parameters: PiRecallParameters = {
    query:
      '  decision\tto\nschedule   automatic psr index maintenance after every completed conversation while preserving exact retrieval whitespace  ',
    limit: 3,
    scope: 'global',
  };
  const originalParameters = structuredClone(parameters);

  const component = tool.renderCall(
    parameters,
    {
      bold(text) {
        return text;
      },
      fg(color, text) {
        void color;
        return text;
      },
    },
    { isError: false, lastComponent: undefined },
  );
  const rendered = stripVTControlCharacters(component.render(1_000).join('\n')).trimEnd();

  assert.equal(
    rendered,
    'pi-session-recall “decision to schedule automatic psr index maintenance after …”',
  );

  await tool.execute('call-presentation', parameters, undefined, undefined, {
    cwd: '/trusted/project',
  });

  assert.deepEqual(parameters, originalParameters);
  assert.deepEqual(calls, [
    {
      query:
        'decision\tto\nschedule   automatic psr index maintenance after every completed conversation while preserving exact retrieval whitespace',
      limit: 3,
      options: {
        scope: RecallSearchScope.GLOBAL,
        invocationDirectory: '/trusted/project',
      },
    },
  ]);
});

void test('service-injected Pi recall tool definition executes complete recall evidence', async () => {
  const calls: unknown[] = [];
  const result = createTestRankedRecallSearchResult({
    id: 'factory-result',
    content: 'The queue uses source-backed recall evidence.',
    sessionPath: '/sessions/factory.jsonl',
    entryId: { value: 'factory-entry' },
    sourceLineStart: 12,
    sourceLineEnd: 14,
  });
  const search = {
    results: [result],
    totalChunks: 42_318,
    indexMaintenanceStatus: null,
    searchPolicy: createEmptySearch(RecallSearchScope.PROJECT).searchPolicy,
  };
  const service = {
    async search(query, limit, options) {
      calls.push({ query, limit, options });
      return search;
    },
    async index() {
      throw new Error('search adapter must not index');
    },
  } satisfies RecallConversationService;
  const tool = createPiRecallToolDefinition(service);

  const execution = await tool.execute(
    'recall-call',
    { query: '  queue decision  ' },
    undefined,
    undefined,
    { cwd: '/trusted/project' },
  );

  assert.equal(tool.name, 'pi-session-recall');
  assert.deepEqual(calls, [
    {
      query: 'queue decision',
      limit: 5,
      options: {
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/trusted/project',
      },
    },
  ]);
  assert.equal(execution.content[0]?.type, 'text');
  assert.match(execution.content[0]?.type === 'text' ? execution.content[0].text : '', /42318/);
  assert.match(
    execution.content[0]?.type === 'text' ? execution.content[0].text : '',
    /The queue uses source-backed recall evidence\./,
  );
  assert.deepEqual(execution.details?.sources[0], {
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
    sessionOrigin: '/project',
    projectIdentity: null,
    projectIdentitySource: null,
    sessionPath: '/sessions/factory.jsonl',
    entryId: 'factory-entry',
    contributingEntryIds: ['entry-factory-result'],
    sourceLineStart: 12,
    sourceLineEnd: 14,
    sourceBlockStart: 0,
    sourceBlockEnd: 0,
    characterStart: 0,
    characterEnd: 45,
    isOnActiveBranch: true,
    rankingScore: 0.02,
    activeBranchPrior: 0,
    fusedScore: 0.02,
    dense: { rank: 1, cosineDistance: 0.1 },
    lexical: null,
    identifier: null,
    duplicateOccurrences: [],
    expandedChunks: [],
  });
});

void test('untruncated Pi recall results show exact UTF-8 payload metrics and expansion hint', async () => {
  initTheme('dark');
  const restoreKeybindings = await usePiToolExpansionKeybinding('alt+x');

  try {
    const firstResult = createTestRankedRecallSearchResult({
      id: 'collapsed-first',
      content: 'Café résumé 🙂',
    });
    const secondResult = createTestRankedRecallSearchResult({
      id: 'collapsed-second',
      content: 'Second evidence line.',
    });
    const service = {
      async search() {
        return {
          results: [firstResult, secondResult],
          totalChunks: 42_318,
          indexMaintenanceStatus: {
            version: 1,
            completedAt: '2026-07-25T12:00:00.000Z',
            scannedSessions: 123,
            failedSessions: 1,
          },
          searchPolicy: createEmptySearch(RecallSearchScope.PROJECT).searchPolicy,
        };
      },
      async index() {
        throw new Error('search adapter must not index');
      },
    } satisfies RecallConversationService;
    const tool = createPiRecallToolDefinition(service, () => new Date('2026-07-25T12:30:59.999Z'));
    const execution = await tool.execute(
      'collapsed-call',
      { query: 'queue decision' },
      undefined,
      undefined,
      { cwd: '/trusted/project' },
    );

    const component = tool.renderResult(
      execution,
      { expanded: false, isPartial: false },
      {
        fg(color, text) {
          void color;
          return text;
        },
      },
      { isError: false, lastComponent: undefined },
    );
    const rendered = stripVTControlCharacters(component.render(1_000).join('\n')).trimEnd();

    assert.equal(execution.details.returnedBytes, 850);
    assert.equal(execution.details.returnedLines, 11);
    assert.deepEqual(execution.details.indexMaintenanceStatus, {
      completedAt: '2026-07-25T12:00:00.000Z',
      scannedSessions: 123,
      failedSessions: 1,
      ageMinutesAtExecution: 30,
    });
    assert.equal('truncation' in execution.details, false);
    assert.equal(
      rendered,
      '2 recall results · project scope · 850B / 11 lines · index checked 30m ago · 1 failed session (alt+x to expand)',
    );
  } finally {
    restoreKeybindings();
  }
});

void test('Pi recall freshness uses fixed execution-time minute, hour, and day ages', async () => {
  initTheme('dark');
  const restoreKeybindings = await usePiToolExpansionKeybinding('alt+x');

  try {
    const result = createTestRankedRecallSearchResult({ id: 'freshness-boundaries' });
    const cases = [
      {
        completedAt: '2026-07-25T12:05:00.000Z',
        currentTime: '2026-07-25T12:00:00.000Z',
        failedSessions: 0,
        expectedFreshness: 'index checked 0m ago',
      },
      {
        completedAt: '2026-07-25T10:01:00.000Z',
        currentTime: '2026-07-25T12:00:59.999Z',
        failedSessions: 0,
        expectedFreshness: 'index checked 1h ago',
      },
      {
        completedAt: '2026-07-23T11:59:00.000Z',
        currentTime: '2026-07-25T12:00:00.000Z',
        failedSessions: 2,
        expectedFreshness: 'index checked 2d ago · 2 failed sessions',
      },
    ];

    for (const freshnessCase of cases) {
      let currentTime = new Date(freshnessCase.currentTime);
      let clockCalls = 0;
      const service = {
        async search() {
          return {
            results: [result],
            totalChunks: 1,
            indexMaintenanceStatus: {
              version: 1 as const,
              completedAt: freshnessCase.completedAt,
              scannedSessions: 7,
              failedSessions: freshnessCase.failedSessions,
            },
            searchPolicy: createEmptySearch(RecallSearchScope.PROJECT).searchPolicy,
          };
        },
        async index() {
          throw new Error('search adapter must not index');
        },
      } satisfies RecallConversationService;
      const tool = createPiRecallToolDefinition(service, () => {
        clockCalls += 1;
        return currentTime;
      });
      const execution = await tool.execute(
        'freshness-boundary-call',
        { query: 'freshness boundary' },
        undefined,
        undefined,
        { cwd: '/trusted/project' },
      );
      currentTime = new Date('2026-08-25T12:00:00.000Z');

      for (let renderCount = 0; renderCount < 2; renderCount += 1) {
        const component = tool.renderResult(
          execution,
          { expanded: false, isPartial: false },
          {
            fg(color, text) {
              void color;
              return text;
            },
          },
          { isError: false, lastComponent: undefined },
        );
        const rendered = stripVTControlCharacters(component.render(1_000).join('\n')).trimEnd();
        assert.match(rendered, new RegExp(freshnessCase.expectedFreshness, 'u'));
      }
      assert.equal(clockCalls, 1);

      const expanded = tool.renderResult(
        execution,
        { expanded: true, isPartial: false },
        {
          fg(color, text) {
            void color;
            return text;
          },
        },
        { isError: false, lastComponent: undefined },
      );
      assert.equal(
        expanded
          .render(2_000)
          .map((line) => line.trimEnd())
          .join('\n'),
        execution.content[0]?.type === 'text' ? execution.content[0].text : '',
      );
    }
  } finally {
    restoreKeybindings();
  }
});

void test('collapsed Pi recall result uses singular wording', async () => {
  initTheme('dark');
  const restoreKeybindings = await usePiToolExpansionKeybinding('alt+x');

  try {
    const result = createTestRankedRecallSearchResult({ id: 'collapsed-singular' });
    const service = {
      async search() {
        return {
          results: [result],
          totalChunks: 9,
          indexMaintenanceStatus: null,
          searchPolicy: createEmptySearch(RecallSearchScope.GLOBAL).searchPolicy,
        };
      },
      async index() {
        throw new Error('search adapter must not index');
      },
    } satisfies RecallConversationService;
    const tool = createPiRecallToolDefinition(service);
    const execution = await tool.execute(
      'singular-call',
      { query: 'one decision', scope: 'global' },
      undefined,
      undefined,
      { cwd: '/trusted/project' },
    );

    const component = tool.renderResult(
      execution,
      { expanded: false, isPartial: false },
      {
        fg(color, text) {
          void color;
          return text;
        },
      },
      { isError: false, lastComponent: undefined },
    );
    const rendered = stripVTControlCharacters(component.render(1_000).join('\n')).trimEnd();

    assert.equal(rendered, '1 recall result · global scope · 487B / 6 lines (alt+x to expand)');

    const oneLineComponent = tool.renderResult(
      {
        ...execution,
        details: {
          ...execution.details,
          returnedBytes: 1,
          returnedLines: 1,
        },
      },
      { expanded: false, isPartial: false },
      {
        fg(color, text) {
          void color;
          return text;
        },
      },
      { isError: false, lastComponent: undefined },
    );
    const oneLineRendered = stripVTControlCharacters(
      oneLineComponent.render(1_000).join('\n'),
    ).trimEnd();
    assert.equal(oneLineRendered, '1 recall result · global scope · 1B / 1 line (alt+x to expand)');

    const {
      returnedBytes: persistedReturnedBytes,
      returnedLines: persistedReturnedLines,
      ...persistedDetails
    } = execution.details;
    void persistedReturnedBytes;
    void persistedReturnedLines;
    const persistedComponent = tool.renderResult(
      { ...execution, details: persistedDetails },
      { expanded: false, isPartial: false },
      {
        fg(color, text) {
          void color;
          return text;
        },
      },
      { isError: false, lastComponent: undefined },
    );
    const persistedRendered = stripVTControlCharacters(
      persistedComponent.render(1_000).join('\n'),
    ).trimEnd();
    assert.equal(persistedRendered, '1 recall result · global scope (alt+x to expand)');
  } finally {
    restoreKeybindings();
  }
});

void test('expanded Pi recall rendering equals complete execution evidence exactly', async () => {
  const previousChunk = createTestSessionConversationChunk({
    id: 'expanded-previous',
    content: 'Neighbor context before the winning evidence.',
  });
  const winningChunk = createTestSessionConversationChunk({
    id: 'expanded-winning',
    content: 'Winning evidence with exact source geometry.',
  });
  const duplicate = createTestRecallSearchResult({
    id: 'expanded-duplicate',
    sessionPath: '/sessions/duplicate.jsonl',
  });
  const result = createTestRankedRecallSearchResult({
    id: 'expanded-result',
    content: winningChunk.content,
    contributingEntryIds: [{ value: 'expanded-entry' }, { value: 'contributing-entry' }],
    toolCallEntryId: { value: 'tool-call-entry' },
    toolResultEntryId: { value: 'tool-result-entry' },
    lexical: { rank: 2, fullTextScore: 1.25 },
    identifier: { rank: 3, fullTextScore: 0.75 },
    duplicateOccurrences: [duplicate],
    neighborContext: {
      content: `${previousChunk.content} ${winningChunk.content}`,
      chunks: [previousChunk, winningChunk],
    },
  });
  const service = {
    async search() {
      return {
        results: [result],
        totalChunks: 12,
        indexMaintenanceStatus: null,
        searchPolicy: createEmptySearch(RecallSearchScope.GLOBAL).searchPolicy,
      };
    },
    async index() {
      throw new Error('search adapter must not index');
    },
  } satisfies RecallConversationService;
  const tool = createPiRecallToolDefinition(service);
  const execution = await tool.execute(
    'expanded-call',
    { query: 'rich evidence', scope: 'global' },
    undefined,
    undefined,
    { cwd: '/trusted/project' },
  );
  const executionText =
    execution.content[0]?.type === 'text' ? execution.content[0].text : 'missing text content';
  assert.match(executionText, /ranking 0\.0200/);
  assert.match(executionText, /Neighbor context before the winning evidence/);
  assert.match(executionText, /Contributing entries:/);
  assert.match(executionText, /Call source:/);
  assert.match(executionText, /Expanded chunks:/);
  assert.match(executionText, /Duplicate occurrence:/);
  assert.match(executionText, /Source:/);

  const component = tool.renderResult(
    execution,
    { expanded: true, isPartial: false },
    {
      fg(color, text) {
        void color;
        return text;
      },
    },
    { isError: false, lastComponent: undefined },
  );
  const rendered = component
    .render(2_000)
    .map((line) => line.trimEnd())
    .join('\n');

  assert.equal(rendered, executionText);
});

void test('zero-match Pi recall rendering stays concise without an expansion hint', async () => {
  initTheme('dark');
  const restoreKeybindings = await usePiToolExpansionKeybinding('alt+x');

  try {
    const service = {
      async search() {
        return {
          ...createEmptySearch(RecallSearchScope.PROJECT),
          totalChunks: 42_318,
          indexMaintenanceStatus: {
            version: 1,
            completedAt: '2026-07-25T12:00:00.000Z',
            scannedSessions: 123,
            failedSessions: 2,
          },
        };
      },
      async index() {
        throw new Error('search adapter must not index');
      },
    } satisfies RecallConversationService;
    const tool = createPiRecallToolDefinition(service, () => new Date('2026-07-25T13:00:00.000Z'));
    const execution = await tool.execute(
      'zero-match-call',
      { query: 'missing decision' },
      undefined,
      undefined,
      { cwd: '/trusted/project' },
    );

    assert.equal('returnedBytes' in execution.details, false);
    assert.equal('returnedLines' in execution.details, false);

    for (const expanded of [false, true]) {
      const component = tool.renderResult(
        execution,
        { expanded, isPartial: false },
        {
          fg(color, text) {
            void color;
            return text;
          },
        },
        { isError: false, lastComponent: undefined },
      );
      const rendered = stripVTControlCharacters(component.render(1_000).join('\n')).trimEnd();

      assert.equal(
        rendered,
        'No matching past conversations found · project scope · index checked 1h ago · 2 failed sessions',
      );
      assert.doesNotMatch(rendered, /alt\+x|expand/iu);
    }
  } finally {
    restoreKeybindings();
  }
});

void test('truncated Pi recall execution stores full metadata and marks collapsed output', async () => {
  initTheme('dark');
  const restoreKeybindings = await usePiToolExpansionKeybinding('alt+x');

  try {
    const duplicateOccurrences = Array.from({ length: 2_100 }, (_, index) =>
      createTestRecallSearchResult({ id: `truncation-duplicate-${index}` }),
    );
    const result = createTestRankedRecallSearchResult({
      id: 'truncated-result',
      duplicateOccurrences,
    });
    const truncatedService = {
      async search() {
        return {
          results: [result],
          totalChunks: 42_318,
          indexMaintenanceStatus: null,
          searchPolicy: createEmptySearch(RecallSearchScope.PROJECT).searchPolicy,
        };
      },
      async index() {
        throw new Error('search adapter must not index');
      },
    } satisfies RecallConversationService;
    const truncatedTool = createPiRecallToolDefinition(truncatedService);
    const truncatedExecution = await truncatedTool.execute(
      'truncated-call',
      { query: 'many duplicates' },
      undefined,
      undefined,
      { cwd: '/trusted/project' },
    );
    const truncatedText =
      truncatedExecution.content[0]?.type === 'text'
        ? truncatedExecution.content[0].text
        : 'missing text content';

    assert.match(truncatedText, /\[Recall output truncated to 50\.0KB\.\]$/u);
    const truncation = truncatedExecution.details.truncation;
    assert.ok(truncation);
    assert.deepEqual(Object.keys(truncation).sort(), [
      'content',
      'firstLineExceedsLimit',
      'lastLinePartial',
      'maxBytes',
      'maxLines',
      'outputBytes',
      'outputLines',
      'totalBytes',
      'totalLines',
      'truncated',
      'truncatedBy',
    ]);
    assert.equal(truncation.truncated, true);
    assert.equal(truncation.maxLines, 2_000);
    assert.equal(truncation.maxBytes, 50 * 1_024);
    assert.ok(truncation.totalLines > 2_000);
    assert.ok(truncation.outputBytes <= truncation.maxBytes);
    assert.equal(truncatedExecution.details.returnedBytes, truncation.outputBytes);
    assert.equal(truncatedExecution.details.returnedLines, truncation.outputLines);
    assert.ok(Buffer.byteLength(truncatedText, 'utf8') > truncation.outputBytes);

    const component = truncatedTool.renderResult(
      truncatedExecution,
      { expanded: false, isPartial: false },
      {
        fg(color, text) {
          void color;
          return text;
        },
      },
      { isError: false, lastComponent: undefined },
    );
    const rendered = stripVTControlCharacters(component.render(1_000).join('\n')).trimEnd();
    assert.equal(
      rendered,
      '1 recall result · project scope · 50.0KB / 278 lines · output truncated (alt+x to expand)',
    );

    const expandedComponent = truncatedTool.renderResult(
      truncatedExecution,
      { expanded: true, isPartial: false },
      {
        fg(color, text) {
          void color;
          return text;
        },
      },
      { isError: false, lastComponent: undefined },
    );
    const expandedText = expandedComponent
      .render(100_000)
      .map((line) => line.trimEnd())
      .join('\n');
    assert.equal(expandedText, truncatedText);

    const untruncatedService = {
      async search() {
        return createEmptySearch(RecallSearchScope.GLOBAL);
      },
      async index() {
        throw new Error('search adapter must not index');
      },
    } satisfies RecallConversationService;
    const untruncatedExecution = await createPiRecallToolDefinition(untruncatedService).execute(
      'untruncated-call',
      { query: 'ordinary result', scope: 'global' },
      undefined,
      undefined,
      { cwd: '/trusted/project' },
    );
    assert.equal('truncation' in untruncatedExecution.details, false);
  } finally {
    restoreKeybindings();
  }
});

void test('Pi recall validation and execution errors retain error presentation', async () => {
  const service = {
    async search() {
      throw new Error('Recall index unavailable');
    },
    async index() {
      throw new Error('search adapter must not index');
    },
  } satisfies RecallConversationService;
  const tool = createPiRecallToolDefinition(service);

  await assert.rejects(
    tool.execute('blank-query-call', { query: '   ' }, undefined, undefined, {
      cwd: '/trusted/project',
    }),
    /Recall query must not be blank/u,
  );
  await assert.rejects(
    tool.execute('failed-search-call', { query: 'missing index' }, undefined, undefined, {
      cwd: '/trusted/project',
    }),
    /Recall index unavailable/u,
  );

  const callComponent = tool.renderCall(
    { query: 'missing index' },
    {
      bold(text) {
        return text;
      },
      fg(color, text) {
        void color;
        return text;
      },
    },
    { isError: true, lastComponent: undefined },
  );
  const renderedCall = stripVTControlCharacters(callComponent.render(1_000).join('\n')).trimEnd();
  assert.equal(renderedCall, 'pi-session-recall');
  assert.doesNotMatch(renderedCall, /missing index/iu);

  const component = tool.renderResult(
    { content: [{ type: 'text', text: 'Recall index unavailable' }], details: undefined },
    { expanded: false, isPartial: false },
    {
      fg(color, text) {
        void color;
        return text;
      },
    },
    { isError: true, lastComponent: undefined },
  );
  const rendered = component
    .render(1_000)
    .map((line) => line.trimEnd())
    .join('\n');

  assert.equal(rendered, 'Recall index unavailable');
  assert.doesNotMatch(rendered, /recall results|indexed documents|to expand/iu);
});

void test('Pi tool details retain line, block, character, and contributing-entry provenance', () => {
  const result = createTestRankedRecallSearchResult({
    id: 'source-result',
    sessionPath: '/sessions/source.jsonl',
    entryId: { value: 'source-entry' },
    contributingEntryIds: [{ value: 'source-entry' }, { value: 'context-entry' }],
    sourceLineStart: 20,
    sourceLineEnd: 24,
    sourceBlockStart: 1,
    sourceBlockEnd: 3,
    characterStart: 8,
    characterEnd: 88,
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
  });

  const details = createPiRecallToolDetails({
    results: [result],
    totalChunks: 1,
    indexMaintenanceStatus: null,
    searchPolicy: createEmptySearch(RecallSearchScope.GLOBAL).searchPolicy,
  });

  assert.deepEqual(details.sources[0], {
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
    sessionOrigin: '/project',
    projectIdentity: null,
    projectIdentitySource: null,
    sessionPath: '/sessions/source.jsonl',
    entryId: 'source-entry',
    contributingEntryIds: ['source-entry', 'context-entry'],
    sourceLineStart: 20,
    sourceLineEnd: 24,
    sourceBlockStart: 1,
    sourceBlockEnd: 3,
    characterStart: 8,
    characterEnd: 88,
    isOnActiveBranch: true,
    rankingScore: 0.02,
    activeBranchPrior: 0,
    fusedScore: 0.02,
    dense: { rank: 1, cosineDistance: 0.1 },
    lexical: null,
    identifier: null,
    duplicateOccurrences: [],
    expandedChunks: [],
  });
});

void test('Pi extension registers only the read-only recall tool and directs maintenance to psr', async () => {
  let registeredToolCount = 0;
  let registeredName = '';
  let registeredDescription = '';
  let registeredParameters: unknown;
  await recallExtension({
    registerTool(tool) {
      registeredToolCount += 1;
      registeredName = tool.name;
      registeredDescription = tool.description;
      registeredParameters = tool.parameters;
    },
  });

  assert.equal(registeredToolCount, 1);
  assert.equal(registeredName, 'pi-session-recall');
  assert.match(registeredDescription, /only standalone `psr index` maintenance does/);
  assert.doesNotMatch(registeredDescription, /Qwen|query.plann|background/iu);
  const schemaText = JSON.stringify(registeredParameters);
  assert.match(schemaText, /query/);
  assert.match(schemaText, /scope/);
  assert.doesNotMatch(schemaText, /mode|rebuild/);
});
