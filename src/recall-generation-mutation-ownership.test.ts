import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import test, { type TestContext } from 'node:test';

import ts from 'typescript';

const recallGenerationMutationSymbolNames = new Set([
  'writeRecallActiveGenerationPointer',
  'writeRecallGenerationRegistry',
]);

interface RecallGenerationMutationOwnershipViolation {
  modulePath: string;
  symbolName: string;
}

function listTypeScriptSourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry): string[] => {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      return listTypeScriptSourceFiles(entryPath);
    }
    return entry.isFile() && /\.[cm]?tsx?$/u.test(entry.name) ? [entryPath] : [];
  });
}

function resolveAliasedTypeScriptSymbol(checker: ts.TypeChecker, symbol: ts.Symbol): ts.Symbol {
  let resolvedSymbol = symbol;
  const visitedSymbols = new Set<ts.Symbol>();
  while (
    (resolvedSymbol.flags & ts.SymbolFlags.Alias) !== 0 &&
    !visitedSymbols.has(resolvedSymbol)
  ) {
    visitedSymbols.add(resolvedSymbol);
    resolvedSymbol = checker.getAliasedSymbol(resolvedSymbol);
  }
  return resolvedSymbol;
}

function identifyRecallGenerationMutationSymbol(
  checker: ts.TypeChecker,
  symbol: ts.Symbol | undefined,
  generationStateFilePath: string,
): string | null {
  if (symbol === undefined) {
    return null;
  }
  const resolvedSymbol = resolveAliasedTypeScriptSymbol(checker, symbol);
  if (!recallGenerationMutationSymbolNames.has(resolvedSymbol.name)) {
    return null;
  }
  const isGenerationStateDeclaration = resolvedSymbol.declarations?.some(
    (declaration) => resolve(declaration.getSourceFile().fileName) === generationStateFilePath,
  );
  return isGenerationStateDeclaration === true ? resolvedSymbol.name : null;
}

function collectRecallGenerationMutationOwnershipViolations(
  sourceRoot: string,
): RecallGenerationMutationOwnershipViolation[] {
  const normalizedSourceRoot = resolve(sourceRoot);
  const generationStateFilePath = join(normalizedSourceRoot, 'recall-generation-state.ts');
  const generationTransitionsFilePath = join(
    normalizedSourceRoot,
    'recall-generation-transitions.ts',
  );
  const program = ts.createProgram({
    rootNames: listTypeScriptSourceFiles(normalizedSourceRoot),
    options: {
      module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext,
      noEmit: true,
      skipLibCheck: true,
      target: ts.ScriptTarget.ES2024,
    },
  });
  const checker = program.getTypeChecker();
  const violations = new Map<string, RecallGenerationMutationOwnershipViolation>();

  for (const sourceFile of program.getSourceFiles()) {
    const sourceFilePath = resolve(sourceFile.fileName);
    if (
      sourceFile.isDeclarationFile ||
      !sourceFilePath.startsWith(`${normalizedSourceRoot}/`) ||
      sourceFilePath === generationStateFilePath ||
      sourceFilePath === generationTransitionsFilePath ||
      /\.test(?:-utils)?\.[cm]?tsx?$/u.test(sourceFilePath)
    ) {
      continue;
    }

    const recordViolation = (symbolName: string): void => {
      const modulePath = relative(normalizedSourceRoot, sourceFilePath);
      violations.set(`${modulePath}:${symbolName}`, { modulePath, symbolName });
    };
    const visitNode = (node: ts.Node): void => {
      if (ts.isIdentifier(node)) {
        const symbolName = identifyRecallGenerationMutationSymbol(
          checker,
          checker.getSymbolAtLocation(node),
          generationStateFilePath,
        );
        if (symbolName !== null) {
          recordViolation(symbolName);
        }
      }
      if (
        ts.isExportDeclaration(node) &&
        node.moduleSpecifier !== undefined &&
        (node.exportClause === undefined || ts.isNamespaceExport(node.exportClause))
      ) {
        const moduleSymbol = checker.getSymbolAtLocation(node.moduleSpecifier);
        if (moduleSymbol !== undefined) {
          for (const exportedSymbol of checker.getExportsOfModule(moduleSymbol)) {
            const symbolName = identifyRecallGenerationMutationSymbol(
              checker,
              exportedSymbol,
              generationStateFilePath,
            );
            if (symbolName !== null) {
              recordViolation(symbolName);
            }
          }
        }
      }
      ts.forEachChild(node, visitNode);
    };
    visitNode(sourceFile);
  }

  return [...violations.values()].sort(
    (left, right) =>
      left.modulePath.localeCompare(right.modulePath) ||
      left.symbolName.localeCompare(right.symbolName),
  );
}

function assertRecallGenerationMutationOwnership(sourceRoot: string): void {
  const violations = collectRecallGenerationMutationOwnershipViolations(sourceRoot);
  if (violations.length === 0) {
    return;
  }
  throw new Error(
    violations
      .map(
        ({ modulePath, symbolName }) =>
          `Recall generation mutation ownership violation: ${modulePath} accesses ${symbolName}`,
      )
      .join('\n'),
  );
}

const generationStateSource = `
export function writeRecallActiveGenerationPointer(): void {}
export function writeRecallGenerationRegistry(): void {}
export function readRecallGenerationRegistry(): void {}
`;

async function createMutationOwnershipFixture(
  context: TestContext,
  files: Readonly<Record<string, string>>,
): Promise<string> {
  const sourceRoot = await mkdtemp(join(tmpdir(), 'recall-generation-mutation-ownership-'));
  context.after(() => rm(sourceRoot, { recursive: true, force: true }));
  await Promise.all(
    Object.entries({
      'recall-generation-state.ts': generationStateSource,
      'recall-generation-transitions.ts': `
import {
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';

export function activateGeneration(): void {
  writeRecallGenerationRegistry();
  writeRecallActiveGenerationPointer();
}
`,
      ...files,
    }).map(async ([relativePath, source]) => {
      const filePath = join(sourceRoot, relativePath);
      await mkdir(dirname(filePath), { recursive: true });
      await writeFile(filePath, source, 'utf8');
    }),
  );
  return sourceRoot;
}

void test('production generation registry and active-pointer mutation stays owned by named transitions', () => {
  assert.doesNotThrow(() => assertRecallGenerationMutationOwnership(join(import.meta.dirname)));
});

void test('generation mutation ownership reports a direct production import by module and symbol', async (context) => {
  const sourceRoot = await createMutationOwnershipFixture(context, {
    'rogue-operation.ts': `
import { writeRecallGenerationRegistry } from './recall-generation-state.js';
`,
  });

  assert.throws(
    () => assertRecallGenerationMutationOwnership(sourceRoot),
    /Recall generation mutation ownership violation: rogue-operation\.ts accesses writeRecallGenerationRegistry/u,
  );
});

void test('generation mutation ownership follows aliases, re-exports, and namespace access', async (context) => {
  const sourceRoot = await createMutationOwnershipFixture(context, {
    'aliased-operation.ts': `
import { writeRecallActiveGenerationPointer as publishPointer } from './recall-generation-state.js';
publishPointer();
`,
    'generation-state-barrel.ts': `
export {
  writeRecallGenerationRegistry as publishRegistry,
} from './recall-generation-state.js';
`,
    'namespace-operation.ts': `
import * as generationState from './recall-generation-state.js';
generationState.writeRecallGenerationRegistry();
`,
    'star-generation-state-barrel.ts': `
export * from './recall-generation-state.js';
`,
  });

  assert.throws(
    () => assertRecallGenerationMutationOwnership(sourceRoot),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(
        error.message,
        /aliased-operation\.ts accesses writeRecallActiveGenerationPointer/u,
      );
      assert.match(
        error.message,
        /generation-state-barrel\.ts accesses writeRecallGenerationRegistry/u,
      );
      assert.match(
        error.message,
        /namespace-operation\.ts accesses writeRecallGenerationRegistry/u,
      );
      assert.match(
        error.message,
        /star-generation-state-barrel\.ts accesses writeRecallActiveGenerationPointer/u,
      );
      assert.match(
        error.message,
        /star-generation-state-barrel\.ts accesses writeRecallGenerationRegistry/u,
      );
      return true;
    },
  );
});

void test('generation state, named transitions, and test fixtures retain mutation access', async (context) => {
  const sourceRoot = await createMutationOwnershipFixture(context, {
    'durable-generation-fixture.test.ts': `
import {
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
writeRecallActiveGenerationPointer();
writeRecallGenerationRegistry();
`,
  });

  assert.doesNotThrow(() => assertRecallGenerationMutationOwnership(sourceRoot));
});
