import { execFileSync } from 'node:child_process';

import type { LiveQueryPlannedFailureSemanticsEvidence } from './query-planned-recall-evaluation.js';

interface RequiredRecallEvaluationSemanticTest {
  file: string;
  identity: string;
}

const PLANNER_FALLBACK_SEMANTIC_TEST: RequiredRecallEvaluationSemanticTest = {
  file: 'src/recall-conversation-service.test.ts',
  identity:
    'query-planned recall routes agent lists through capability-specific embedding operations',
};
const RERANKER_FAILURE_SEMANTIC_TEST: RequiredRecallEvaluationSemanticTest = {
  file: 'src/recall-conversation-service.test.ts',
  identity: 'recall service fails clearly when Qwen reranking is unavailable',
};
const PI_TOOL_CONTRACT_SEMANTIC_TESTS: readonly RequiredRecallEvaluationSemanticTest[] = [
  {
    file: 'src/recall-extension.test.ts',
    identity: 'Pi session recall registers collision-free tool guidance and index command',
  },
  {
    file: 'src/recall-extension.test.ts',
    identity: 'Pi recall tool details expose query-plan and position-aware ranking evidence',
  },
];

function escapeTestNamePattern(identity: string): string {
  return identity.replace(/[\\^$.*+?()[\]{}|]/gu, '\\$&');
}

/** Executes one exact Node test identity and rejects zero, duplicate, skipped, or failed execution. */
export function runExactRecallEvaluationSemanticTest(
  projectDirectory: string,
  testFile: string,
  testIdentity: string,
): true {
  const exactPattern = `^${escapeTestNamePattern(testIdentity)}$`;
  const childEnvironment = { ...process.env };
  delete childEnvironment.NODE_TEST_CONTEXT;
  const output = execFileSync(
    process.execPath,
    [
      '--import',
      'tsx',
      '--test',
      '--test-reporter=tap',
      '--test-name-pattern',
      exactPattern,
      testFile,
    ],
    {
      cwd: projectDirectory,
      encoding: 'utf8',
      env: childEnvironment,
      maxBuffer: 16 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  const lines = output.split(/\r?\n/u);
  const subtestLine = `# Subtest: ${testIdentity}`;
  const passedTestLine = new RegExp(`^ok \\d+ - ${escapeTestNamePattern(testIdentity)}$`, 'u');
  const executedCount = lines.filter((line) => line === subtestLine).length;
  const passedCount = lines.filter((line) => passedTestLine.test(line)).length;
  if (executedCount !== 1 || passedCount !== 1) {
    throw new Error(
      `Recall acceptance required test identity did not execute exactly once: ${testIdentity}`,
    );
  }
  return true;
}

function runRequiredRecallEvaluationSemanticTest(
  projectDirectory: string,
  requiredTest: RequiredRecallEvaluationSemanticTest,
): true {
  return runExactRecallEvaluationSemanticTest(
    projectDirectory,
    requiredTest.file,
    requiredTest.identity,
  );
}

/** Derives every failure-semantics field from independently executed exact test identities. */
export function verifyRequiredRecallEvaluationSemanticChecks(
  projectDirectory: string,
): LiveQueryPlannedFailureSemanticsEvidence {
  const plannerFallbackPublicServicePassed = runRequiredRecallEvaluationSemanticTest(
    projectDirectory,
    PLANNER_FALLBACK_SEMANTIC_TEST,
  );
  const rerankerFailurePublicServicePassed = runRequiredRecallEvaluationSemanticTest(
    projectDirectory,
    RERANKER_FAILURE_SEMANTIC_TEST,
  );
  const piToolContractPassed = PI_TOOL_CONTRACT_SEMANTIC_TESTS.every((requiredTest) =>
    runRequiredRecallEvaluationSemanticTest(projectDirectory, requiredTest),
  );
  return {
    plannerFallbackPublicServicePassed,
    rerankerFailurePublicServicePassed,
    piToolContractPassed,
  };
}
