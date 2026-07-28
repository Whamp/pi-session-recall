import { homedir } from 'node:os';
import { join } from 'node:path';

import {
  isCanonicalPathWithinBoundary,
  resolveCanonicalPathBoundary,
} from './trusted-path-boundary.js';

/** Paths used to prove a benchmark or evaluation cannot overlap recall-owned data. */
export interface AssertRecallTestDataRootOptions {
  testDataRoot: string;
  repositoryRoot: string;
  homeDirectory?: string;
  environment?: NodeJS.ProcessEnv;
  configuredProtectedPaths?: readonly string[];
}

function canonicalPathsOverlap(left: string, right: string): boolean {
  return (
    left === right ||
    isCanonicalPathWithinBoundary(left, right) ||
    isCanonicalPathWithinBoundary(right, left)
  );
}

/** Resolves real paths and rejects any scratch root that overlaps production or committed evidence. */
export async function assertRecallTestDataRoot(
  options: AssertRecallTestDataRootOptions,
): Promise<string> {
  const homeDirectory = options.homeDirectory ?? homedir();
  const environment = options.environment ?? process.env;
  const environmentProtectedPaths = [
    environment.PI_RECALL_CONFIG,
    environment.PI_RECALL_DATA_DIRECTORY,
    environment.PI_RECALL_SESSIONS_DIRECTORY,
  ].filter((path): path is string => path !== undefined && path.length > 0);
  const protectedPaths = [
    join(homeDirectory, '.pi', 'agent', 'recall'),
    join(homeDirectory, '.pi', 'agent', 'sessions'),
    join(homeDirectory, '.pi', 'agent', 'settings.json'),
    join(homeDirectory, '.pi', 'agent', 'recall.json'),
    join(options.repositoryRoot, 'docs', 'evaluation'),
    join(options.repositoryRoot, 'evaluation', 'corpus'),
    ...environmentProtectedPaths,
    ...(options.configuredProtectedPaths ?? []),
  ];
  const [canonicalTestDataRoot, ...canonicalProtectedPaths] = await Promise.all([
    resolveCanonicalPathBoundary(options.testDataRoot),
    ...protectedPaths.map((path) => resolveCanonicalPathBoundary(path)),
  ]);
  const overlappingPath = canonicalProtectedPaths.find((path) =>
    canonicalPathsOverlap(canonicalTestDataRoot, path),
  );
  if (overlappingPath !== undefined) {
    throw new Error(
      `Recall test data root overlaps protected path: ${canonicalTestDataRoot} and ${overlappingPath}`,
    );
  }
  return canonicalTestDataRoot;
}
