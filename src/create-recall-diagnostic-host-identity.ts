import { arch, cpus, platform } from 'node:os';

/** Hardware and pinned runtime identity attached to target-host recall diagnostics. */
export interface RecallDiagnosticHostIdentity {
  nodeVersion: string;
  platform: NodeJS.Platform;
  architecture: string;
  cpuModel: string;
  zvecVersion: '0.6.0';
}

/** Captures scalar host identity without reading user configuration or production recall paths. */
export function createRecallDiagnosticHostIdentity(): RecallDiagnosticHostIdentity {
  return {
    nodeVersion: process.version,
    platform: platform(),
    architecture: arch(),
    cpuModel: cpus()[0]?.model ?? 'unknown',
    zvecVersion: '0.6.0',
  };
}
