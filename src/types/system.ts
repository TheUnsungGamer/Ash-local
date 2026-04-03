export interface SystemLogState {
  logs: string[];
}

export interface SystemStats {
  tokensPerSecond: number;
  latency: number;
  gpuUsage: number;
  cpuUsage: number;
  vramUsed: number;
  vramTotal: number;
  contextSize: number;
}