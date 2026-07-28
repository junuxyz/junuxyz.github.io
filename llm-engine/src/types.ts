export const SCENE_IDS = [
  "request",
  "tokenization",
  "scheduling",
  "model-runner",
  "prefill",
  "decode",
  "streaming",
  "batching",
  "kv-cache",
  "prefix-cache",
  "distributed",
  "speculative",
  "observability"
] as const;

export type SceneId = (typeof SCENE_IDS)[number];
export type Act = 1 | 2;

export interface SceneDefinition {
  id: SceneId;
  act: Act;
  group: string;
  title: string;
  summary: string;
}

export type TokenKind = "special" | "word" | "generated";

export interface Token {
  id: number;
  text: string;
  span: [number, number];
  kind: TokenKind;
}

export type RequestState =
  | "waiting"
  | "running"
  | "streaming"
  | "finished"
  | "preempted";

export interface SchedulerCandidate {
  id: string;
  label: string;
  requestedTokens: number;
  state: "waiting" | "running";
  color: "blue" | "green" | "amber";
}

export interface SchedulerDecision {
  scheduled: Array<{ id: string; tokens: number }>;
  remainingTokenBudget: number;
}

export type TensorKind =
  | "input_ids"
  | "positions"
  | "slot_mapping"
  | "block_table";

export type RunningScene =
  | "request"
  | "prefill"
  | "decode"
  | "streaming"
  | "distributed"
  | "speculative";

export interface EngineState {
  activeSceneId: SceneId;
  sidebarCollapsed: boolean;
  runningScene: RunningScene | null;
  requestStep: number;
  tokenPrompt: "common" | "rare";
  selectedToken: number;
  schedulerBudget: number;
  schedulerRound: number;
  runnerFocus: "req-a" | "req-b" | "req-c";
  runnerTensor: TensorKind;
  prefillLength: number;
  prefillLayer: number;
  decodeStep: number;
  decodeFocus: number;
  streamStep: number;
  batchBudget: number;
  chunkedPrefill: boolean;
  batchRound: number;
  kvLogicalBlock: number;
  prefixCache: boolean;
  parallelDegree: 1 | 2 | 4;
  parallelStep: number;
  speculativeWidth: 2 | 4;
  speculativeStep: number;
  load: number;
}

export type EngineEvent =
  | { type: "GO_TO_SCENE"; sceneId: SceneId }
  | { type: "TOGGLE_SIDEBAR" }
  | { type: "TOGGLE_RUN"; sceneId: RunningScene }
  | { type: "STEP_SCENE"; sceneId: RunningScene }
  | { type: "RESET_SCENE"; sceneId: RunningScene }
  | { type: "TICK" }
  | { type: "SET_REQUEST_STEP"; step: number }
  | { type: "SET_TOKEN_PROMPT"; prompt: "common" | "rare" }
  | { type: "SELECT_TOKEN"; index: number }
  | { type: "SET_SCHEDULER_BUDGET"; budget: number }
  | { type: "RUN_SCHEDULER" }
  | { type: "SET_RUNNER_FOCUS"; requestId: "req-a" | "req-b" | "req-c" }
  | { type: "SET_RUNNER_TENSOR"; tensor: TensorKind }
  | { type: "SET_PREFILL_LENGTH"; length: number }
  | { type: "SET_DECODE_FOCUS"; index: number }
  | { type: "SET_BATCH_BUDGET"; budget: number }
  | { type: "TOGGLE_CHUNKED_PREFILL" }
  | { type: "RUN_BATCH" }
  | { type: "SET_KV_LOGICAL_BLOCK"; block: number }
  | { type: "TOGGLE_PREFIX_CACHE" }
  | { type: "SET_PARALLEL_DEGREE"; degree: 1 | 2 | 4 }
  | { type: "SET_SPECULATIVE_WIDTH"; width: 2 | 4 }
  | { type: "SET_LOAD"; load: number };
