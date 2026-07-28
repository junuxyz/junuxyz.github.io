import { sceneIndex } from "./scenes";
import {
  SCENE_IDS,
  type EngineEvent,
  type EngineState,
  type RunningScene,
  type SceneId,
  type SchedulerCandidate,
  type SchedulerDecision,
  type Token
} from "./types";

export const COMMON_PROMPT = "Explain KV cache";
export const RARE_PROMPT = "Unbelievably fast";

export const COMMON_TOKENS: Token[] = [
  { id: 683, text: "Explain", span: [0, 7], kind: "word" },
  { id: 521, text: " KV", span: [7, 10], kind: "word" },
  { id: 8304, text: " cache", span: [10, 16], kind: "word" }
];

export const RARE_TOKENS: Token[] = [
  { id: 1844, text: "Un", span: [0, 2], kind: "word" },
  { id: 982, text: "believ", span: [2, 8], kind: "word" },
  { id: 1312, text: "ably", span: [8, 12], kind: "word" },
  { id: 2948, text: " fast", span: [12, 17], kind: "word" }
];

export const GENERATED_TOKENS = ["It", " reuses", " prior", " keys", ".", "<eos>"];

export const SCHEDULER_CANDIDATES: SchedulerCandidate[] = [
  {
    id: "decode-a",
    label: "decode · A",
    requestedTokens: 1,
    state: "running",
    color: "blue"
  },
  {
    id: "decode-b",
    label: "decode · B",
    requestedTokens: 1,
    state: "running",
    color: "green"
  },
  {
    id: "prefill-c",
    label: "prefill · C",
    requestedTokens: 9,
    state: "waiting",
    color: "amber"
  }
];

export function scheduleRequests(
  candidates: SchedulerCandidate[],
  tokenBudget: number,
  chunkedPrefill: boolean
): SchedulerDecision {
  let remaining = tokenBudget;
  const scheduled: SchedulerDecision["scheduled"] = [];
  const ordered = [...candidates].sort((a, b) => {
    if (a.state === b.state) return 0;
    return a.state === "running" ? -1 : 1;
  });

  for (const candidate of ordered) {
    if (remaining === 0) break;
    if (!chunkedPrefill && candidate.requestedTokens > remaining) continue;
    const tokens = Math.min(candidate.requestedTokens, remaining);
    scheduled.push({ id: candidate.id, tokens });
    remaining -= tokens;
  }

  return { scheduled, remainingTokenBudget: remaining };
}

export function createInitialState(sceneId: SceneId = "request"): EngineState {
  return {
    activeSceneId: sceneId,
    sidebarCollapsed: false,
    runningScene: null,
    requestStep: 0,
    tokenPrompt: "common",
    selectedToken: 0,
    schedulerBudget: 8,
    schedulerRound: 0,
    runnerFocus: "req-a",
    runnerTensor: "input_ids",
    prefillLength: 8,
    prefillLayer: 0,
    decodeStep: 0,
    decodeFocus: 0,
    streamStep: 0,
    batchBudget: 8,
    chunkedPrefill: true,
    batchRound: 0,
    kvLogicalBlock: 0,
    prefixCache: false,
    parallelDegree: 2,
    parallelStep: 0,
    speculativeWidth: 4,
    speculativeStep: 0,
    load: 42
  };
}

export function adjacentScene(sceneId: SceneId, delta: number): SceneId {
  const next = Math.max(
    0,
    Math.min(SCENE_IDS.length - 1, sceneIndex(sceneId) + delta)
  );
  return SCENE_IDS[next];
}

const maxStep: Record<RunningScene, number> = {
  request: 7,
  prefill: 4,
  decode: GENERATED_TOKENS.length,
  streaming: GENERATED_TOKENS.length,
  distributed: 4,
  speculative: 3
};

function currentStep(state: EngineState, scene: RunningScene): number {
  switch (scene) {
    case "request":
      return state.requestStep;
    case "prefill":
      return state.prefillLayer;
    case "decode":
      return state.decodeStep;
    case "streaming":
      return state.streamStep;
    case "distributed":
      return state.parallelStep;
    case "speculative":
      return state.speculativeStep;
  }
}

function withStep(
  state: EngineState,
  scene: RunningScene,
  value: number
): EngineState {
  switch (scene) {
    case "request":
      return { ...state, requestStep: value };
    case "prefill":
      return { ...state, prefillLayer: value };
    case "decode":
      return {
        ...state,
        decodeStep: value,
        decodeFocus: Math.max(0, state.prefillLength + value - 1)
      };
    case "streaming":
      return { ...state, streamStep: value };
    case "distributed":
      return { ...state, parallelStep: value };
    case "speculative":
      return { ...state, speculativeStep: value };
  }
}

function advanceScene(
  state: EngineState,
  scene: RunningScene,
  wrap: boolean
): EngineState {
  const step = currentStep(state, scene);
  const next = step >= maxStep[scene] ? (wrap ? 0 : step) : step + 1;
  const advanced = withStep(state, scene, next);
  return step >= maxStep[scene] || next >= maxStep[scene]
    ? { ...advanced, runningScene: null }
    : advanced;
}

export function engineReducer(
  state: EngineState,
  event: EngineEvent
): EngineState {
  switch (event.type) {
    case "GO_TO_SCENE":
      return {
        ...state,
        activeSceneId: event.sceneId,
        runningScene: null
      };
    case "TOGGLE_SIDEBAR":
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };
    case "TOGGLE_RUN": {
      if (state.runningScene === event.sceneId) {
        return { ...state, runningScene: null };
      }
      const atEnd =
        currentStep(state, event.sceneId) >= maxStep[event.sceneId];
      const prepared = atEnd ? withStep(state, event.sceneId, 0) : state;
      return { ...prepared, runningScene: event.sceneId };
    }
    case "STEP_SCENE":
      return advanceScene(
        { ...state, runningScene: null },
        event.sceneId,
        true
      );
    case "RESET_SCENE":
      return {
        ...withStep(state, event.sceneId, 0),
        runningScene: null
      };
    case "TICK":
      return state.runningScene
        ? advanceScene(state, state.runningScene, false)
        : state;
    case "SET_REQUEST_STEP":
      return {
        ...state,
        requestStep: Math.max(0, Math.min(7, event.step)),
        runningScene: null
      };
    case "SET_TOKEN_PROMPT":
      return {
        ...state,
        tokenPrompt: event.prompt,
        selectedToken: 0
      };
    case "SELECT_TOKEN":
      return { ...state, selectedToken: event.index };
    case "SET_SCHEDULER_BUDGET":
      return {
        ...state,
        schedulerBudget: event.budget,
        schedulerRound: 0
      };
    case "RUN_SCHEDULER":
      return {
        ...state,
        schedulerRound: (state.schedulerRound + 1) % 4
      };
    case "SET_RUNNER_FOCUS":
      return { ...state, runnerFocus: event.requestId };
    case "SET_RUNNER_TENSOR":
      return { ...state, runnerTensor: event.tensor };
    case "SET_PREFILL_LENGTH":
      return {
        ...state,
        prefillLength: event.length,
        prefillLayer: 0,
        runningScene:
          state.runningScene === "prefill" ? null : state.runningScene
      };
    case "SET_DECODE_FOCUS":
      return { ...state, decodeFocus: event.index };
    case "SET_BATCH_BUDGET":
      return { ...state, batchBudget: event.budget, batchRound: 0 };
    case "TOGGLE_CHUNKED_PREFILL":
      return {
        ...state,
        chunkedPrefill: !state.chunkedPrefill,
        batchRound: 0
      };
    case "RUN_BATCH":
      return { ...state, batchRound: (state.batchRound + 1) % 4 };
    case "SET_KV_LOGICAL_BLOCK":
      return { ...state, kvLogicalBlock: event.block };
    case "TOGGLE_PREFIX_CACHE":
      return { ...state, prefixCache: !state.prefixCache };
    case "SET_PARALLEL_DEGREE":
      return {
        ...state,
        parallelDegree: event.degree,
        parallelStep: 0,
        runningScene:
          state.runningScene === "distributed" ? null : state.runningScene
      };
    case "SET_SPECULATIVE_WIDTH":
      return {
        ...state,
        speculativeWidth: event.width,
        speculativeStep: 0,
        runningScene:
          state.runningScene === "speculative" ? null : state.runningScene
      };
    case "SET_LOAD":
      return { ...state, load: event.load };
    default:
      return state;
  }
}
