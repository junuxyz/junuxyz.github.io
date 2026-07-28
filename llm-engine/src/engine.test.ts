import { describe, expect, it } from "vitest";
import {
  SCHEDULER_CANDIDATES,
  adjacentScene,
  createInitialState,
  engineReducer,
  scheduleRequests
} from "./engine";
import { sceneFromHash } from "./scenes";
import type { EngineState } from "./types";

describe("engine state machine", () => {
  it("keeps scene navigation bounded", () => {
    expect(adjacentScene("request", -1)).toBe("request");
    expect(adjacentScene("request", 1)).toBe("tokenization");
    expect(adjacentScene("observability", 1)).toBe("observability");
  });

  it("plays and resets a request simulation deterministically", () => {
    let state = createInitialState("request");
    state = engineReducer(state, {
      type: "TOGGLE_RUN",
      sceneId: "request"
    });
    expect(state.runningScene).toBe("request");

    state = engineReducer(state, { type: "TICK" });
    state = engineReducer(state, { type: "TICK" });
    expect(state.requestStep).toBe(2);

    state = engineReducer(state, {
      type: "RESET_SCENE",
      sceneId: "request"
    });
    expect(state.requestStep).toBe(0);
    expect(state.runningScene).toBeNull();
  });

  it("stops decode at EOS", () => {
    let state = createInitialState("decode");
    state = engineReducer(state, {
      type: "TOGGLE_RUN",
      sceneId: "decode"
    });
    for (let index = 0; index < 10; index += 1) {
      state = engineReducer(state, { type: "TICK" });
    }
    expect(state.decodeStep).toBe(6);
    expect(state.runningScene).toBeNull();
  });

  it("finishes streaming at EOS and stays terminal", () => {
    let state = createInitialState("streaming");
    state = engineReducer(state, {
      type: "TOGGLE_RUN",
      sceneId: "streaming"
    });
    for (let index = 0; index < 10; index += 1) {
      state = engineReducer(state, { type: "TICK" });
    }
    expect(state.streamStep).toBe(6);
    expect(state.runningScene).toBeNull();
  });

  it("resets dependent state when a control changes", () => {
    let state: EngineState = {
      ...createInitialState("prefill"),
      prefillLayer: 3,
      runningScene: "prefill" as const
    };
    state = engineReducer(state, {
      type: "SET_PREFILL_LENGTH",
      length: 11
    });
    expect(state.prefillLength).toBe(11);
    expect(state.prefillLayer).toBe(0);
    expect(state.runningScene).toBeNull();
  });

  it("toggles production mechanisms without hidden timers", () => {
    let state = createInitialState("batching");
    state = engineReducer(state, { type: "TOGGLE_CHUNKED_PREFILL" });
    state = engineReducer(state, { type: "TOGGLE_PREFIX_CACHE" });
    state = engineReducer(state, {
      type: "SET_PARALLEL_DEGREE",
      degree: 4
    });
    expect(state.chunkedPrefill).toBe(false);
    expect(state.prefixCache).toBe(true);
    expect(state.parallelDegree).toBe(4);
  });
});

describe("scheduler budget", () => {
  it("serves running decodes and chunks prefill into the remainder", () => {
    const result = scheduleRequests(SCHEDULER_CANDIDATES, 8, true);
    expect(result.scheduled).toEqual([
      { id: "decode-a", tokens: 1 },
      { id: "decode-b", tokens: 1 },
      { id: "prefill-c", tokens: 6 }
    ]);
    expect(result.remainingTokenBudget).toBe(0);
  });

  it("leaves an oversized prefill waiting when chunking is disabled", () => {
    const result = scheduleRequests(SCHEDULER_CANDIDATES, 8, false);
    expect(result.scheduled).toEqual([
      { id: "decode-a", tokens: 1 },
      { id: "decode-b", tokens: 1 }
    ]);
    expect(result.remainingTokenBudget).toBe(6);
  });
});

describe("stable fragments", () => {
  it("accepts direct scene fragments and falls back safely", () => {
    expect(sceneFromHash("#prefill")).toBe("prefill");
    expect(sceneFromHash("#prefix-cache")).toBe("prefix-cache");
    expect(sceneFromHash("#not-a-scene")).toBe("request");
  });
});
