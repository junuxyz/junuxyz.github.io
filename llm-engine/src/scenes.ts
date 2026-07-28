import { SCENE_IDS, type SceneDefinition, type SceneId } from "./types";

export const scenes: SceneDefinition[] = [
  {
    id: "request",
    act: 1,
    group: "The handoff",
    title: "Follow one request",
    summary: "Step the packet through every boundary between the client and GPU."
  },
  {
    id: "tokenization",
    act: 1,
    group: "Request processing",
    title: "Text is not what the model sees",
    summary: "Select a token to reveal the exact characters and vocabulary ID it represents."
  },
  {
    id: "scheduling",
    act: 1,
    group: "Scheduler",
    title: "Admission is a budget decision",
    summary: "Change the token budget, then run one scheduling iteration."
  },
  {
    id: "model-runner",
    act: 1,
    group: "ModelRunner",
    title: "Requests become one packed batch",
    summary: "Select a request or tensor row to trace how ragged sequences become GPU inputs."
  },
  {
    id: "prefill",
    act: 1,
    group: "GPU execution",
    title: "Prefill moves a whole prompt at once",
    summary: "Advance layer by layer and watch every prompt position populate the KV cache."
  },
  {
    id: "decode",
    act: 1,
    group: "GPU execution",
    title: "Decode trades compute for memory reads",
    summary: "Generate one token and see how much old KV state must be revisited."
  },
  {
    id: "streaming",
    act: 1,
    group: "Return path",
    title: "A token becomes an SSE event",
    summary: "Play the return path until EOS closes the stream and releases memory."
  },
  {
    id: "batching",
    act: 2,
    group: "Under load",
    title: "Continuous batching fills the gaps",
    summary: "Toggle chunked prefill to see who gets the next iteration’s token budget."
  },
  {
    id: "kv-cache",
    act: 2,
    group: "Memory",
    title: "Logical order, physical pages",
    summary: "Select a logical KV block to find its non-contiguous physical location."
  },
  {
    id: "prefix-cache",
    act: 2,
    group: "Memory",
    title: "Shared prefixes should be computed once",
    summary: "Enable prefix reuse and watch duplicate prompt blocks collapse into shared pages."
  },
  {
    id: "distributed",
    act: 2,
    group: "Scale out",
    title: "More GPUs add a synchronization wall",
    summary: "Change tensor parallelism, then advance one layer through compute and all-reduce."
  },
  {
    id: "speculative",
    act: 2,
    group: "Faster decode",
    title: "Draft many, verify once",
    summary: "Compare a draft burst with the tokens the target model accepts."
  },
  {
    id: "observability",
    act: 2,
    group: "Operations",
    title: "Load turns latency into a queue",
    summary: "Raise arrival pressure and watch the first production signals move together."
  }
];

export function sceneIndex(sceneId: SceneId): number {
  return SCENE_IDS.indexOf(sceneId);
}

export function sceneFromHash(hash: string): SceneId {
  const candidate = hash.replace(/^#/, "") as SceneId;
  return SCENE_IDS.includes(candidate) ? candidate : "request";
}
