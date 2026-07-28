import { motion, useReducedMotion } from "motion/react";
import {
  COMMON_PROMPT,
  COMMON_TOKENS,
  GENERATED_TOKENS,
  RARE_PROMPT,
  RARE_TOKENS,
  SCHEDULER_CANDIDATES,
  scheduleRequests
} from "./engine";
import type { Dispatch, ReactNode } from "react";
import type {
  EngineEvent,
  EngineState,
  RunningScene,
  SceneId,
  TensorKind
} from "./types";
import s from "./App.module.css";

interface SceneVisualProps {
  active: boolean;
  dispatch: Dispatch<EngineEvent>;
  sceneId: SceneId;
  state: EngineState;
}

function LessonCard({
  active,
  children,
  controls,
  label
}: {
  active: boolean;
  children: ReactNode;
  controls?: ReactNode;
  label: string;
}) {
  const reducedMotion = useReducedMotion();
  return (
    <motion.div
      className={s.lessonCard}
      animate={{
        opacity: active ? 1 : 0.42,
        y: active || reducedMotion ? 0 : 8
      }}
      transition={{ duration: reducedMotion ? 0 : 0.32 }}
      role="group"
      aria-label={label}
    >
      <div className={s.cardTop}>
        <span>{label}</span>
        {controls && <div className={s.cardControls}>{controls}</div>}
      </div>
      <div className={s.cardBody}>{children}</div>
    </motion.div>
  );
}

function Insight({ children }: { children: ReactNode }) {
  return (
    <div className={s.insight}>
      <span aria-hidden="true">◆</span>
      <p>{children}</p>
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
  tone = "blue"
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "blue" | "green" | "amber" | "violet" | "red";
}) {
  return (
    <div className={`${s.metric} ${s[`metric${tone}`]}`}>
      <small>{label}</small>
      <strong>{value}</strong>
      {detail && <span>{detail}</span>}
    </div>
  );
}

function StepControls({
  dispatch,
  scene,
  running,
  stepLabel = "Step"
}: {
  dispatch: Dispatch<EngineEvent>;
  scene: RunningScene;
  running: boolean;
  stepLabel?: string;
}) {
  return (
    <>
      <button
        type="button"
        className={s.iconControl}
        onClick={() => dispatch({ type: "RESET_SCENE", sceneId: scene })}
        aria-label={`Reset ${scene} simulation`}
      >
        ↺
      </button>
      <button
        type="button"
        className={`${s.controlButton} ${running ? s.controlActive : ""}`}
        onClick={() => dispatch({ type: "TOGGLE_RUN", sceneId: scene })}
        aria-pressed={running}
        aria-label={running ? `Pause ${scene} simulation` : `Play ${scene} simulation`}
      >
        {running ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        className={s.controlButton}
        onClick={() => dispatch({ type: "STEP_SCENE", sceneId: scene })}
      >
        {stepLabel} →
      </button>
    </>
  );
}

const requestStages = [
  {
    label: "Client",
    meta: "POST",
    detail: "Sends text, sampling parameters, and stream: true."
  },
  {
    label: "API",
    meta: "JSON",
    detail: "Validates the payload and opens the SSE connection."
  },
  {
    label: "Tokenizer",
    meta: "IDs",
    detail: "Turns characters into model vocabulary IDs."
  },
  {
    label: "Scheduler",
    meta: "queue",
    detail: "Admits the request when tokens and KV pages fit."
  },
  {
    label: "ModelRunner",
    meta: "batch",
    detail: "Packs ragged requests into model input tensors."
  },
  {
    label: "Prefill",
    meta: "N tokens",
    detail: "Processes every prompt position and writes its KV state."
  },
  {
    label: "Decode",
    meta: "1 token",
    detail: "Reads prior KV state, samples, and appends one new token."
  },
  {
    label: "Stream",
    meta: "SSE",
    detail: "Detokenizes output and releases pages when generation stops."
  }
];

function RequestVisual({ active, dispatch, state }: SceneVisualProps) {
  return (
    <LessonCard
      active={active}
      label="THE REQUEST › END-TO-END"
      controls={
        <StepControls
          dispatch={dispatch}
          scene="request"
          running={state.runningScene === "request"}
          stepLabel="Next boundary"
        />
      }
    >
      <div className={s.requestFlow}>
        <div className={s.requestRail} aria-hidden="true">
          <motion.i
            animate={{ left: `${(state.requestStep / 7) * 100}%` }}
            transition={{ type: "spring", stiffness: 220, damping: 28 }}
          />
        </div>
        <div className={s.requestStages}>
          {requestStages.map((stage, index) => (
            <button
              type="button"
              key={stage.label}
              className={`${s.requestStage} ${
                index === state.requestStep ? s.requestStageActive : ""
              } ${index < state.requestStep ? s.requestStageDone : ""}`}
              onClick={() => dispatch({ type: "SET_REQUEST_STEP", step: index })}
              aria-pressed={index === state.requestStep}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{stage.label}</strong>
              <small>{stage.meta}</small>
            </button>
          ))}
        </div>
        <motion.div
          className={s.focusNote}
          key={state.requestStep}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <span>{requestStages[state.requestStep].label}</span>
          <p>{requestStages[state.requestStep].detail}</p>
          <code>req_7f3a</code>
        </motion.div>
      </div>
      <Insight>
        The connection stays at the API edge; a request ID crosses every engine
        boundary.
      </Insight>
    </LessonCard>
  );
}

function TokenizationVisual({ active, dispatch, state }: SceneVisualProps) {
  const tokens = state.tokenPrompt === "common" ? COMMON_TOKENS : RARE_TOKENS;
  const prompt = state.tokenPrompt === "common" ? COMMON_PROMPT : RARE_PROMPT;
  const selected = tokens[Math.min(state.selectedToken, tokens.length - 1)];
  const chars = [...prompt];

  return (
    <LessonCard
      active={active}
      label="REQUEST PROCESSING › TOKENIZATION"
      controls={
        <div className={s.segmented} aria-label="Prompt example">
          <button
            type="button"
            aria-pressed={state.tokenPrompt === "common"}
            className={state.tokenPrompt === "common" ? s.segmentedActive : ""}
            onClick={() =>
              dispatch({ type: "SET_TOKEN_PROMPT", prompt: "common" })
            }
          >
            Familiar text
          </button>
          <button
            type="button"
            aria-pressed={state.tokenPrompt === "rare"}
            className={state.tokenPrompt === "rare" ? s.segmentedActive : ""}
            onClick={() =>
              dispatch({ type: "SET_TOKEN_PROMPT", prompt: "rare" })
            }
          >
            Rare pieces
          </button>
        </div>
      }
    >
      <div className={s.tokenizerLayout}>
        <div className={s.textSource}>
          <span>UTF-8 text</span>
          <p aria-label={prompt}>
            {chars.map((character, index) => (
              <mark
                className={
                  index >= selected.span[0] && index < selected.span[1]
                    ? s.charActive
                    : undefined
                }
                key={`${character}-${index}`}
              >
                {character === " " ? "\u00a0" : character}
              </mark>
            ))}
          </p>
        </div>
        <div className={s.tokenArrow} aria-hidden="true">
          <span>vocabulary lookup</span>
          <i />
          <b>→</b>
        </div>
        <div className={s.tokenOutput}>
          <span>{tokens.length} model tokens</span>
          <div className={s.tokenButtons}>
            {tokens.map((token, index) => (
              <button
                type="button"
                key={`${token.id}-${token.text}`}
                className={index === state.selectedToken ? s.tokenActive : ""}
                aria-pressed={index === state.selectedToken}
                onClick={() => dispatch({ type: "SELECT_TOKEN", index })}
              >
                <strong>{token.text.replace(/^ /, "·")}</strong>
                <small>{token.id}</small>
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className={s.tokenDetail}>
        <span>characters {selected.span[0]}–{selected.span[1] - 1}</span>
        <strong>“{selected.text.replace(/^ /, "␠")}”</strong>
        <span>vocab id {selected.id}</span>
      </div>
      <Insight>
        Token count—not character count—sets prompt cost, context use, and KV
        allocation.
      </Insight>
    </LessonCard>
  );
}

function SchedulingVisual({ active, dispatch, state }: SceneVisualProps) {
  const decision = scheduleRequests(
    SCHEDULER_CANDIDATES,
    state.schedulerBudget,
    true
  );
  const reveal = state.schedulerRound > 0;

  return (
    <LessonCard
      active={active}
      label="ENGINE CORE › ADMISSION"
      controls={
        <>
          <label className={s.rangeControl}>
            <span>Token budget</span>
            <input
              type="range"
              min="3"
              max="12"
              value={state.schedulerBudget}
              onChange={(event) =>
                dispatch({
                  type: "SET_SCHEDULER_BUDGET",
                  budget: Number(event.currentTarget.value)
                })
              }
              aria-label="Scheduler token budget"
            />
            <b>{state.schedulerBudget}</b>
          </label>
          <button
            type="button"
            className={s.controlButton}
            onClick={() => dispatch({ type: "RUN_SCHEDULER" })}
          >
            Run iteration →
          </button>
        </>
      }
    >
      <div className={s.scheduler}>
        <div className={s.schedulerQueue}>
          <span className={s.columnLabel}>priority queue</span>
          {SCHEDULER_CANDIDATES.map((request) => {
            const scheduled =
              decision.scheduled.find((item) => item.id === request.id)?.tokens ??
              0;
            return (
              <div className={s.queueRow} key={request.id}>
                <span className={`${s.requestDot} ${s[request.color]}`} />
                <strong>{request.label}</strong>
                <small>
                  {request.state === "running" ? "running" : "waiting"} · asks{" "}
                  {request.requestedTokens}
                </small>
                <b>{reveal ? `${scheduled} scheduled` : "—"}</b>
              </div>
            );
          })}
        </div>
        <div className={s.budgetPanel}>
          <span className={s.columnLabel}>next GPU iteration</span>
          <div
            className={`${s.budgetSlots} ${reveal ? s.budgetRevealed : ""}`}
            style={{
              gridTemplateColumns: `repeat(${state.schedulerBudget}, minmax(18px, 1fr))`
            }}
          >
            {Array.from({ length: state.schedulerBudget }, (_, index) => {
              let cursor = 0;
              const owner = decision.scheduled.find((item) => {
                const hit = index >= cursor && index < cursor + item.tokens;
                cursor += item.tokens;
                return hit;
              });
              return (
                <span
                  key={index}
                  className={owner ? s[owner.id.split("-")[0]] : s.empty}
                >
                  {reveal && owner
                    ? owner.id === "prefill-c"
                      ? "C"
                      : owner.id.at(-1)?.toUpperCase()
                    : ""}
                </span>
              );
            })}
          </div>
          <div className={s.budgetLegend}>
            <span>
              <i className={s.decode} /> running decode first
            </span>
            <span>
              <i className={s.prefill} /> prefill uses the remainder
            </span>
          </div>
        </div>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="scheduled"
          value={
            reveal
              ? `${decision.scheduled.reduce((sum, item) => sum + item.tokens, 0)}`
              : "—"
          }
          detail="tokens"
        />
        <Metric
          label="unused budget"
          value={reveal ? `${decision.remainingTokenBudget}` : "—"}
          detail="tokens"
          tone="amber"
        />
        <Metric
          label="policy"
          value="decode first"
          detail="then chunk prefill"
          tone="green"
        />
      </div>
      <Insight>
        The scheduler does not send “requests” to the GPU; it spends a per-step
        token budget.
      </Insight>
    </LessonCard>
  );
}

const runnerRequests = [
  {
    id: "req-a" as const,
    label: "A",
    input_ids: [683, 521, 8304],
    positions: [0, 1, 2],
    slot_mapping: [4, 5, 6],
    block_table: [1, 5]
  },
  {
    id: "req-b" as const,
    label: "B",
    input_ids: [1102],
    positions: [12],
    slot_mapping: [23],
    block_table: [0, 7, 2]
  },
  {
    id: "req-c" as const,
    label: "C",
    input_ids: [2391, 1148],
    positions: [0, 1],
    slot_mapping: [12, 13],
    block_table: [3]
  }
];

const tensorLabels: Array<{ id: TensorKind; note: string }> = [
  { id: "input_ids", note: "vocabulary IDs" },
  { id: "positions", note: "sequence offsets" },
  { id: "slot_mapping", note: "KV write slots" },
  { id: "block_table", note: "logical → physical" }
];

function ModelRunnerVisual({ active, dispatch, state }: SceneVisualProps) {
  return (
    <LessonCard
      active={active}
      label="MODEL RUNNER › BATCH ASSEMBLY"
      controls={
        <div className={s.runnerTabs} aria-label="Focus request">
          {runnerRequests.map((request) => (
            <button
              type="button"
              key={request.id}
              aria-pressed={state.runnerFocus === request.id}
              className={
                state.runnerFocus === request.id ? s.runnerTabActive : ""
              }
              onClick={() =>
                dispatch({
                  type: "SET_RUNNER_FOCUS",
                  requestId: request.id
                })
              }
            >
              request {request.label}
            </button>
          ))}
        </div>
      }
    >
      <div className={s.runnerLayout}>
        <div className={s.runnerSources}>
          <span className={s.columnLabel}>ragged sequences</span>
          {runnerRequests.map((request) => (
            <button
              type="button"
              key={request.id}
              aria-pressed={state.runnerFocus === request.id}
              onClick={() =>
                dispatch({
                  type: "SET_RUNNER_FOCUS",
                  requestId: request.id
                })
              }
              className={
                state.runnerFocus === request.id ? s.runnerSourceActive : ""
              }
            >
              <b>{request.label}</b>
              <span>
                {request.input_ids.map((_, index) => (
                  <i key={index} />
                ))}
              </span>
              <small>{request.input_ids.length} token(s)</small>
            </button>
          ))}
        </div>
        <div className={s.packArrow}>
          <span>pack</span>
          <b>→</b>
        </div>
        <div className={s.tensorPanel}>
          <span className={s.columnLabel}>contiguous GPU tensors</span>
          {tensorLabels.map((tensor) => (
            <button
              type="button"
              key={tensor.id}
              aria-pressed={state.runnerTensor === tensor.id}
              className={
                state.runnerTensor === tensor.id ? s.tensorRowActive : ""
              }
              onClick={() =>
                dispatch({ type: "SET_RUNNER_TENSOR", tensor: tensor.id })
              }
            >
              <span>
                <strong>{tensor.id}</strong>
                <small>{tensor.note}</small>
              </span>
              <code>
                {runnerRequests.flatMap((request) =>
                  request[tensor.id].map((value, index) => (
                    <i
                      key={`${request.id}-${index}`}
                      className={
                        request.id === state.runnerFocus ? s.tensorCellActive : ""
                      }
                    >
                      {value}
                    </i>
                  ))
                )}
              </code>
            </button>
          ))}
        </div>
      </div>
      <Insight>
        ModelRunner preserves each sequence’s identity through positions,
        mappings, and block tables—not padding.
      </Insight>
    </LessonCard>
  );
}

function PrefillVisual({ active, dispatch, state }: SceneVisualProps) {
  const layers = ["Embed", "Attention", "MLP", "Output"];
  return (
    <LessonCard
      active={active}
      label="GPU EXECUTION › PREFILL"
      controls={
        <>
          <label className={s.rangeControl}>
            <span>Prompt tokens</span>
            <input
              type="range"
              min="4"
              max="12"
              value={state.prefillLength}
              onChange={(event) =>
                dispatch({
                  type: "SET_PREFILL_LENGTH",
                  length: Number(event.currentTarget.value)
                })
              }
              aria-label="Prefill prompt length"
            />
            <b>{state.prefillLength}</b>
          </label>
          <StepControls
            dispatch={dispatch}
            scene="prefill"
            running={state.runningScene === "prefill"}
            stepLabel="Next layer"
          />
        </>
      }
    >
      <div className={s.prefillGrid}>
        <div className={s.prefillAxis}>
          <span>model layer</span>
          <span>prompt positions →</span>
        </div>
        {layers.map((layer, row) => (
          <div className={s.prefillRow} key={layer}>
            <strong>{layer}</strong>
            <div>
              {Array.from({ length: state.prefillLength }, (_, column) => (
                <motion.span
                  key={column}
                  className={
                    row < state.prefillLayer
                      ? s.prefillDone
                      : row === state.prefillLayer && state.prefillLayer < 4
                        ? s.prefillCurrent
                        : ""
                  }
                  animate={{
                    opacity: row <= state.prefillLayer ? 1 : 0.28,
                    scale: row === state.prefillLayer ? 1 : 0.96
                  }}
                  transition={{ delay: column * 0.018 }}
                >
                  {column}
                </motion.span>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className={s.prefillCache}>
        <span>KV writes</span>
        <div>
          {Array.from({ length: state.prefillLength }, (_, index) => (
            <i
              key={index}
              className={state.prefillLayer > 1 ? s.kvFilled : undefined}
            >
              {index}
            </i>
          ))}
        </div>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="positions in parallel"
          value={`${state.prefillLength}`}
          detail="same forward pass"
        />
        <Metric
          label="work shape"
          value="wide"
          detail="matrix-heavy"
          tone="violet"
        />
        <Metric
          label="KV state"
          value={state.prefillLayer > 1 ? "populated" : "waiting"}
          detail="used by decode"
          tone="green"
        />
      </div>
      <Insight>
        Prefill is wide: every prompt position advances through the same layer
        together.
      </Insight>
    </LessonCard>
  );
}

function DecodeVisual({ active, dispatch, state }: SceneVisualProps) {
  const cacheLength = state.prefillLength + state.decodeStep;
  const maxCells = state.prefillLength + GENERATED_TOKENS.length;
  const focus = Math.min(state.decodeFocus, Math.max(0, cacheLength - 1));
  const focusNew = focus >= state.prefillLength;
  const currentToken =
    state.decodeStep === 0
      ? "prompt tail"
      : GENERATED_TOKENS[Math.min(state.decodeStep - 1, GENERATED_TOKENS.length - 1)];

  return (
    <LessonCard
      active={active}
      label="GPU EXECUTION › DECODE"
      controls={
        <StepControls
          dispatch={dispatch}
          scene="decode"
          running={state.runningScene === "decode"}
          stepLabel="Generate token"
        />
      }
    >
      <div className={s.decodeFlow}>
        <div className={s.decodeInput}>
          <span>one input position</span>
          <strong>{currentToken}</strong>
          <small>position {state.prefillLength + state.decodeStep}</small>
        </div>
        <div className={s.decodeArrow}>
          <span>attention reads</span>
          <b>→</b>
        </div>
        <div className={s.decodeCachePanel}>
          <div>
            <span>existing KV cache</span>
            <small>click a cell</small>
          </div>
          <div className={s.decodeCells}>
            {Array.from({ length: maxCells }, (_, index) => {
              const filled = index < cacheLength;
              const generated = index >= state.prefillLength && filled;
              return (
                <button
                  type="button"
                  key={index}
                  disabled={!filled}
                  aria-label={`KV position ${index}${
                    generated ? ", generated" : ", prompt"
                  }`}
                  aria-pressed={focus === index}
                  className={`${filled ? s.cacheFilled : ""} ${
                    generated ? s.cacheGenerated : ""
                  } ${focus === index ? s.cacheFocus : ""}`}
                  onClick={() =>
                    dispatch({ type: "SET_DECODE_FOCUS", index })
                  }
                >
                  {index}
                </button>
              );
            })}
          </div>
        </div>
        <div className={s.decodeArrow}>
          <span>sample</span>
          <b>→</b>
        </div>
        <div className={s.decodeOutput}>
          <span>next token</span>
          <strong>
            {state.decodeStep === 0
              ? "?"
              : GENERATED_TOKENS[
                  Math.min(state.decodeStep - 1, GENERATED_TOKENS.length - 1)
                ]}
          </strong>
          <small>+ one KV write</small>
        </div>
      </div>
      <div className={s.cacheDetail}>
        <span>KV position {focus}</span>
        <strong>{focusNew ? "new this decode loop" : "reused from prefill"}</strong>
        <span>{focusNew ? "1 write" : "read by attention"}</span>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="old positions read"
          value={`${cacheLength}`}
          detail="grows every token"
        />
        <Metric label="new positions" value="1" detail="per sequence" tone="green" />
        <Metric
          label="work shape"
          value="tall"
          detail="memory-heavy"
          tone="amber"
        />
      </div>
      <Insight>
        Decode computes one new position but revisits every cached position, so
        memory traffic grows with context.
      </Insight>
    </LessonCard>
  );
}

function StreamingVisual({ active, dispatch, state }: SceneVisualProps) {
  const tokenIndex = Math.max(0, state.streamStep - 1);
  const token =
    state.streamStep === 0
      ? null
      : GENERATED_TOKENS[Math.min(tokenIndex, GENERATED_TOKENS.length - 1)];
  const eos = token === "<eos>";
  const visibleText = GENERATED_TOKENS.slice(
    0,
    Math.min(state.streamStep, GENERATED_TOKENS.length - 1)
  ).join("");

  return (
    <LessonCard
      active={active}
      label="RETURN PATH › STREAMING"
      controls={
        <StepControls
          dispatch={dispatch}
          scene="streaming"
          running={state.runningScene === "streaming"}
          stepLabel="Next token"
        />
      }
    >
      <div className={s.streamFlow}>
        {[
          ["Sample", token ? (eos ? "EOS" : `id ${1102 + tokenIndex}`) : "waiting"],
          ["Decode", token ? (eos ? "stop" : `“${token}”`) : "—"],
          ["SSE", token ? (eos ? "[DONE]" : `data: ${token}`) : "—"],
          ["Socket", token ? (eos ? "close" : "flush") : "open"],
          ["Client", eos ? "complete" : token ? "paint" : "waiting"]
        ].map(([label, value], index) => (
          <div
            className={`${s.streamNode} ${
              state.streamStep > 0 && index <= Math.min(4, state.streamStep)
                ? s.streamNodeActive
                : ""
            }`}
            key={label}
          >
            <span>{label}</span>
            <strong>{value}</strong>
            {index < 4 && <i aria-hidden="true">→</i>}
          </div>
        ))}
      </div>
      <div className={s.clientOutput}>
        <span>browser output</span>
        <p>
          {visibleText || "…"}
          {!eos && <i aria-hidden="true" />}
        </p>
        <b className={eos ? s.streamComplete : ""}>
          {eos ? "stream closed · KV pages released" : "text/event-stream open"}
        </b>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="time to first token"
          value={state.streamStep > 0 ? "observed" : "pending"}
          detail="request → first chunk"
        />
        <Metric
          label="inter-token latency"
          value={state.streamStep > 1 ? "cadence visible" : "pending"}
          detail="chunk → next chunk"
          tone="violet"
        />
        <Metric
          label="allocated pages"
          value={eos ? "0" : "3"}
          detail={eos ? "released" : "request owns them"}
          tone={eos ? "green" : "amber"}
        />
      </div>
      <Insight>
        Streaming hides total generation time, but TTFT and token cadence remain
        separate latency signals.
      </Insight>
    </LessonCard>
  );
}

function BatchingVisual({ active, dispatch, state }: SceneVisualProps) {
  const decision = scheduleRequests(
    SCHEDULER_CANDIDATES,
    state.batchBudget,
    state.chunkedPrefill
  );
  const used =
    state.batchRound > 0
      ? decision.scheduled.reduce((sum, item) => sum + item.tokens, 0)
      : 0;

  return (
    <LessonCard
      active={active}
      label="UNDER LOAD › CONTINUOUS BATCHING"
      controls={
        <>
          <button
            type="button"
            className={`${s.toggleControl} ${
              state.chunkedPrefill ? s.toggleOn : ""
            }`}
            aria-pressed={state.chunkedPrefill}
            onClick={() => dispatch({ type: "TOGGLE_CHUNKED_PREFILL" })}
          >
            <i />
            Chunked prefill
          </button>
          <label className={s.rangeControl}>
            <span>Budget</span>
            <input
              type="range"
              min="4"
              max="12"
              value={state.batchBudget}
              onChange={(event) =>
                dispatch({
                  type: "SET_BATCH_BUDGET",
                  budget: Number(event.currentTarget.value)
                })
              }
              aria-label="Batch token budget"
            />
            <b>{state.batchBudget}</b>
          </label>
          <button
            type="button"
            className={s.controlButton}
            onClick={() => dispatch({ type: "RUN_BATCH" })}
          >
            Next iteration →
          </button>
        </>
      }
    >
      <div className={s.batchTimeline}>
        <div className={s.batchHeader}>
          <span>request lanes</span>
          <span>iteration {state.batchRound || "—"}</span>
        </div>
        {SCHEDULER_CANDIDATES.map((candidate) => {
          const scheduled =
            decision.scheduled.find((item) => item.id === candidate.id)?.tokens ??
            0;
          return (
            <div className={s.batchLane} key={candidate.id}>
              <strong>{candidate.label}</strong>
              <span className={s.batchLaneTrack}>
                {Array.from({ length: candidate.requestedTokens }, (_, index) => (
                  <i
                    key={index}
                    className={
                      state.batchRound > 0 && index < scheduled
                        ? s[`lane${candidate.color}`]
                        : ""
                    }
                  />
                ))}
              </span>
              <small>
                {state.batchRound === 0
                  ? "ready"
                  : scheduled
                    ? `${scheduled} token${scheduled > 1 ? "s" : ""}`
                    : "waits"}
              </small>
            </div>
          );
        })}
      </div>
      <div
        className={s.batchBudgetBar}
        style={{
          gridTemplateColumns: `repeat(${state.batchBudget}, minmax(0, 1fr))`
        }}
      >
        {Array.from({ length: state.batchBudget }, (_, index) => (
          <i key={index} className={index < used ? s.batchUsed : s.batchIdle} />
        ))}
      </div>
      <div className={s.metricRow}>
        <Metric label="decodes advanced" value={state.batchRound ? "2 / 2" : "—"} />
        <Metric
          label="prefill chunk"
          value={
            state.batchRound
              ? `${decision.scheduled.find((item) => item.id === "prefill-c")?.tokens ?? 0}`
              : "—"
          }
          detail="tokens"
          tone="amber"
        />
        <Metric
          label="budget used"
          value={state.batchRound ? `${used} / ${state.batchBudget}` : "—"}
          detail={used < state.batchBudget ? "idle slots" : "full"}
          tone={used < state.batchBudget ? "red" : "green"}
        />
      </div>
      <Insight>
        Chunking lets short decode work coexist with a long prefill instead of
        leaving capacity idle.
      </Insight>
    </LessonCard>
  );
}

const logicalToPhysical = [5, 1, 7, 3];

function KVCacheVisual({ active, dispatch, state }: SceneVisualProps) {
  const physical = logicalToPhysical[state.kvLogicalBlock];
  return (
    <LessonCard active={active} label="MEMORY › PAGED KV CACHE">
      <div className={s.pagedLayout}>
        <div className={s.logicalBlocks}>
          <span>sequence A · logical order</span>
          <div>
            {logicalToPhysical.map((_, index) => (
              <button
                type="button"
                key={index}
                aria-pressed={state.kvLogicalBlock === index}
                className={
                  state.kvLogicalBlock === index ? s.logicalActive : ""
                }
                onClick={() =>
                  dispatch({ type: "SET_KV_LOGICAL_BLOCK", block: index })
                }
              >
                <small>logical</small>
                <strong>{index}</strong>
                <span>tokens {index * 4}–{index * 4 + 3}</span>
              </button>
            ))}
          </div>
        </div>
        <div className={s.pageTable}>
          <span>block table lookup</span>
          <strong>
            logical {state.kvLogicalBlock} <b>→</b> physical {physical}
          </strong>
          <small>one indirection, no contiguous allocation</small>
        </div>
        <div className={s.physicalBlocks}>
          <span>GPU memory · physical pages</span>
          <div>
            {Array.from({ length: 8 }, (_, index) => {
              const logical = logicalToPhysical.indexOf(index);
              return (
                <span
                  key={index}
                  className={`${logical >= 0 ? s.physicalUsed : ""} ${
                    index === physical ? s.physicalActive : ""
                  }`}
                >
                  <small>p{index}</small>
                  <strong>{logical >= 0 ? `L${logical}` : "free"}</strong>
                </span>
              );
            })}
          </div>
        </div>
      </div>
      <div className={s.metricRow}>
        <Metric label="logical blocks" value="4" detail="ordered sequence" />
        <Metric label="physical pages" value="4" detail="scattered" tone="violet" />
        <Metric label="copy on growth" value="none" detail="append another page" tone="green" />
      </div>
      <Insight>
        Page tables make free KV memory fungible; a sequence can grow without one
        contiguous reservation.
      </Insight>
    </LessonCard>
  );
}

function PrefixCacheVisual({ active, dispatch, state }: SceneVisualProps) {
  const shared = state.prefixCache;
  const physicalPages = shared ? 7 : 10;
  return (
    <LessonCard
      active={active}
      label="MEMORY › PREFIX REUSE"
      controls={
        <button
          type="button"
          className={`${s.toggleControl} ${shared ? s.toggleOn : ""}`}
          aria-pressed={shared}
          onClick={() => dispatch({ type: "TOGGLE_PREFIX_CACHE" })}
        >
          <i />
          Prefix cache
        </button>
      }
    >
      <div className={s.prefixLayout}>
        {["A", "B"].map((request) => (
          <div className={s.prefixRequest} key={request}>
            <strong>request {request}</strong>
            <div>
              {[0, 1, 2, 3, 4].map((block) => (
                <span
                  key={block}
                  className={block < 3 ? s.prefixShared : s.prefixUnique}
                >
                  <small>{block < 3 ? "system" : request}</small>
                  <b>{block < 3 ? `P${block}` : `${request}${block - 3}`}</b>
                </span>
              ))}
            </div>
          </div>
        ))}
        <div className={s.prefixLink} aria-hidden="true">
          <i className={shared ? s.prefixLinked : ""} />
          <span>{shared ? "same hash → same pages" : "duplicate allocation"}</span>
        </div>
        <div className={s.prefixPhysical}>
          <span>physical KV pages</span>
          <div>
            {Array.from({ length: 10 }, (_, index) => (
              <i
                key={index}
                className={`${index < physicalPages ? s.prefixAllocated : ""} ${
                  shared && index < 3 ? s.prefixSharedPhysical : ""
                }`}
              >
                {index < physicalPages ? index : "free"}
              </i>
            ))}
          </div>
        </div>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="prompt blocks computed"
          value={shared ? "7" : "10"}
          detail={shared ? "3 cache hits" : "no reuse"}
        />
        <Metric
          label="physical pages"
          value={`${physicalPages}`}
          detail="for two requests"
          tone="violet"
        />
        <Metric
          label="duplicate prefix"
          value={shared ? "0" : "3"}
          detail={shared ? "shared safely" : "extra pages"}
          tone={shared ? "green" : "amber"}
        />
      </div>
      <Insight>
        Prefix caching skips both prompt compute and KV allocation when completed
        token blocks match.
      </Insight>
    </LessonCard>
  );
}

function DistributedVisual({ active, dispatch, state }: SceneVisualProps) {
  const phaseLabels = ["Ready", "Shard input", "Compute", "All-reduce", "Next layer"];
  return (
    <LessonCard
      active={active}
      label="SCALE OUT › TENSOR PARALLELISM"
      controls={
        <>
          <div className={s.segmented} aria-label="Tensor parallel degree">
            {([1, 2, 4] as const).map((degree) => (
              <button
                type="button"
                key={degree}
                aria-pressed={state.parallelDegree === degree}
                className={
                  state.parallelDegree === degree ? s.segmentedActive : ""
                }
                onClick={() =>
                  dispatch({ type: "SET_PARALLEL_DEGREE", degree })
                }
              >
                {degree} GPU{degree > 1 ? "s" : ""}
              </button>
            ))}
          </div>
          <StepControls
            dispatch={dispatch}
            scene="distributed"
            running={state.runningScene === "distributed"}
            stepLabel="Advance"
          />
        </>
      }
    >
      <div className={s.distributedStage}>
        <div
          className={s.gpuGrid}
          style={{
            gridTemplateColumns: `repeat(${state.parallelDegree}, minmax(0, 1fr))`
          }}
        >
          {Array.from({ length: state.parallelDegree }, (_, index) => (
            <div
              className={`${s.gpuCard} ${
                state.parallelStep === 2 ? s.gpuCompute : ""
              } ${state.parallelStep === 3 ? s.gpuSync : ""}`}
              key={index}
            >
              <span>GPU {index}</span>
              <strong>weight shard {index + 1}/{state.parallelDegree}</strong>
              <div>
                {Array.from({ length: 12 }, (_, cell) => (
                  <i key={cell} />
                ))}
              </div>
              <small>
                {state.parallelStep === 2
                  ? "local GEMM"
                  : state.parallelStep === 3
                    ? "waiting at collective"
                    : "resident"}
              </small>
            </div>
          ))}
        </div>
        <div
          className={`${s.allReduce} ${
            state.parallelStep === 3 ? s.allReduceActive : ""
          }`}
        >
          <span>collective barrier</span>
          <strong>
            {state.parallelDegree === 1 ? "no exchange" : "all-reduce"}
          </strong>
          <i />
        </div>
        <div className={s.phaseTrack}>
          {phaseLabels.map((phase, index) => (
            <span
              key={phase}
              className={index === state.parallelStep ? s.phaseActive : ""}
            >
              <i />
              {phase}
            </span>
          ))}
        </div>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="weight per GPU"
          value={`${Math.round(100 / state.parallelDegree)}%`}
          detail="sharded"
        />
        <Metric
          label="workers in lockstep"
          value={`${state.parallelDegree}`}
          detail="same layer"
          tone="violet"
        />
        <Metric
          label="collectives / layer"
          value={state.parallelDegree === 1 ? "0" : "2"}
          detail="synchronization cost"
          tone="amber"
        />
      </div>
      <Insight>
        Tensor parallelism shrinks each GPU’s matrix work but adds collective
        synchronization inside every layer.
      </Insight>
    </LessonCard>
  );
}

function SpeculativeVisual({ active, dispatch, state }: SceneVisualProps) {
  const candidates =
    state.speculativeWidth === 2
      ? [" reuses", " prior"]
      : [" reuses", " prior", " values", " slowly"];
  const accepted = state.speculativeWidth === 2 ? 2 : 2;

  return (
    <LessonCard
      active={active}
      label="FASTER DECODE › SPECULATIVE VERIFICATION"
      controls={
        <>
          <div className={s.segmented} aria-label="Draft width">
            {([2, 4] as const).map((width) => (
              <button
                type="button"
                key={width}
                className={
                  state.speculativeWidth === width ? s.segmentedActive : ""
                }
                aria-pressed={state.speculativeWidth === width}
                onClick={() =>
                  dispatch({ type: "SET_SPECULATIVE_WIDTH", width })
                }
              >
                draft {width}
              </button>
            ))}
          </div>
          <StepControls
            dispatch={dispatch}
            scene="speculative"
            running={state.runningScene === "speculative"}
            stepLabel="Advance"
          />
        </>
      }
    >
      <div className={s.speculativeFlow}>
        <div className={s.modelLane}>
          <span>small draft model</span>
          <strong>{state.speculativeStep >= 1 ? "proposes in sequence" : "ready"}</strong>
          <div>
            {candidates.map((token, index) => (
              <motion.i
                key={token}
                animate={{
                  opacity: state.speculativeStep >= 1 ? 1 : 0.18,
                  y: state.speculativeStep >= 1 ? 0 : 6
                }}
                transition={{ delay: index * 0.06 }}
              >
                {token}
              </motion.i>
            ))}
          </div>
        </div>
        <div className={s.verifyArrow}>
          <span>one target pass</span>
          <b>↓</b>
        </div>
        <div className={s.modelLane}>
          <span>large target model</span>
          <strong>
            {state.speculativeStep >= 2 ? "verifies in parallel" : "waiting"}
          </strong>
          <div>
            {candidates.map((token, index) => {
              const resolved = state.speculativeStep >= 3;
              const isAccepted = index < accepted;
              return (
                <i
                  key={token}
                  className={`${resolved && isAccepted ? s.tokenAccepted : ""} ${
                    resolved && !isAccepted ? s.tokenRejected : ""
                  }`}
                >
                  {resolved ? (isAccepted ? "✓" : "×") : "?"} {token}
                </i>
              );
            })}
          </div>
        </div>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="drafted"
          value={state.speculativeStep ? `${state.speculativeWidth}` : "—"}
          detail="cheap proposals"
        />
        <Metric
          label="accepted"
          value={state.speculativeStep >= 3 ? `${accepted}` : "—"}
          detail="committed at once"
          tone="green"
        />
        <Metric
          label="rejected"
          value={
            state.speculativeStep >= 3
              ? `${state.speculativeWidth - accepted}`
              : "—"
          }
          detail="resume normal decode"
          tone="red"
        />
      </div>
      <Insight>
        Speedup comes from accepting several cheap proposals per expensive target
        pass—not from trusting the draft model.
      </Insight>
    </LessonCard>
  );
}

function ObservabilityVisual({ active, dispatch, state }: SceneVisualProps) {
  const pressure =
    state.load < 55 ? "healthy" : state.load < 80 ? "busy" : "saturated";
  const queue = Math.max(0, Math.round((state.load - 42) / 7));
  const cache = Math.min(98, Math.round(state.load * 0.94));
  const cadence = state.load < 55 ? "steady" : state.load < 80 ? "widening" : "bursty";

  return (
    <LessonCard
      active={active}
      label="OPERATIONS › QUEUEING PRESSURE"
      controls={
        <label className={s.rangeControl}>
          <span>Arrival load</span>
          <input
            type="range"
            min="25"
            max="100"
            value={state.load}
            onChange={(event) =>
              dispatch({
                type: "SET_LOAD",
                load: Number(event.currentTarget.value)
              })
            }
            aria-label="Arrival load"
          />
          <b>{state.load}%</b>
        </label>
      }
    >
      <div className={s.opsLayout}>
        <div className={s.arrivals}>
          <span>incoming requests</span>
          <div>
            {Array.from({ length: 12 }, (_, index) => (
              <motion.i
                key={index}
                animate={{
                  opacity: index < Math.ceil(state.load / 9) ? 1 : 0.14,
                  x: index < Math.ceil(state.load / 9) ? 0 : -4
                }}
              />
            ))}
          </div>
        </div>
        <div className={s.queueViz}>
          <span>scheduler queue</span>
          <div>
            {Array.from({ length: 8 }, (_, index) => (
              <i key={index} className={index < queue ? s.queueFilled : ""} />
            ))}
          </div>
          <strong>{pressure}</strong>
        </div>
        <div className={s.engineCapacity}>
          <span>engine capacity</span>
          <div>
            <i style={{ width: `${Math.min(100, state.load)}%` }} />
          </div>
          <small>fixed service rate</small>
        </div>
      </div>
      <div className={s.metricRow}>
        <Metric
          label="queue depth"
          value={queue === 0 ? "empty" : queue < 5 ? "rising" : "high"}
          detail="admission delay"
          tone={queue < 5 ? "blue" : "red"}
        />
        <Metric
          label="KV utilization"
          value={`${cache}%`}
          detail={cache > 88 ? "preemption risk" : "headroom remains"}
          tone={cache > 88 ? "red" : "violet"}
        />
        <Metric
          label="token cadence"
          value={cadence}
          detail="decode health"
          tone={cadence === "steady" ? "green" : "amber"}
        />
      </div>
      <Insight>
        Queue depth, KV pressure, and token cadence separate demand, memory, and
        execution bottlenecks.
      </Insight>
    </LessonCard>
  );
}

export function SceneVisual(props: SceneVisualProps) {
  switch (props.sceneId) {
    case "request":
      return <RequestVisual {...props} />;
    case "tokenization":
      return <TokenizationVisual {...props} />;
    case "scheduling":
      return <SchedulingVisual {...props} />;
    case "model-runner":
      return <ModelRunnerVisual {...props} />;
    case "prefill":
      return <PrefillVisual {...props} />;
    case "decode":
      return <DecodeVisual {...props} />;
    case "streaming":
      return <StreamingVisual {...props} />;
    case "batching":
      return <BatchingVisual {...props} />;
    case "kv-cache":
      return <KVCacheVisual {...props} />;
    case "prefix-cache":
      return <PrefixCacheVisual {...props} />;
    case "distributed":
      return <DistributedVisual {...props} />;
    case "speculative":
      return <SpeculativeVisual {...props} />;
    case "observability":
      return <ObservabilityVisual {...props} />;
  }
}
