import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useReducer,
  useRef
} from "react";
import { MotionConfig, motion, useReducedMotion } from "motion/react";
import { adjacentScene, createInitialState, engineReducer } from "./engine";
import { sceneFromHash, sceneIndex, scenes } from "./scenes";
import { SceneVisual } from "./SceneVisuals";
import { SCENE_IDS, type SceneId } from "./types";
import s from "./App.module.css";

const chapterGlyphs: Record<SceneId, string> = {
  request: "↗",
  tokenization: "Aa",
  scheduling: "≡",
  "model-runner": "▦",
  prefill: "▧",
  decode: "→",
  streaming: "≋",
  batching: "⋮",
  "kv-cache": "□",
  "prefix-cache": "⊕",
  distributed: "◇",
  speculative: "»",
  observability: "⌁"
};

export default function App() {
  const initialSceneRef = useRef(sceneFromHash(window.location.hash));
  const initialScene = initialSceneRef.current;
  const [state, dispatch] = useReducer(
    engineReducer,
    initialScene,
    createInitialState
  );
  const reducedMotion = useReducedMotion();
  const scrollerRef = useRef<HTMLDivElement>(null);
  const sectionRefs = useRef(new Map<SceneId, HTMLElement>());
  const activeRef = useRef(initialScene);
  const scrollTargetRef = useRef<SceneId | null>(initialScene);
  const currentIndex = sceneIndex(state.activeSceneId);

  useLayoutEffect(() => {
    const section = sectionRefs.current.get(initialScene);
    const scroller = scrollerRef.current;
    if (!section || !scroller) return;
    scroller.scrollTop = section.offsetTop;
  }, [initialScene]);

  useEffect(() => {
    activeRef.current = state.activeSceneId;
    const hash = `#${state.activeSceneId}`;
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }
  }, [state.activeSceneId]);

  const goTo = useCallback(
    (sceneId: SceneId, behavior: ScrollBehavior = "smooth") => {
      activeRef.current = sceneId;
      scrollTargetRef.current = sceneId;
      dispatch({ type: "GO_TO_SCENE", sceneId });
      sectionRefs.current.get(sceneId)?.scrollIntoView({
        behavior: reducedMotion ? "auto" : behavior,
        block: "start"
      });
    },
    [reducedMotion]
  );

  const goRelative = useCallback(
    (delta: number) => goTo(adjacentScene(state.activeSceneId, delta)),
    [goTo, state.activeSceneId]
  );

  useEffect(() => {
    const root = scrollerRef.current;
    if (!root) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible || visible.intersectionRatio < 0.58) return;
        const sceneId = visible.target.id as SceneId;
        if (scrollTargetRef.current) {
          if (sceneId !== scrollTargetRef.current) return;
          scrollTargetRef.current = null;
        }
        if (sceneId === activeRef.current) return;
        activeRef.current = sceneId;
        dispatch({ type: "GO_TO_SCENE", sceneId });
      },
      { root, threshold: [0.58, 0.75] }
    );
    sectionRefs.current.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onHashChange = () => goTo(sceneFromHash(window.location.hash), "auto");
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [goTo]);

  useEffect(() => {
    if (!state.runningScene) return;
    const timer = window.setInterval(
      () => dispatch({ type: "TICK" }),
      reducedMotion ? 900 : 720
    );
    return () => window.clearInterval(timer);
  }, [reducedMotion, state.runningScene]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target?.matches(
          "input, button, a, textarea, select, [contenteditable='true']"
        )
      ) {
        return;
      }
      if (["ArrowDown", "ArrowRight", "PageDown"].includes(event.key)) {
        event.preventDefault();
        goRelative(1);
      }
      if (["ArrowUp", "ArrowLeft", "PageUp"].includes(event.key)) {
        event.preventDefault();
        goRelative(-1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goRelative]);

  return (
    <MotionConfig reducedMotion="user">
      <div
        className={`${s.app} ${state.sidebarCollapsed ? s.appCollapsed : ""}`}
      >
        <aside className={s.sidebar} aria-label="LLM engine chapters">
          <div className={s.brandRow}>
            <a href="/" className={s.brand} aria-label="Junu Park home">
              <span className={s.brandMark} aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
              </span>
              <span className={s.brandCopy}>
                <strong>LLM Engine</strong>
                <small>interactive explainer</small>
              </span>
            </a>
            <button
              type="button"
              className={s.collapse}
              aria-expanded={!state.sidebarCollapsed}
              aria-label={
                state.sidebarCollapsed
                  ? "Expand chapter navigation"
                  : "Collapse chapter navigation"
              }
              onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })}
            >
              {state.sidebarCollapsed ? "›" : "‹"}
            </button>
          </div>

          <nav className={s.nav}>
            {[1, 2].map((act) => (
              <div className={s.act} key={act}>
                <p className={s.actLabel}>
                  <span>Act {act}</span>
                  <span>{act === 1 ? "One request" : "Under load"}</span>
                </p>
                {scenes
                  .filter((scene) => scene.act === act)
                  .map((scene, index) => {
                    const active = state.activeSceneId === scene.id;
                    return (
                      <button
                        type="button"
                        className={`${s.navItem} ${active ? s.navActive : ""}`}
                        onClick={() => goTo(scene.id)}
                        aria-current={active ? "step" : undefined}
                        key={scene.id}
                      >
                        <span className={s.navGlyph}>
                          {chapterGlyphs[scene.id]}
                        </span>
                        <span className={s.navCopy}>
                          <small>
                            {String(index + 1).padStart(2, "0")} · {scene.group}
                          </small>
                          <strong>{scene.title}</strong>
                        </span>
                      </button>
                    );
                  })}
              </div>
            ))}
          </nav>

          <div className={s.sidebarFoot}>
            <span>Deterministic simulation</span>
            <span>no live model</span>
          </div>
        </aside>

        <main className={s.main}>
          <header className={s.topbar}>
            <span className={s.mobileBrand}>LLM Engine</span>
            <span className={s.position}>
              {String(currentIndex + 1).padStart(2, "0")} /{" "}
              {String(SCENE_IDS.length).padStart(2, "0")}
            </span>
            <span className={s.topbarHint}>← → to navigate</span>
            <a href="/" className={s.exit}>
              junupark.xyz ↗
            </a>
          </header>

          <div className={s.scroller} ref={scrollerRef}>
            {scenes.map((scene) => {
              const active = scene.id === state.activeSceneId;
              return (
                <section
                  className={s.scene}
                  id={scene.id}
                  key={scene.id}
                  ref={(element) => {
                    if (element) sectionRefs.current.set(scene.id, element);
                  }}
                  aria-labelledby={`${scene.id}-title`}
                >
                  <div className={s.sceneInner}>
                    <motion.header
                      className={s.sceneHeading}
                      animate={{ opacity: active ? 1 : 0.42 }}
                      transition={{ duration: reducedMotion ? 0 : 0.28 }}
                    >
                      <p>
                        Act {scene.act} <span>›</span> {scene.group}
                      </p>
                      <h1 id={`${scene.id}-title`}>{scene.title}</h1>
                      <div>{scene.summary}</div>
                    </motion.header>

                    <SceneVisual
                      active={active}
                      dispatch={dispatch}
                      sceneId={scene.id}
                      state={state}
                    />
                  </div>
                </section>
              );
            })}
          </div>

          <div className={s.dots} aria-label="Lesson progress">
            {scenes.map((scene) => (
              <button
                type="button"
                key={scene.id}
                aria-label={`Go to ${scene.title}`}
                aria-current={
                  scene.id === state.activeSceneId ? "step" : undefined
                }
                className={
                  scene.id === state.activeSceneId ? s.dotActive : undefined
                }
                onClick={() => goTo(scene.id)}
              />
            ))}
          </div>

          <nav className={s.mobileNav} aria-label="Scene navigation">
            <button
              type="button"
              disabled={currentIndex === 0}
              onClick={() => goRelative(-1)}
              aria-label="Previous scene"
            >
              ←
            </button>
            <span>
              {String(currentIndex + 1).padStart(2, "0")} /{" "}
              {String(SCENE_IDS.length).padStart(2, "0")}
            </span>
            <button
              type="button"
              disabled={currentIndex === SCENE_IDS.length - 1}
              onClick={() => goRelative(1)}
              aria-label="Next scene"
            >
              →
            </button>
          </nav>
        </main>
      </div>
    </MotionConfig>
  );
}
