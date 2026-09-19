import React, { useEffect, useMemo, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Point this at your FastAPI server. Falls back to an offline        */
/*  approximation if the server isn't running.                        */
/* ------------------------------------------------------------------ */
const API_BASE = "http://127.0.0.1:8000";

const INK = "#000000";
const BONE = "#EFEDE6";
const ASH = "#6E6E6E";
const BLOOD = "#C21807";

const DISPLAY = 'Impact, Haettenschweiler, "Arial Narrow Bold", "Franklin Gothic Heavy", sans-serif';
const TEXT = '"Helvetica Neue", Helvetica, Arial, sans-serif';

const CSS = `
.zs-root *, .zs-root *::before, .zs-root *::after { box-sizing: border-box; }
.zs-root { background: ${INK}; color: ${BONE}; font-family: ${TEXT}; min-height: 100%; }

.zs-root button { font-family: inherit; cursor: pointer; }
.zs-root :focus-visible { outline: 2px solid ${BONE}; outline-offset: 3px; }

@keyframes zs-develop { from { opacity: 0; filter: contrast(0.2) brightness(3); } to { opacity: 1; filter: none; } }
@keyframes zs-slam { 0% { transform: scale(1.35); opacity: 0; } 60% { transform: scale(0.985); opacity: 1; } 100% { transform: scale(1); opacity: 1; } }
@keyframes zs-twitch { 0%, 92%, 100% { transform: translate(0,0); } 94% { transform: translate(-2px,1px); } 96% { transform: translate(2px,-1px); } }
@keyframes zs-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

.zs-panel { animation: zs-develop 620ms steps(6, end) both; }
.zs-title { animation: zs-slam 420ms cubic-bezier(.2,.9,.2,1) both; }
.zs-twitch { animation: zs-twitch 5.5s infinite steps(1, end); }
.zs-rise { animation: zs-rise 420ms ease-out both; }

/* Choice chips ---------------------------------------------------- */
.zs-chip {
  background: transparent; color: ${BONE};
  border: 1px solid ${BONE};
  padding: 9px 13px; font-size: 13px; line-height: 1.1;
  letter-spacing: 0.01em;
}
.zs-chip:hover { background: rgba(239,237,230,0.14); }
.zs-chip[data-on="true"] { background: ${BONE}; color: ${INK}; font-weight: 700; }

/* Range ----------------------------------------------------------- */
.zs-range { -webkit-appearance: none; appearance: none; width: 100%; background: transparent; height: 26px; }
.zs-range::-webkit-slider-runnable-track { height: 1px; background: ${BONE}; }
.zs-range::-moz-range-track { height: 1px; background: ${BONE}; }
.zs-range::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 15px; height: 22px; background: ${BONE}; border: 0; border-radius: 0; margin-top: -11px;
}
.zs-range::-moz-range-thumb { width: 15px; height: 22px; background: ${BONE}; border: 0; border-radius: 0; }
.zs-range:active::-webkit-slider-thumb { background: ${BLOOD}; }
.zs-range:active::-moz-range-thumb { background: ${BLOOD}; }

/* Buttons --------------------------------------------------------- */
.zs-cta {
  background: ${BONE}; color: ${INK}; border: 0;
  font-family: ${DISPLAY}; letter-spacing: 0.03em;
  padding: 14px 34px; font-size: 26px; line-height: 1;
}
.zs-cta:hover { background: ${BLOOD}; color: ${BONE}; }
.zs-cta:disabled { background: ${ASH}; color: ${INK}; cursor: wait; }
.zs-ghost {
  background: transparent; color: ${BONE}; border: 1px solid ${BONE};
  padding: 11px 22px; font-size: 14px;
}
.zs-ghost:hover { background: ${BONE}; color: ${INK}; }

@media (prefers-reduced-motion: reduce) {
  .zs-panel, .zs-title, .zs-twitch, .zs-rise { animation: none !important; }
}
`;

/* ------------------------------------------------------------------ */
/*  Field definitions — one per feature the model was trained on      */
/* ------------------------------------------------------------------ */
const FIELDS = [
  {
    key: "RIDAGEYR", type: "range", label: "Age",
    note: "The single biggest factor in the model. Younger = better odds. Sorry, older folks — the data is what it is.",
    min: 18, max: 85, step: 1,
    fmt: (v) => (v >= 85 ? "85 or older" : `${v} years old`),
  },
  {
    key: "RIAGENDR", type: "choice", label: "Sex",
    options: [[1, "Male"], [2, "Female"]],
  },
  {
    key: "RIDRETH1", type: "choice", label: "Race and ethnicity",
    options: [
      [1, "Mexican American"], [2, "Other Hispanic"], [3, "White, non-Hispanic"],
      [4, "Black, non-Hispanic"], [5, "Other or multiracial"],
    ],
  },
  {
    key: "DMDEDUC2", type: "choice", label: "Highest level of school finished",
    options: [
      [1, "Below 9th grade"], [2, "Some high school"], [3, "High school or GED"],
      [4, "Some college"], [5, "College degree or higher"],
    ],
  },
  {
    key: "DMDMARTL", type: "choice", label: "Marital status",
    options: [
      [1, "Married"], [2, "Widowed"], [3, "Divorced"],
      [4, "Separated"], [5, "Never married"], [6, "Living with a partner"],
    ],
  },
  {
    key: "DMDCITZN", type: "choice", label: "US citizenship",
    options: [[1, "Citizen"], [2, "Not a citizen"]],
  },
  {
    key: "INDFMPIR", type: "range", label: "Family income against the poverty line",
    note: "Higher income = statistically better outcomes. Turns out money helps even before the apocalypse.",
    min: 0, max: 5, step: 0.1,
    fmt: (v) => (v >= 5 ? "5× the poverty line or more" : `${v.toFixed(1)}× the poverty line`),
  },
  {
    key: "DMDHHSIZ", type: "range", label: "People living in your household",
    min: 1, max: 7, step: 1,
    fmt: (v) => (v >= 7 ? "7 or more people" : v === 1 ? "Just you" : `${v} people`),
  },
];

const DEFAULTS = {
  RIDAGEYR: 34, RIAGENDR: 1, RIDRETH1: 3, DMDEDUC2: 4,
  DMDMARTL: 5, INDFMPIR: 2.5, DMDCITZN: 1, DMDHHSIZ: 3,
};

/* Offline stand-in so the UI is usable before the API is up. Rough
   coefficients in the spirit of the real logistic fit — not the model. */
function offlineEstimate(p) {
  let z = -9.4
    + 0.105 * p.RIDAGEYR
    + 0.42 * (p.RIAGENDR === 1 ? 1 : 0)
    - 0.20 * p.INDFMPIR
    - 0.14 * (p.DMDEDUC2 - 1)
    - 0.03 * (p.DMDHHSIZ - 1);
  if (p.DMDMARTL === 2) z += 0.35;
  if (p.DMDMARTL === 3 || p.DMDMARTL === 4) z += 0.22;
  if (p.DMDCITZN === 2) z -= 0.30;
  if (p.RIDRETH1 === 4) z += 0.18;
  return (1 - 1 / (1 + Math.exp(-z))) * 100;
}

function verdictFor(pct) {
  if (pct >= 95) return ["You make it to the sequel", "Somehow you're fine. You find a fortified Costco on day two and never leave. Irritatingly, you thrive."];
  if (pct >= 85) return ["Solid survivor material", "You're the one who figures out the walkie-talkies, rations the canned beans, and gives the group pep talks. Annoying, but effective."];
  if (pct >= 70) return ["You'll last a while", "You survive the first outbreak, the second wave, and one very bad decision involving a shopping mall. After that it gets complicated."];
  if (pct >= 50) return ["It's a coin flip, honestly", "Half the people like you made it. The other half went back for their phone charger. The model is not sure which one you are."];
  if (pct >= 30) return ["The odds are not in your favour", "You're the character who gets a full backstory episode right before the midseason finale. You know what happens in the midseason finale."];
  return ["You are the cold open", "The camera opens on your street. There's a close-up of your face. The title card appears. You do not appear again."];
}

/* ------------------------------------------------------------------ */
/*  Torn-edge silhouettes for the hero panels                          */
/* ------------------------------------------------------------------ */
function Figure({ variant, inverted }) {
  const fg = inverted ? INK : BONE;
  const bg = inverted ? BONE : INK;
  const fid = `tear-${variant}-${inverted ? "i" : "n"}`;
  const hid = `dots-${variant}-${inverted ? "i" : "n"}`;

  const bodies = {
    0: ( // reaching arm, head low
      <g>
        <ellipse cx="34" cy="150" rx="21" ry="25" />
        <path d="M13 174 L57 172 L64 260 L8 260 Z" />
        <path d="M52 160 L74 96 L86 22 L96 26 L84 100 L64 170 Z" />
        <path d="M92 12 L104 16 L100 30 L86 26 Z" />
      </g>
    ),
    1: ( // head close-up
      <g>
        <ellipse cx="50" cy="108" rx="44" ry="54" />
        <path d="M6 150 L94 150 L100 260 L0 260 Z" />
        <ellipse cx="34" cy="100" rx="7" ry="9" fill={bg} />
        <ellipse cx="66" cy="98" rx="9" ry="11" fill={bg} />
        <path d="M30 132 L72 130 L70 142 L32 144 Z" fill={bg} />
      </g>
    ),
    2: ( // standing, head tilted
      <g>
        <ellipse cx="54" cy="52" rx="19" ry="23" />
        <path d="M32 76 L74 74 L82 168 L24 170 Z" />
        <path d="M32 84 L20 160 L12 210 L22 212 L32 162 L44 92 Z" />
        <path d="M74 82 L88 156 L92 206 L82 208 L74 158 Z" />
        <path d="M28 168 L52 168 L48 260 L34 260 Z" />
        <path d="M56 168 L80 168 L76 260 L62 260 Z" />
      </g>
    ),
    3: ( // figure in a doorway
      <g>
        <rect x="14" y="18" width="72" height="242" fill={fg} opacity="0.22" />
        <rect x="22" y="30" width="56" height="230" fill={bg} />
        <ellipse cx="50" cy="86" rx="15" ry="19" />
        <path d="M34 106 L66 106 L74 196 L26 196 Z" />
        <path d="M30 198 L48 198 L46 260 L32 260 Z" />
        <path d="M52 198 L70 198 L68 260 L54 260 Z" />
      </g>
    ),
    4: ( // long hair, facing out
      <g>
        <path d="M22 62 Q50 24 78 62 L84 168 L70 172 L66 96 L34 96 L30 172 L16 168 Z" />
        <ellipse cx="50" cy="76" rx="22" ry="27" />
        <path d="M26 122 L74 122 L84 260 L16 260 Z" />
        <ellipse cx="41" cy="72" rx="5" ry="7" fill={bg} />
        <ellipse cx="60" cy="72" rx="5" ry="7" fill={bg} />
      </g>
    ),
  };

  return (
    <svg viewBox="0 0 100 260" preserveAspectRatio="xMidYMid slice"
         style={{ display: "block", width: "100%", height: "100%", background: bg }}
         aria-hidden="true">
      <defs>
        <filter id={fid} x="-25%" y="-15%" width="150%" height="130%">
          <feTurbulence type="fractalNoise" baseFrequency="0.045 0.09" numOctaves="4" seed={variant * 7 + 3} result="n" />
          <feDisplacementMap in="SourceGraphic" in2="n" scale="11" xChannelSelector="R" yChannelSelector="G" />
        </filter>
        <pattern id={hid} width="5" height="5" patternUnits="userSpaceOnUse">
          <rect width="5" height="5" fill="none" />
          <circle cx="2.5" cy="2.5" r="1.5" fill={fg} />
        </pattern>
      </defs>
      <g fill={fg} filter={`url(#${fid})`}>{bodies[variant]}</g>
      <rect x="0" y="196" width="100" height="64" fill={`url(#${hid})`} opacity="0.30" />
      <rect x="0" y="0" width="100" height="34" fill={`url(#${hid})`} opacity="0.20" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Hero / loader                                                      */
/* ------------------------------------------------------------------ */
const BOOT_LINES = [
  "Raiding 20 years of government death records...",
  "Teaching a machine what kills people (for science)...",
  "Filtering out people who said 'none of your business'...",
  "Asking XGBoost very nicely to predict your doom...",
  "Your fate has been computed. Probably.",
];

function Hero({ onStart, serverState }) {
  const reduce = useRef(
    typeof window !== "undefined" &&
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
  ).current;

  const [step, setStep] = useState(reduce ? 6 : 0);

  useEffect(() => {
    if (reduce) return;
    const timers = [];
    for (let i = 1; i <= 6; i++) timers.push(setTimeout(() => setStep(i), 330 * i));
    return () => timers.forEach(clearTimeout);
  }, [reduce]);

  const ready = step >= 6;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", justifyContent: "center", padding: "28px 20px" }}>
      <div style={{ width: "100%", maxWidth: 1040, margin: "0 auto" }}>

        <div style={{
          display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 4,
          height: "clamp(220px, 38vh, 340px)", border: `1px solid ${BONE}`, padding: 4,
        }}>
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className={step > i ? "zs-panel" : undefined}
                 style={{ overflow: "hidden", background: INK, visibility: step > i ? "visible" : "hidden" }}>
              <Figure variant={i} inverted={i === 1 || i === 4} />
            </div>
          ))}
        </div>

        <h1 className={ready ? "zs-title" : undefined}
            style={{
              fontFamily: DISPLAY, fontSize: "clamp(58px, 15.5vw, 186px)", lineHeight: 0.82,
              margin: "10px 0 0", letterSpacing: "-0.01em", textAlign: "center",
              visibility: ready ? "visible" : "hidden", transformOrigin: "center top",
            }}>
          <span className="zs-twitch" style={{ display: "inline-block" }}>WILL YOU SURVIVE?</span>
        </h1>

        <div style={{
          marginTop: 18, display: "flex", flexWrap: "wrap", gap: "18px 40px",
          alignItems: "flex-end", justifyContent: "space-between",
        }}>
          <p style={{ margin: 0, maxWidth: "48ch", fontSize: 15, lineHeight: 1.55, color: BONE }}>
            The zombie apocalypse is here. Humanity is falling. And you're standing
            there wondering: <em>do I have what it takes?</em> Answer 8 questions about
            yourself. We feed them to a machine learning model trained on 20 years of
            real US mortality data.
            {" "}
            <span style={{ color: ASH }}>It will tell you, with deeply misplaced confidence, whether you make it out alive.</span>
          </p>

          <div style={{ minWidth: 220 }}>
            <p style={{ margin: "0 0 12px", fontSize: 12, color: ASH, minHeight: 17 }}>
              {reduce ? BOOT_LINES[4] : BOOT_LINES[Math.min(step, 4)]}
              {serverState === "offline" && ready ? " · running offline" : ""}
            </p>
            <button className="zs-cta" onClick={onStart} disabled={!ready}>
              {ready ? "Find out if I survive" : "Reanimating..."}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Intake                                                             */
/* ------------------------------------------------------------------ */
function ChoiceField({ field, value, onChange }) {
  return (
    <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
      <legend style={{ fontSize: 16, fontWeight: 700, padding: 0, marginBottom: 10 }}>{field.label}</legend>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {field.options.map(([code, text]) => (
          <button key={code} type="button" className="zs-chip"
                  data-on={value === code}
                  aria-pressed={value === code}
                  onClick={() => onChange(field.key, code)}>
            {text}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function RangeField({ field, value, onChange }) {
  return (
    <div>
      <label htmlFor={field.key} style={{ fontSize: 16, fontWeight: 700, display: "block", marginBottom: 6 }}>
        {field.label}
      </label>
      <div style={{ fontFamily: DISPLAY, fontSize: 30, lineHeight: 1, marginBottom: 2 }}>
        {field.fmt(value)}
      </div>
      <input id={field.key} className="zs-range" type="range"
             min={field.min} max={field.max} step={field.step} value={value}
             onChange={(e) => onChange(field.key, parseFloat(e.target.value))} />
      {field.note && <p style={{ margin: "2px 0 0", fontSize: 12, color: ASH, maxWidth: "52ch" }}>{field.note}</p>}
    </div>
  );
}

function Intake({ person, setField, onSubmit, busy, error, onBack }) {
  return (
    <div style={{ minHeight: "100vh", padding: "26px 20px 64px" }}>
      <div style={{ width: "100%", maxWidth: 1040, margin: "0 auto" }}>

        <header style={{
          display: "flex", alignItems: "baseline", justifyContent: "space-between",
          gap: 20, flexWrap: "wrap", borderBottom: `1px solid ${BONE}`, paddingBottom: 14,
        }}>
          <h2 style={{ fontFamily: DISPLAY, fontSize: "clamp(34px, 7vw, 62px)", margin: 0, lineHeight: 0.9 }}>
            The horde needs your details
          </h2>
          <button className="zs-ghost" onClick={onBack}>← Run away</button>
        </header>

        <div style={{
          display: "grid", gap: 34, marginTop: 30,
          gridTemplateColumns: "repeat(auto-fit, minmax(310px, 1fr))",
        }}>
          {FIELDS.map((f) =>
            f.type === "choice"
              ? <ChoiceField key={f.key} field={f} value={person[f.key]} onChange={setField} />
              : <RangeField key={f.key} field={f} value={person[f.key]} onChange={setField} />
          )}
        </div>

        <div style={{ marginTop: 42, borderTop: `1px solid ${BONE}`, paddingTop: 22 }}>
          <button className="zs-cta" onClick={onSubmit} disabled={busy}>
            {busy ? "Consulting the dead records..." : "Calculate my survival odds"}
          </button>
          {error && (
            <p style={{ marginTop: 14, fontSize: 13, color: BLOOD, maxWidth: "62ch" }}>{error}</p>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Result                                                             */
/* ------------------------------------------------------------------ */
function Result({ pct, source, onRedo }) {
  const [shown, setShown] = useState(0);
  const reduce = useRef(
    typeof window !== "undefined" &&
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
  ).current;

  useEffect(() => {
    if (reduce) { setShown(pct); return; }
    let raf, start;
    const tick = (t) => {
      if (!start) start = t;
      const k = Math.min((t - start) / 900, 1);
      setShown(pct * (1 - Math.pow(1 - k, 3)));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pct, reduce]);

  const [headline, line] = verdictFor(pct);

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", padding: "40px 20px" }}>
      <div className="zs-rise" style={{ width: "100%", maxWidth: 1040, margin: "0 auto" }}>

        <p style={{ margin: 0, fontSize: 15, color: ASH }}>Your chance of surviving the zombie apocalypse</p>

        <div style={{ display: "flex", alignItems: "flex-start", gap: 4, lineHeight: 0.78 }}>
          <span style={{ fontFamily: DISPLAY, fontSize: "clamp(110px, 30vw, 300px)", color: BLOOD }}>
            {shown.toFixed(1)}
          </span>
          <span style={{ fontFamily: DISPLAY, fontSize: "clamp(40px, 9vw, 92px)", color: BLOOD, marginTop: "0.18em" }}>
            %
          </span>
        </div>

        <h2 style={{ fontFamily: DISPLAY, fontSize: "clamp(30px, 6.5vw, 58px)", margin: "14px 0 8px", lineHeight: 0.92 }}>
          {headline}
        </h2>
        <p style={{ margin: 0, fontSize: 16, maxWidth: "46ch", lineHeight: 1.5 }}>{line}</p>

        <div style={{ marginTop: 34, display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="zs-cta" onClick={onRedo}>Try lying this time</button>
        </div>

        <p style={{ marginTop: 34, fontSize: 12.5, color: ASH, maxWidth: "64ch", lineHeight: 1.6 }}>
          The fine print: this model was actually trained on real US mortality data
          (NHANES 1999–2018, CDC linked mortality follow-up). It predicts the statistical
          likelihood of dying before 2019 based purely on demographics.
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
export default function App() {
  const [stage, setStage] = useState("hero");
  const [person, setPerson] = useState(DEFAULTS);
  const [pct, setPct] = useState(null);
  const [source, setSource] = useState("api");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const setField = (key, value) => setPerson((p) => ({ ...p, [key]: value }));

  async function submit() {
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(person),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setPct(data.survival_percentage);
      setSource("api");
      setStage("result");
    } catch (e) {
      setPct(offlineEstimate(person));
      setSource("offline");
      setError(
        "Could not reach the model server, so this is the browser's rough fallback. " +
        "Start the API with: uvicorn app.main:app --reload"
      );
      setStage("result");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="zs-root">
      <style>{CSS}</style>
      {stage === "hero" && <Hero onStart={() => setStage("intake")} serverState={source} />}
      {stage === "intake" && (
        <Intake person={person} setField={setField} onSubmit={submit}
                busy={busy} error={error} onBack={() => setStage("hero")} />
      )}
      {stage === "result" && pct !== null && (
        <Result pct={pct} source={source} onRedo={() => setStage("intake")} />
      )}
    </div>
  );
}