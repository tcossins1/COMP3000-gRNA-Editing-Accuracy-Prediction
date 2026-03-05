import React, { useMemo } from "react";
import { validateSequence, normalize, gcContent, composition } from "../utils/sequence";

export default function GuideInputCard({
  examples,
  sequence,
  setSequence,
  onPredict,
  loading,
  error,
  apiOnline,
}) {
  const v = useMemo(() => validateSequence(sequence), [sequence]);
  const seqNorm = v.ok ? v.seq : normalize(sequence);
  const gc = useMemo(() => (seqNorm.length ? gcContent(seqNorm) : 0), [seqNorm]);
  const comp = useMemo(() => composition(seqNorm), [seqNorm]);

  return (
    <section className="card">
      <h2>Guide input</h2>
      <p className="muted">
        Enter a 20-nt guide (A/T/G/C).
      </p>

      <div className="examples">
        {examples.map((ex) => (
          <button key={ex.label} className="chip" onClick={() => setSequence(ex.seq)} type="button">
            {ex.label}
          </button>
        ))}
      </div>

      <label className="label">20-nt gRNA sequence</label>
      <div className={`inputWrap ${v.ok ? "ok" : sequence.trim() ? "bad" : ""}`}>
        <input
          className="input"
          value={sequence}
          onChange={(e) => setSequence(e.target.value)}
          placeholder="e.g., ATGCGTAGCTAAGCTAGCAC"
          spellCheck="false"
          autoCapitalize="characters"
        />
        <div className="len">{seqNorm.length}/20</div>
      </div>

      {!v.ok && sequence.trim() && <div className="hint bad">{v.msg}</div>}
      {v.ok && <div className="hint good">Looks valid ✓</div>}

      <div className="meter">
        <div className="meterTop">
          <div className="k">GC content</div>
          <div className="v">{gc.toFixed(2)}</div>
        </div>
        <div className="bar">
          <div className="fill" style={{ width: `${Math.round(gc * 100)}%` }} />
        </div>
        <div className="miniRow">
          <span>A: {comp.A}</span>
          <span>T: {comp.T}</span>
          <span>G: {comp.G}</span>
          <span>C: {comp.C}</span>
        </div>
      </div>

      <button className="primary" onClick={onPredict} disabled={!v.ok || loading}>
        {loading ? "Predicting…" : "Predict"}
      </button>

      {error && <div className="alert">{error}</div>}

      <div className="divider" />

      <div className="smallGrid">
        <div>
          <div className="k">Status</div>
          <div className="v2">{apiOnline === true ? "Online" : apiOnline === false ? "Offline" : "Checking…"}</div>
        </div>
      </div>
    </section>
  );
}