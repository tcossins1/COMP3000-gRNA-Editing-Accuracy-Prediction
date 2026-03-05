import React, { useMemo } from "react";
import { validateSequence, normalize } from "../utils/sequence";

export default function GuideInputCard({
  sequence,
  setSequence,
  onPredict,
  loading,
  error,
}) {
  const v = useMemo(() => validateSequence(sequence), [sequence]);
  const seqNorm = v.ok ? v.seq : normalize(sequence);

  return (
    <section className="card">
      <h2>Guide input</h2>
      <p className="muted">
        Enter a 20-nt guide (A/T/G/C).
      </p>

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

      <button className="primary" onClick={onPredict} disabled={!v.ok || loading}>
        {loading ? "Predicting…" : "Predict"}
      </button>

      {error && <div className="alert">{error}</div>}
    </section>
  );
}