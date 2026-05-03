import React from "react";

export default function ResultsCard({ result, history, fallbackGc }) {
  return (
    <section className="card">
      <h2>Results</h2>
      {!result ? (
        <div className="empty">
          <div className="emptyTitle">No prediction yet</div>
          <div className="muted">Run a prediction to see the score, interpretation, and details.</div>
        </div>
      ) : (
        <>
          <div className="resultHero">
            <div>
              <div className="k">Predicted efficiency</div>
              <div className="big">{result.prediction.toFixed(4)}</div>
            </div>
            <span
              className={`pill ${
                result.band.tone === "good" ? "pill-good" : result.band.tone === "mid" ? "pill-mid" : "pill-bad"
              }`}
            >
              {result.band.label} expected cutting
            </span>
          </div>

          <div className="resultCards">
            <div className="miniCard">
              <div className="k">Features used</div>
              <div className="v">{(result.features?.gc_content ?? fallbackGc).toFixed(2)} GC</div>
              <div className="mutedSmall">Baseline uses GC only</div>
            </div>
            <div className="miniCard">
              <div className="k">Model</div>
              <div className="v">{result.model}</div>
              <div className="mutedSmall">Linear regression</div>
            </div>
          </div>

          <div className="divider" />

          <div className="k">Recent predictions</div>
          <div className="table">
            <div className="thead">
              <span>Sequence</span>
              <span>GC</span>
              <span>Score</span>
            </div>
            {history.map((h) => (
              <div className="trow" key={h.ts}>
                <code className="seq">{h.sequence}</code>
                <span>{(h.features?.gc_content ?? 0).toFixed(2)}</span>
                <span className="score">{h.prediction.toFixed(4)}</span>
              </div>
            ))}
          </div>

          <div className="mutedSmall" style={{ marginTop: 10 }}>
            Next upgrade: replace “Low/Medium/High” with dataset percentiles + add richer sequence features.
          </div>
        </>
      )}
    </section>
  );
}