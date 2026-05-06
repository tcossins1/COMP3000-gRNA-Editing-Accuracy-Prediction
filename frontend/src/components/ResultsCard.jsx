import React from "react";
import FeatureBreakdown from "./FeatureBreakdown";

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
              <div className="k">Efficiency band</div>
              <div className="big">{result.band.label}</div>
              <div className="mutedSmall">Score: {result.prediction.toFixed(4)}</div>
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
              <div className="k">Sequence</div>
              <div className="v2" style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>
                {result.sequence}
              </div>
              <div className="mutedSmall">20-nt guide</div>
            </div>
          </div>

          <div style={{ marginTop: 10 }}>
            <FeatureBreakdown features={result.features} fallbackGc={fallbackGc} />
          </div>

          <div className="divider" />

          <div className="k">Recent predictions</div>
          <div className="table">
            <div className="thead">
              <span>Sequence</span>
              <span>Band</span>
              <span>Score</span>
            </div>
            {history.map((h) => (
              <div className="trow" key={h.ts}>
                <code className="seq">{h.sequence}</code>
                <span>{h.band.label}</span>
                <span className="score">{h.prediction.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}