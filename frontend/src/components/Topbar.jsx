import React from "react";

export default function Topbar({ apiOnline }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="logo">FC9</div>
        <div>
          <div className="title">ForeCas9</div>
          <div className="subtitle">gRNA Cutting Efficiency Predictor</div>
        </div>
      </div>

      <div className="status">
        <span className={`pill ${apiOnline === false ? "pill-bad" : apiOnline === true ? "pill-good" : "pill-mid"}`}>
          {apiOnline === true ? "API Online" : apiOnline === false ? "API Offline" : "Checking API…"}
        </span>
        <span className="pill pill-mid">Baseline model</span>
      </div>
    </header>
  );
}