import React from "react";

function clamp01(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function BarRow({ label, value }) {
  const v = clamp01(value);
  return (
    <div className="fbRow">
      <div className="fbRowTop">
        <span className="k">{label}</span>
        <span className="v">{v.toFixed(2)}</span>
      </div>
      <div className="bar fbBar">
        <div className="fill" style={{ width: `${Math.round(v * 100)}%` }} />
      </div>
    </div>
  );
}

function RiskChip({ label, tone }) {
  return (
    <span className={`pill ${tone === "good" ? "pill-good" : tone === "mid" ? "pill-mid" : "pill-bad"}`}>
      {label}
    </span>
  );
}

export default function FeatureBreakdown({ features, fallbackGc = 0 }) {
  const f = features || {};

  const gc = f.gc_content ?? fallbackGc;
  const gc1 = f.gc_1_10 ?? null;
  const gc2 = f.gc_11_20 ?? null;

  const polyT = Number(f.has_poly_t4 ?? 0) === 1;
  const maxH = Number(f.max_homopolymer ?? 0);

  // Simple UX banding (not "truth", just a helpful visual)
  const homopolymerTone = maxH >= 5 ? "bad" : maxH >= 3 ? "mid" : "good";

  return (
    <div className="miniCard fbCard">
      <div className="fbHeader">
        <div>
          <div className="k">Feature breakdown</div>
          <div className="mutedSmall">Interpretable guide diagnostics</div>
        </div>
        <div className="fbChips">
          <RiskChip
            label={polyT ? "Poly-T (TTTT) detected" : "No Poly-T (TTTT)"}
            tone={polyT ? "bad" : "good"}
          />
          <RiskChip
            label={`Max homopolymer: ${maxH || 0}`}
            tone={homopolymerTone}
          />
        </div>
      </div>

      <div className="divider" />

      <div className="fbGrid">
        <BarRow label="GC content (overall)" value={gc} />
        {gc1 !== null && <BarRow label="GC positions 1–10" value={gc1} />}
        {gc2 !== null && <BarRow label="GC positions 11–20" value={gc2} />}
      </div>

      <div className="divider" />

      <div className="fbNucComp">
        <div className="k">Nucleotide composition</div>
        <div className="nucBars">
          <div className="nucBar">
            <span className="nucLabel">A</span>
            <div className="bar nucBarFill">
              <div className="fill nucA" style={{ width: `${Math.round((f.a_count || 0) / 20 * 100)}%` }} />
            </div>
            <span className="nucCount">{f.a_count || 0}</span>
          </div>
          <div className="nucBar">
            <span className="nucLabel">T</span>
            <div className="bar nucBarFill">
              <div className="fill nucT" style={{ width: `${Math.round((f.t_count || 0) / 20 * 100)}%` }} />
            </div>
            <span className="nucCount">{f.t_count || 0}</span>
          </div>
          <div className="nucBar">
            <span className="nucLabel">G</span>
            <div className="bar nucBarFill">
              <div className="fill nucG" style={{ width: `${Math.round((f.g_count || 0) / 20 * 100)}%` }} />
            </div>
            <span className="nucCount">{f.g_count || 0}</span>
          </div>
          <div className="nucBar">
            <span className="nucLabel">C</span>
            <div className="bar nucBarFill">
              <div className="fill nucC" style={{ width: `${Math.round((f.c_count || 0) / 20 * 100)}%` }} />
            </div>
            <span className="nucCount">{f.c_count || 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
}