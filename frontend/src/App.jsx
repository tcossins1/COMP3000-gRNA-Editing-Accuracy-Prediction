import React, { useMemo, useState } from "react";
import "./styles.css";

import Topbar from "./components/Topbar";
import GuideInputCard from "./components/GuideInputCard";
import ResultsCard from "./components/ResultsCard";

import { useApiHealth } from "./hooks/useApiHealth";
import { predictGuide } from "./api/forecas9";

import { validateSequence, normalize, gcContent } from "./utils/sequence";
import { bandFromPrediction } from "./utils/scoring";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const EXAMPLES = [
  { label: "Example (balanced)", seq: "TATGACGTAACTGACTGATC" },
  { label: "Example (high GC)", seq: "GGGCGGCGGCGGCGGCGGCG" },
  { label: "Example (low GC)", seq: "ATATATATATATATATATAT" },
];

export default function App() {
  const [sequence, setSequence] = useState(EXAMPLES[0].seq);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const [apiOnline, setApiOnline] = useApiHealth(API_BASE);

  const v = useMemo(() => validateSequence(sequence), [sequence]);
  const seqNorm = v.ok ? v.seq : normalize(sequence);
  const fallbackGc = useMemo(() => (seqNorm.length ? gcContent(seqNorm) : 0), [seqNorm]);

  async function onPredict() {
    setError("");
    setResult(null);

    const vv = validateSequence(sequence);
    if (!vv.ok) return setError(vv.msg);
    if (apiOnline === false) return setError("API appears offline. Start the FastAPI server and try again.");

    setLoading(true);
    try {
      const data = await predictGuide(vv.seq, { baseUrl: API_BASE });

      const pred = data.prediction;
      const newResult = {
        sequence: data.sequence ?? vv.seq,
        prediction: pred,
        features: data.features ?? { gc_content: fallbackGc },
        model: data.model ?? "baseline",
        band: bandFromPrediction(pred),
        ts: new Date().toISOString(),
      };

      setResult(newResult);
      setHistory((h) => [newResult, ...h].slice(0, 8));
    } catch (e) {
      setError(e?.message || "Prediction failed.");
      if ((e?.message || "").toLowerCase().includes("could not reach")) {
        setApiOnline(false);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <Topbar apiOnline={apiOnline} />

      <main className="grid">
        <GuideInputCard
          API_BASE={API_BASE}
          examples={EXAMPLES}
          sequence={sequence}
          setSequence={setSequence}
          onPredict={onPredict}
          loading={loading}
          error={error}
          apiOnline={apiOnline}
        />

        <ResultsCard result={result} history={history} fallbackGc={fallbackGc} />
      </main>

      <footer className="footer">
        <span className="mutedSmall">ForeCas9 • React + FastAPI • Baseline predictor</span>
      </footer>
    </div>
  );
}