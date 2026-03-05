import React, { useMemo, useState } from "react";
import "./styles.css";

import Topbar from "./components/Topbar";
import GuideInputCard from "./components/GuideInputCard";
import ResultsCard from "./components/ResultsCard";

import { predictGuide } from "./api/forecas9";

import { validateSequence, normalize, gcContent } from "./utils/sequence";
import { bandFromPrediction } from "./utils/scoring";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [sequence, setSequence] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const v = useMemo(() => validateSequence(sequence), [sequence]);
  const seqNorm = v.ok ? v.seq : normalize(sequence);
  const fallbackGc = useMemo(() => (seqNorm.length ? gcContent(seqNorm) : 0), [seqNorm]);

  async function onPredict() {
    setError("");
    setResult(null);

    const vv = validateSequence(sequence);
    if (!vv.ok) return setError(vv.msg);

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
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <Topbar />

      <main className="grid">
        <GuideInputCard
          API_BASE={API_BASE}
          sequence={sequence}
          setSequence={setSequence}
          onPredict={onPredict}
          loading={loading}
          error={error}
        />

        <ResultsCard result={result} history={history} fallbackGc={fallbackGc} />
      </main>
    </div>
  );
}