export function bandFromPrediction(pred) {
  if (pred >= 1.5) return { label: "High", tone: "good" };
  if (pred >= 0.5) return { label: "Medium", tone: "mid" };
  return { label: "Low", tone: "bad" };
}