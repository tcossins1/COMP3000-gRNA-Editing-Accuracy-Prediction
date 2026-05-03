export const VALID = new Set(["A", "T", "G", "C"]);

export function normalize(raw) {
  return (raw || "").trim().toUpperCase().replace(/\s+/g, "");
}

export function validateSequence(raw) {
  const seq = normalize(raw);
  if (!seq) return { ok: false, msg: "Enter a 20-nt guide sequence." };
  if (seq.length !== 20) return { ok: false, msg: "Sequence must be exactly 20 nucleotides." };
  for (const ch of seq) if (!VALID.has(ch)) return { ok: false, msg: "Only A, T, G, C are allowed." };
  return { ok: true, seq };
}

export function gcContent(seq) {
  if (!seq) return 0;
  let gc = 0;
  for (const ch of seq) if (ch === "G" || ch === "C") gc++;
  return gc / seq.length;
}

export function composition(seq) {
  const out = { A: 0, T: 0, G: 0, C: 0 };
  for (const ch of seq) if (out[ch] !== undefined) out[ch]++;
  return out;
}