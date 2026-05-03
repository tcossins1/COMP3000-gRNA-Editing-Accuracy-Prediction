import { requestJson } from "./client";

/** GET /health */
export function getHealth({ baseUrl } = {}) {
  return requestJson("/health", { baseUrl });
}

/** POST /predict */
export function predictGuide(sequence, { baseUrl } = {}) {
  return requestJson("/predict", {
    baseUrl,
    method: "POST",
    body: { sequence },
  });
}