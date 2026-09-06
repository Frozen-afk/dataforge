// API client for the Latent Loop Lab backend.
//
// Every call returns data that carries its own `evidence` object. Pages render
// that badge rather than hard-coding one, so a precomputed sweep can never be
// displayed as if it were a live measurement.

export const API_BASE =
  import.meta.env?.VITE_API_BASE || "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (cause) {
    throw new ApiError(
      `Could not reach the backend at ${API_BASE}. Start it with ` +
        `"uvicorn server:app --reload" from the Backend directory.`,
      0
    );
  }

  if (!response.ok) {
    let detail = `Backend returned ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // Keep the status-code message.
    }
    throw new ApiError(detail, response.status);
  }

  return response.json();
}

export const api = {
  health: () => request("/health"),
  presets: () => request("/presets"),

  exactPreset: (preset, R, alpha = 1) =>
    request(`/exact/preset/${preset}/${R}?alpha=${alpha}`),

  exactRun: (body) =>
    request("/exact/run", { method: "POST", body: JSON.stringify(body) }),

  exactGenerate: (body) =>
    request("/exact/generate", { method: "POST", body: JSON.stringify(body) }),

  trajectorySummary: (preset, R) =>
    request(`/exact/trajectory/summary?preset=${preset}&R=${R}`),

  learnedStatus: () => request("/learned/status"),
  learnedExperiment: () => request("/learned/experiment"),
  learnedExamples: () => request("/learned/examples"),

  learnedRun: (body) =>
    request("/learned/run", { method: "POST", body: JSON.stringify(body) }),

  learnedSweep: (body) =>
    request("/learned/sweep", { method: "POST", body: JSON.stringify(body) }),
};

export { ApiError };
