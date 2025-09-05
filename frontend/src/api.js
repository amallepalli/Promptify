import axios from "axios";

// In dev we use a Vite proxy so the frontend calls "/api/*"
export const api = axios.create({
  baseURL: "/api",
  timeout: 60000,
});

export async function health() {
  const r = await api.get("/health");
  return r.data;
}

export async function classify(prompt) {
  const r = await api.post("/classify", { prompt });
  return r.data; // { label: "cot" | "roleplay" }
}

export async function generate({ prompt, adapter_override = null }) {
  const r = await api.post("/generate", { prompt, adapter_override });
  return r.data; // { adapter, output }
}
