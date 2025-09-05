import { useEffect, useState } from "react";
import { health, generate } from "./api";
import "./App.css";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [adapter, setAdapter] = useState(null);
  const [override, setOverride] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [error, setError] = useState(null);

  useEffect(() => {
    health().then(() => setApiStatus("ok")).catch(() => setApiStatus("down"));
  }, []);

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setOutput("");
    setAdapter(null);
    try {
      const res = await generate({ prompt, adapter_override: override || null });
      setAdapter(res.adapter);
      setOutput(res.output);
    } catch (e) {
      setError(e?.message ?? "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Promptify</h1>
      <div className="status">
        API status:{" "}
        <b className={apiStatus === "ok" ? "ok" : apiStatus === "down" ? "down" : ""}>
          {apiStatus}
        </b>
      </div>

      <form onSubmit={onSubmit}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt…"
          required
        />
        <div className="row">
          <label>Adapter override:</label>
          <select value={override} onChange={(e) => setOverride(e.target.value)}>
            <option value="">auto (use classifier)</option>
            <option value="cot">cot</option>
            <option value="roleplay">roleplay</option>
          </select>
          <button type="submit" disabled={loading}>
            {loading ? "Generating…" : "Generate"}
          </button>
        </div>
      </form>

      {adapter && <div className="info"><b>Adapter selected:</b> {adapter}</div>}
      {error && <div className="error"><b>Error:</b> {error}</div>}

      {output && (
        <div className="output">
          <h2>Output</h2>
          <pre>{output}</pre>
        </div>
      )}
    </div>
  );
}
