import React, { useState } from "react";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [improvedPrompt, setImprovedPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setImprovedPrompt("");

    try {
      const response = await fetch("http://localhost:8000/optimize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch optimized prompt");
      }

      const data = await response.json();
      setImprovedPrompt(data.improved_prompt);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-6">
      <div className="bg-white shadow-xl rounded-2xl p-6 w-full max-w-lg">
        <h1 className="text-2xl font-bold mb-4 text-center">
          Prompt Optimizer
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here..."
            className="w-full p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={5}
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-xl hover:bg-blue-700 transition disabled:opacity-50"
          >
            {loading ? "Optimizing..." : "Optimize Prompt"}
          </button>
        </form>

        {error && (
          <p className="text-red-500 mt-4 text-center font-medium">{error}</p>
        )}

        {improvedPrompt && (
          <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-xl">
            <h2 className="text-lg font-semibold mb-2">Improved Prompt:</h2>
            <p className="text-gray-800 whitespace-pre-wrap">{improvedPrompt}</p>
          </div>
        )}
      </div>
    </div>
  );
}
