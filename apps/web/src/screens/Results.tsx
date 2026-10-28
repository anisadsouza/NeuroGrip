/** Recorded experiment results, kept separate from the live control screen. */

import { useEffect, useState } from 'react';

interface ModelResult {
  readonly mean_accuracy: number;
  readonly std_accuracy: number;
  readonly expected_calibration_error: number;
}

interface Comparison {
  readonly source: string;
  readonly best_model: string;
  readonly models: Record<string, ModelResult>;
  readonly caveat: string;
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

async function readComparison(url: string): Promise<Comparison | null> {
  const response = await fetch(url);
  return response.ok ? response.json() as Promise<Comparison> : null;
}

export function Results() {
  const [simulated, setSimulated] = useState<Comparison | null>(null);
  const [recorded, setRecorded] = useState<Comparison | null>(null);

  useEffect(() => {
    void Promise.all([
      readComparison(`${import.meta.env.BASE_URL}results/model_comparison.json`),
      readComparison(`${import.meta.env.BASE_URL}results/model_comparison_ninapro.json`),
    ]).then(([simulation, ninaPro]) => {
      setSimulated(simulation);
      setRecorded(ninaPro);
    });
  }, []);

  return (
    <section className="results" aria-labelledby="results-title">
      <header className="results-head">
        <h2 id="results-title">Evaluation results</h2>
        <p className="muted">Simulator and recorded-data results are deliberately never merged.</p>
      </header>

      <ResultTable title="Simulated sEMG" result={simulated} />
      <ResultTable title="Recorded NinaPro DB2" result={recorded} />
    </section>
  );
}

function ResultTable({ title, result }: { title: string; result: Comparison | null }) {
  return (
    <section className="panel results-table" aria-labelledby={`${title}-heading`}>
      <h3 id={`${title}-heading`}>{title}</h3>
      {result === null ? (
        <p className="muted">
          No artifact staged yet. Run the corresponding experiment, then
          <code> npm run build:assets</code> to make it available here.
        </p>
      ) : (
        <>
          <table>
            <caption>{result.source} · selected model: {result.best_model}</caption>
            <thead><tr><th>Model</th><th>LOSO accuracy</th><th>SD</th><th>ECE</th></tr></thead>
            <tbody>
              {Object.entries(result.models).map(([name, model]) => (
                <tr key={name} data-selected={name === result.best_model || undefined}>
                  <th scope="row">{name}</th>
                  <td>{percentage(model.mean_accuracy)}</td>
                  <td>{model.std_accuracy.toFixed(3)}</td>
                  <td>{model.expected_calibration_error.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small muted">{result.caveat}</p>
        </>
      )}
    </section>
  );
}
