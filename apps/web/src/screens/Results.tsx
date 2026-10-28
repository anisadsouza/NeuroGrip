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

interface DecodedRun {
  readonly gesture: string;
  readonly windows: number;
}

interface TimelineEvent {
  readonly recording: string;
  readonly repetition: number;
  readonly performed: string;
  readonly start_ms: number;
  readonly end_ms: number;
  readonly windows: number;
  readonly decoded: readonly DecodedRun[];
}

interface HeldoutEvaluation {
  readonly source: string;
  readonly model: string;
  readonly training_subjects: readonly number[];
  readonly test_subject: number;
  readonly n_test: number;
  readonly gesture_order: readonly string[];
  readonly dataset_movements: Readonly<Record<string, string>>;
  readonly accuracy: number;
  readonly expected_calibration_error: number;
  readonly confusion_matrix: readonly (readonly number[])[];
  readonly quality_gate: {
    readonly windows_tested: number;
    readonly accepted: number;
    readonly rejected: number;
    readonly acceptance_rate: number;
    readonly top1_accuracy_before_gate: number;
    readonly top1_accuracy_on_accepted: number | null;
    readonly rejected_by_performed_gesture: Readonly<Record<string, number>>;
    readonly method: string;
  };
  readonly latency: {
    readonly runtime: string;
    readonly n_windows: number;
    readonly scope: string;
    readonly p50_ms: number;
    readonly p95_ms: number;
    readonly p99_ms: number;
    readonly max_ms: number;
  };
  readonly timeline: readonly TimelineEvent[];
  readonly caveat: string;
}

interface BrowserLatency {
  readonly n: number;
  readonly windowsProcessed?: number;
  readonly rejectedWindows?: number;
  readonly stablePredictionWindows?: number;
  readonly correctStablePredictions?: number;
  readonly stablePredictionAccuracy?: number | null;
  readonly zeroSignalRejected?: boolean;
  readonly threaded: boolean;
  readonly backend: string;
  readonly p95Ms: number;
  readonly note: string;
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function headingId(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') + '-heading';
}

async function readOptionalJson<T>(url: string): Promise<T | null> {
  const response = await fetch(url);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  if (response.headers.get('content-type')?.includes('text/html')) {
    // Vite preview and some static hosts serve index.html with HTTP 200 for a
    // missing optional public artifact instead of returning a 404.
    return null;
  }
  return response.json() as Promise<T>;
}

export function Results() {
  const [simulated, setSimulated] = useState<Comparison | null>(null);
  const [recorded, setRecorded] = useState<Comparison | null>(null);
  const [heldout, setHeldout] = useState<HeldoutEvaluation | null>(null);
  const [simulatedLatency, setSimulatedLatency] = useState<BrowserLatency | null>(null);
  const [recordedLatency, setRecordedLatency] = useState<BrowserLatency | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([
      readOptionalJson<Comparison>(`${import.meta.env.BASE_URL}results/model_comparison.json`),
      readOptionalJson<Comparison>(`${import.meta.env.BASE_URL}results/model_comparison_ninapro.json`),
      readOptionalJson<HeldoutEvaluation>(`${import.meta.env.BASE_URL}results/ninapro_holdout.json`),
      readOptionalJson<BrowserLatency>(`${import.meta.env.BASE_URL}results/browser_latency.json`),
      readOptionalJson<BrowserLatency>(`${import.meta.env.BASE_URL}results/browser_latency_ninapro.json`),
    ]).then(([simulation, ninaPro, evaluation, simulationLatency, ninaProLatency]) => {
      setSimulated(simulation);
      setRecorded(ninaPro);
      setHeldout(evaluation);
      setSimulatedLatency(simulationLatency);
      setRecordedLatency(ninaProLatency);
    }).catch((error: unknown) => {
      setLoadError(error instanceof Error ? error.message : String(error));
    });
  }, []);

  return (
    <section className="results" aria-labelledby="results-title">
      <header className="results-head">
        <h2 id="results-title">Recorded-data results</h2>
        <p className="muted">
          NinaPro DB2 · held-out subject evaluation. Simulator results are kept separate below.
        </p>
      </header>

      {loadError ? <p className="notice notice-stop" role="alert">{loadError}</p> : null}
      <HeldoutPanel result={heldout} browserLatency={recordedLatency} />
      <details className="results-details">
        <summary>Model comparison and simulator results</summary>
        <div className="results-detail-grid">
          <ResultTable title="Recorded NinaPro DB2 model comparison" result={recorded} />
          <ResultTable title="Simulated sEMG model comparison" result={simulated} />
          <LatencyPanel title="NinaPro browser diagnostics" result={recordedLatency} />
          <LatencyPanel title="Simulator browser diagnostics" result={simulatedLatency} />
        </div>
      </details>
    </section>
  );
}

function ResultTable({ title, result }: { title: string; result: Comparison | null }) {
  const id = headingId(title);
  return (
    <section className="panel results-table" aria-labelledby={id}>
      <h3 id={id}>{title}</h3>
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
          {result.source === 'ninapro_db2' ? (
            <p className="small muted">
              DB2 label 5 is finger abduction; it is an explicitly qualified proxy
              for the app's open-hand class, not an exact movement-name match.
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}

function LatencyPanel({
  title,
  result,
}: {
  title: string;
  result: BrowserLatency | null;
}) {
  const id = headingId(title);
  return (
    <section className="panel results-table" aria-labelledby={id}>
      <h3 id={id}>{title}</h3>
      {result ? (
        <>
          <dl className="readout-list">
            <div><dt>Worker P95</dt><dd className="ng-num">{result.p95Ms.toFixed(3)} ms</dd></div>
            <div>
              <dt>Measured windows</dt>
              <dd className="ng-num">
                {result.n}
                {typeof result.rejectedWindows === 'number'
                  ? ` decoded · ${result.rejectedWindows} rejected`
                  : ' decoded'}
              </dd>
            </div>
            <div><dt>Backend</dt><dd>{result.backend}{result.threaded ? ' · threads enabled' : ' · single-threaded'}</dd></div>
            {typeof result.stablePredictionAccuracy === 'number' &&
            typeof result.stablePredictionWindows === 'number' &&
            typeof result.correctStablePredictions === 'number' ? (
              <div>
                <dt>Stable replay top-1</dt>
                <dd className="ng-num">
                  {(result.stablePredictionAccuracy * 100).toFixed(1)}% ·{' '}
                  {result.correctStablePredictions}/{result.stablePredictionWindows}
                </dd>
              </div>
            ) : null}
            {typeof result.zeroSignalRejected === 'boolean' ? (
              <div>
                <dt>Zero-signal control</dt>
                <dd>{result.zeroSignalRejected ? 'rejected' : 'not rejected'}</dd>
              </div>
            ) : null}
          </dl>
          <p className="small muted">{result.note}</p>
        </>
      ) : (
        <p className="muted">No browser measurement staged. Run <code>npm run test:e2e</code>.</p>
      )}
    </section>
  );
}

function HeldoutPanel({
  result,
  browserLatency,
}: {
  result: HeldoutEvaluation | null;
  browserLatency: BrowserLatency | null;
}) {
  if (!result) {
    return (
      <section className="panel results-table" aria-labelledby="heldout-heading">
        <h3 id="heldout-heading">Subject-held-out NinaPro evaluation</h3>
        <p className="muted">
          No held-out evaluation is available. The model comparison below is not a
          held-out-subject test.
        </p>
      </section>
    );
  }

  const topErrors = result.confusion_matrix
    .flatMap((row, performedIndex) =>
      row.flatMap((count, decodedIndex) =>
        performedIndex === decodedIndex || count === 0
          ? []
          : [{
              performed: result.gesture_order[performedIndex]!,
              decoded: result.gesture_order[decodedIndex]!,
              count,
            }],
      ),
    )
    .sort((a, b) => b.count - a.count)
    .slice(0, 3);
  const stableAccuracy = browserLatency?.stablePredictionAccuracy;

  return (
    <section className="panel results-table" aria-labelledby="heldout-heading">
      <h3 id="heldout-heading">S{result.training_subjects.join(', S')} → S{result.test_subject}</h3>
      <p className="small muted">
        {result.model} · {result.n_test.toLocaleString()} labeled windows · exploratory
        development result
      </p>
      <dl className="results-metrics">
        <div>
          <dt>Held-out accuracy</dt>
          <dd>{percentage(result.accuracy)}</dd>
          <small>{result.n_test.toLocaleString()} windows</small>
        </div>
        <div>
          <dt>Stable replay top-1</dt>
          <dd>{typeof stableAccuracy === 'number' ? percentage(stableAccuracy) : '—'}</dd>
          <small>
            {browserLatency?.correctStablePredictions ?? 0}/
            {browserLatency?.stablePredictionWindows ?? 0} windows
          </small>
        </div>
        <div>
          <dt>Quality-gate accepted</dt>
          <dd>{percentage(result.quality_gate.acceptance_rate)}</dd>
          <small>
            {result.quality_gate.accepted.toLocaleString()} accepted ·{' '}
            {result.quality_gate.rejected.toLocaleString()} rejected
          </small>
        </div>
        <div>
          <dt>Browser worker P95</dt>
          <dd>{browserLatency ? `${browserLatency.p95Ms.toFixed(2)} ms` : '—'}</dd>
          <small>feature extraction + ONNX inference</small>
        </div>
        <div>
          <dt>Calibration error</dt>
          <dd>{result.expected_calibration_error.toFixed(3)}</dd>
          <small>lower is better</small>
        </div>
      </dl>
      <p className="results-caveat">
        Exploratory only: just S1 and S10 are available, and S10 informed diagnostics.
        This is not an untouched final test or clinical validation.
      </p>

      <section className="results-errors" aria-labelledby="top-errors-heading">
        <h4 id="top-errors-heading">Most common wrong predictions</h4>
        <ol>
          {topErrors.map((error) => (
            <li key={`${error.performed}-${error.decoded}`}>
              <span>
                {result.dataset_movements[error.performed] ?? error.performed}
                <span aria-hidden="true"> → </span>
                {result.dataset_movements[error.decoded] ?? error.decoded}
              </span>
              <strong>{error.count}</strong>
            </li>
          ))}
        </ol>
      </section>

      <details className="results-details">
        <summary>View confusion matrix</summary>
        <ConfusionMatrix result={result} />
      </details>
      <details className="results-details">
        <summary>View performed-versus-decoded timeline</summary>
        <Timeline events={result.timeline} />
      </details>
      <details className="results-details">
        <summary>Evaluation and quality-gate notes</summary>
        <div className="results-detail-copy">
          <p>
            CPU ONNX inference P95: {result.latency.p95_ms.toFixed(2)} ms across{' '}
            {result.latency.n_windows} windows.
          </p>
          <p>{result.quality_gate.method}</p>
          <p>
            Rejected performed labels:{' '}
            {Object.entries(result.quality_gate.rejected_by_performed_gesture)
              .filter(([, count]) => count > 0)
              .map(([gesture, count]) =>
                `${result.dataset_movements[gesture] ?? gesture}: ${count}`,
              )
              .join(', ') || 'none'}.
          </p>
          <p>
            Accuracy on accepted windows:{' '}
            {result.quality_gate.top1_accuracy_on_accepted === null
              ? 'not available'
              : percentage(result.quality_gate.top1_accuracy_on_accepted)}
          </p>
          <p>{result.latency.scope}</p>
          <p>{result.caveat}</p>
          <p>{browserLatency?.note}</p>
          <p>
            Zero-signal control:{' '}
            {browserLatency?.zeroSignalRejected === undefined
              ? 'not measured'
              : browserLatency.zeroSignalRejected ? 'rejected' : 'not rejected'}.
          </p>
        </div>
      </details>
    </section>
  );
}

function ConfusionMatrix({ result }: { result: HeldoutEvaluation }) {
  return (
    <div className="matrix-scroll">
      <table className="confusion-matrix">
        <caption>Held-out confusion matrix · rows are performed, columns are decoded</caption>
        <thead>
          <tr>
            <th scope="col">Performed ↓ / Decoded →</th>
            {result.gesture_order.map((gesture) => (
              <th scope="col" key={gesture}>{result.dataset_movements[gesture] ?? gesture}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.gesture_order.map((gesture, row) => (
            <tr key={gesture}>
              <th scope="row">{result.dataset_movements[gesture] ?? gesture}</th>
              {result.confusion_matrix[row]?.map((count, column) => (
                <td key={`${gesture}-${result.gesture_order[column]}`}>{count}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Timeline({ events }: { events: readonly TimelineEvent[] }) {
  return (
    <section className="performed-timeline" aria-labelledby="timeline-heading">
      <h4 id="timeline-heading">Performed vs decoded timeline</h4>
      <p className="small muted">
        Mapped, eligible 200 ms windows in recording order; each window advances 20 ms.
        Transition and unmapped-label windows are omitted.
      </p>
      <ol>
        {events.map((event, index) => (
          <li key={`${event.recording}-${event.repetition}-${event.start_ms}-${index}`}>
            <div className="timeline-label">
              <strong>{event.performed}</strong>
              <span>{event.recording} · rep {event.repetition} · {event.start_ms.toFixed(0)}–{event.end_ms.toFixed(0)} ms</span>
            </div>
            <div className="timeline-track" role="img" aria-label={`${event.performed} performed; decoded sequence ${event.decoded.map((run) => `${run.gesture} for ${run.windows} windows`).join(', ')}`}>
              {event.decoded.map((run, runIndex) => (
                <span
                  className="timeline-part"
                  data-correct={run.gesture === event.performed || undefined}
                  key={`${run.gesture}-${runIndex}`}
                  style={{ flexGrow: run.windows }}
                  title={`${run.gesture}: ${run.windows} windows`}
                />
              ))}
            </div>
            <p className="timeline-decoded">
              {event.decoded.map((run) => `${run.gesture} × ${run.windows}`).join(' · ')}
            </p>
          </li>
        ))}
      </ol>
    </section>
  );
}
