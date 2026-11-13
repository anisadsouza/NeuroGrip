/** Recorded experiment results, kept separate from the live control screen. */

import { useEffect, useState } from 'react';
import { Screen, type ScreenPage } from '../components/Screen.js';
import { MethodNote } from '../components/MethodNote.js';
import { Term } from '../components/Term.js';

/** Plain names for the decoders compared. */
const MODEL_NAME: Readonly<Record<string, string>> = {
  rbf_svm: 'RBF SVM',
  linear_svm: 'Linear SVM',
  lda: 'LDA',
  lda_uniform: 'LDA, uniform prior',
};

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
  const [recordedLatency, setRecordedLatency] = useState<BrowserLatency | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([
      readOptionalJson<Comparison>(`${import.meta.env.BASE_URL}results/model_comparison.json`),
      readOptionalJson<Comparison>(`${import.meta.env.BASE_URL}results/model_comparison_ninapro.json`),
      readOptionalJson<HeldoutEvaluation>(`${import.meta.env.BASE_URL}results/ninapro_holdout.json`),
      readOptionalJson<BrowserLatency>(`${import.meta.env.BASE_URL}results/browser_latency_ninapro.json`),
    ]).then(([simulation, ninaPro, evaluation, ninaProLatency]) => {
      setSimulated(simulation);
      setRecorded(ninaPro);
      setHeldout(evaluation);
      setRecordedLatency(ninaProLatency);
    }).catch((error: unknown) => {
      setLoadError(error instanceof Error ? error.message : String(error));
    });
  }, []);

  const [page, setPage] = useState(0);

  const recordedPages: ScreenPage[] = heldout
    ? [
        {
          id: 'recorded',
          label: 'Real recording',
          summary: 'A recorded person the decoder never trained on. Exploratory.',
          content: <HeldoutPanel result={heldout} browserLatency={recordedLatency} />,
        },
        {
          id: 'confusion',
          label: 'Mistakes',
          summary: 'Rows: performed. Columns: decoded. The diagonal is correct.',
          content: <ConfusionMatrix result={heldout} />,
        },
        {
          id: 'timeline',
          label: 'Timeline',
          summary: 'Performed movements in order, with what the decoder said.',
          content: <Timeline events={heldout.timeline} />,
        },
      ]
    : [];
  const comparisonPage: ScreenPage = {
    id: 'comparison',
    label: 'Model comparison',
    summary: 'Three decoders, each tested on subjects it never saw.',
    content: (
      <div className="results-pair" data-count={recorded ? 2 : 1}>
        <ResultTable title="Simulated sEMG model comparison" result={simulated} />
        {recorded ? <ResultTable title="Recorded NinaPro DB2 model comparison" result={recorded} /> : null}
      </div>
    ),
  };
  // Real data first when it exists. Browser speed lives in step 6, not here.
  const pages: ScreenPage[] = [...recordedPages, comparisonPage];

  return (
    <Screen
      title="Accuracy"
      className="screen-results"
      pages={pages}
      page={page}
      onPage={setPage}
      status={loadError ? <p className="notice notice-stop" role="alert">{loadError}</p> : undefined}
    />
  );
}

function ResultTable({ title, result }: { title: string; result: Comparison | null }) {
  const id = headingId(title);
  return (
    <section className="panel results-table" aria-labelledby={id}>
      <h3 id={id}>{title}</h3>
      {result === null ? (
        <p className="muted">
          Not in this build: NinaPro recordings are licensed separately.
        </p>
      ) : (
        <>
          <table>
            <caption>Selected: {MODEL_NAME[result.best_model] ?? result.best_model}, highest accuracy and lowest ECE</caption>
            <thead><tr><th scope="col">Decoder</th><th scope="col"><Term id="unseen">LOSO accuracy</Term></th><th scope="col">SD</th><th scope="col"><Term id="calibration">ECE</Term></th></tr></thead>
            <tbody>
              {Object.entries(result.models).map(([name, model]) => (
                <tr key={name} data-selected={name === result.best_model || undefined}>
                  <th scope="row">{MODEL_NAME[name] ?? name}</th>
                  <td>
                    <span className="meter-cell">
                      <span className="ng-num">{percentage(model.mean_accuracy)}</span>
                      <span className="meter" data-story="grip" aria-hidden="true">
                        <span style={{ width: `${Math.max(0, Math.min(1, model.mean_accuracy)) * 100}%` }} />
                      </span>
                    </span>
                  </td>
                  <td className="ng-num">±{(model.std_accuracy * 100).toFixed(1)} pts</td>
                  <td>{model.expected_calibration_error.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <MethodNote title={title}>
            <p>{result.caveat}</p>
            {result.source === 'ninapro_db2' ? (
              <p>
                DB2 label 5 is finger abduction; it is an explicitly qualified proxy
                for the app's open-hand class, not an exact movement-name match.
              </p>
            ) : (
              <p>
                Recorded-person results need the NinaPro recordings, which are licensed
                separately; when they are staged, they appear beside this table.
              </p>
            )}
          </MethodNote>
        </>
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
        <h3 id="heldout-heading">A real person it never saw</h3>
        <p className="muted">
          This build does not include the real recordings, which are licensed
          separately. When they are present, this page shows how well the decoder
          did on a recorded person it never trained on. Meanwhile, Model
          comparison shows results on simulated people.
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
          <dt>Accuracy</dt>
          <dd>{percentage(result.accuracy)}</dd>
          <dd className="results-metric-detail">{result.n_test.toLocaleString()} windows</dd>
        </div>
        <div>
          <dt>Right on a steady signal</dt>
          <dd>{typeof stableAccuracy === 'number' ? percentage(stableAccuracy) : '—'}</dd>
          <dd className="results-metric-detail">
            {browserLatency?.correctStablePredictions ?? 0}/
            {browserLatency?.stablePredictionWindows ?? 0} windows
          </dd>
        </div>
        <div>
          <dt>Signal passed the check</dt>
          <dd>{percentage(result.quality_gate.acceptance_rate)}</dd>
          <dd className="results-metric-detail">
            {result.quality_gate.accepted.toLocaleString()} accepted ·{' '}
            {result.quality_gate.rejected.toLocaleString()} rejected
          </dd>
        </div>
        <div>
          <dt>Decision time, P95</dt>
          <dd>{browserLatency ? `${browserLatency.p95Ms.toFixed(2)} ms` : '—'}</dd>
          <dd className="results-metric-detail">feature extraction + ONNX inference</dd>
        </div>
        <div>
          <dt><Term id="calibration">Confidence error</Term></dt>
          <dd>{result.expected_calibration_error.toFixed(3)}</dd>
          <dd className="results-metric-detail">lower is better</dd>
        </div>
      </dl>

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

      <MethodNote title="Held-out evaluation">
          <p>
            Exploratory only: three subjects are available locally, and S10 informed
            diagnostics. This is not an untouched final test or clinical validation.
          </p>
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
      </MethodNote>
    </section>
  );
}

function ConfusionMatrix({ result }: { result: HeldoutEvaluation }) {
  return (
    <div className="matrix-scroll">
      <table className="confusion-matrix">
        <caption>Rows: what the person did. Columns: what the decoder said.</caption>
        <thead>
          <tr>
            <th scope="col">Did ↓ / Said →</th>
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
      <h4 id="timeline-heading">What they did, and what the decoder said</h4>
      <MethodNote>
        <p>
          Mapped, eligible 200 ms windows in recording order; each window advances 20 ms.
          Transition and unmapped-label windows are omitted.
        </p>
      </MethodNote>
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
