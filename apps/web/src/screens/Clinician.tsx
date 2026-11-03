import { useEffect, useState } from 'react';
import { Results } from './Results.js';

type ClinicianView = 'evaluate' | 'robustness' | 'latency' | 'records';

interface ShiftResult {
  shift_rad: number;
  mean_accuracy: number;
  std_accuracy: number;
  min_accuracy: number;
  ci95_low: number;
  ci95_high: number;
  degradation_pp: number;
  per_fold_accuracy: readonly number[];
}

interface RobustnessReport {
  model: string;
  corpus: {
    n_subjects: number;
    reps_per_gesture: number;
    rep_seconds: number;
    n_electrodes: number;
    window_ms: number;
    hop_ms: number;
    n_windows: number;
    n_features: number;
  };
  baseline_accuracy: number;
  shifts: readonly ShiftResult[];
  acceptance: {
    max_allowed_degradation_pp: number;
    meets_bound: boolean;
  };
  caveat: string;
}

interface HeldoutRecord {
  source: string;
  model: string;
  training_subjects: readonly number[];
  test_subject: number;
  subject_excluded_from_training: boolean;
  n_train: number;
  n_test: number;
  accuracy: number;
  expected_calibration_error: number;
  quality_gate: {
    accepted: number;
    rejected: number;
    acceptance_rate: number;
    method: string;
  };
  latency: {
    runtime: string;
    scope: string;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
  };
  caveat: string;
}

interface BrowserLatency {
  measuredAt: string;
  budgetMs: number;
  meetsBudget: boolean;
  windowsProcessed: number;
  rejectedWindows: number;
  stablePredictionWindows: number;
  correctStablePredictions: number;
  stablePredictionAccuracy: number | null;
  zeroSignalRejected: boolean;
  threaded: boolean;
  backend: string;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
  featureP95Ms: number;
  inferenceP95Ms: number;
  note: string;
}

interface LoadedData {
  robustness: RobustnessReport | null;
  heldout: HeldoutRecord | null;
  recordedLatency: BrowserLatency | null;
  simulatedLatency: BrowserLatency | null;
}

const VIEWS: readonly { id: ClinicianView; label: string }[] = [
  { id: 'evaluate', label: 'Evaluate' },
  { id: 'robustness', label: 'Robustness' },
  { id: 'latency', label: 'Latency' },
  { id: 'records', label: 'Records' },
];

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function useClinicianData() {
  const [data, setData] = useState<LoadedData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const optionalJson = async <T,>(url: string): Promise<T | null> => {
      const response = await fetch(url, { signal: controller.signal });
      if (response.status === 404) return null;
      if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
      if (response.headers.get('content-type')?.includes('text/html')) return null;
      return response.json() as Promise<T>;
    };

    void Promise.all([
      optionalJson<RobustnessReport>('/clinician/robustness.json'),
      optionalJson<HeldoutRecord>('/results/ninapro_holdout.json'),
      optionalJson<BrowserLatency>('/results/browser_latency_ninapro.json'),
      optionalJson<BrowserLatency>('/results/browser_latency.json'),
    ])
      .then(([robustness, heldout, recordedLatency, simulatedLatency]) => {
        setData({ robustness, heldout, recordedLatency, simulatedLatency });
      })
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === 'AbortError') return;
        setError(cause instanceof Error ? cause.message : 'Could not load clinician reports');
      });

    return () => controller.abort();
  }, []);

  return { data, error };
}

function RobustnessPanel({ report }: { report: RobustnessReport | null }) {
  if (!report) {
    return (
      <section className="panel clinician-panel" aria-labelledby="robustness-heading">
        <h3 id="robustness-heading">Electrode-shift robustness</h3>
        <p className="muted">
          No robustness report is staged. Generate it with{' '}
          <code>python -m neurogrip.experiments robustness</code>, then run{' '}
          <code>npm run stage:clinician</code>.
        </p>
      </section>
    );
  }

  const width = 620;
  const height = 280;
  const left = 60;
  const top = 20;
  const values = report.shifts.map((point) => point.mean_accuracy);
  const minValue = Math.max(0, Math.floor(Math.min(...values, report.baseline_accuracy) * 10) / 10 - 0.1);
  const maxValue = Math.min(1, Math.ceil(Math.max(...values, report.baseline_accuracy) * 10) / 10 + 0.1);
  const x = (index: number) => left + (index / Math.max(report.shifts.length - 1, 1)) * width;
  const y = (value: number) => top + ((maxValue - value) / Math.max(maxValue - minValue, 0.01)) * height;
  const points = report.shifts.map((point, index) => `${x(index)},${y(point.mean_accuracy)}`).join(' ');

  return (
    <section className="panel clinician-panel" aria-labelledby="robustness-heading">
      <div className="clinician-section-heading">
        <div>
          <h3 id="robustness-heading">Electrode-shift robustness</h3>
          <p className="small muted">
            Simulated LOSO results · {report.model} · {report.corpus.n_subjects} subjects ·
            {' '}{report.corpus.n_windows.toLocaleString()} windows
          </p>
        </div>
        <strong className={report.acceptance.meets_bound ? 'clinician-status' : 'clinician-status is-failed'}>
          {report.acceptance.meets_bound ? 'Within bound' : 'Bound not met'}
        </strong>
      </div>

      <div className="clinician-chart-scroll">
        <svg
          viewBox="0 0 720 360"
          role="img"
          aria-label={`Simulated accuracy by electrode shift for ${report.model}`}
        >
          <rect x={left} y={top} width={width} height={height} className="clinician-chart-bed" />
          {Array.from({ length: 5 }, (_, index) => {
            const value = minValue + ((maxValue - minValue) * index) / 4;
            return (
              <g key={index} className="clinician-chart-grid">
                <line x1={left} x2={left + width} y1={y(value)} y2={y(value)} />
                <text x={left - 10} y={y(value) + 4} textAnchor="end">{percentage(value)}</text>
              </g>
            );
          })}
          <line
            x1={left}
            x2={left + width}
            y1={y(report.baseline_accuracy)}
            y2={y(report.baseline_accuracy)}
            className="clinician-baseline"
          />
          <polyline points={points} className="clinician-data-line" />
          {report.shifts.map((point, index) => (
            <g key={point.shift_rad} className="clinician-data-point">
              <circle cx={x(index)} cy={y(point.mean_accuracy)} r="5" />
              <text x={x(index)} y={top + height + 20} textAnchor="middle">
                {point.shift_rad.toFixed(1)} rad
              </text>
            </g>
          ))}
          <text x={left + width} y={y(report.baseline_accuracy) - 7} textAnchor="end" className="clinician-chart-label">
            unshifted baseline
          </text>
          <text x={left + width / 2} y="344" textAnchor="middle" className="clinician-chart-label">
            Simulated electrode rotation
          </text>
        </svg>
      </div>

      <div className="clinician-table-wrap">
        <table className="clinician-table">
          <caption>Per-shift simulated LOSO accuracy and interval</caption>
          <thead><tr><th scope="col">Shift</th><th scope="col">Accuracy</th><th scope="col">95% interval</th><th scope="col">Change vs baseline</th><th scope="col">Worst subject</th></tr></thead>
          <tbody>
            {report.shifts.map((point) => (
              <tr key={point.shift_rad}>
                <th scope="row">{point.shift_rad.toFixed(1)} rad</th>
                <td>{percentage(point.mean_accuracy)}</td>
                <td>{percentage(point.ci95_low)}–{percentage(point.ci95_high)}</td>
                <td>{point.degradation_pp.toFixed(2)} pp</td>
                <td>{percentage(point.min_accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted clinician-caveat">{report.caveat}</p>
      <p className="small muted">
        The ≤{report.acceptance.max_allowed_degradation_pp.toFixed(0)} percentage-point
        bound is an exploratory simulator gate; it is not a clinical acceptance threshold.
      </p>
    </section>
  );
}

function LatencyCard({ title, result }: { title: string; result: BrowserLatency | null }) {
  return (
    <section className="panel clinician-panel" aria-label={title}>
      <h3>{title}</h3>
      {result ? (
        <>
          <dl className="clinician-metrics">
            <div><dt>Worker P95</dt><dd>{result.p95Ms.toFixed(2)} ms</dd></div>
            <div><dt>Decode budget</dt><dd>{result.meetsBudget ? 'Met' : 'Exceeded'} · {result.budgetMs} ms</dd></div>
            <div><dt>Worker P50 / P99</dt><dd>{result.p50Ms.toFixed(2)} / {result.p99Ms.toFixed(2)} ms</dd></div>
            <div><dt>Feature / inference P95</dt><dd>{result.featureP95Ms.toFixed(2)} / {result.inferenceP95Ms.toFixed(2)} ms</dd></div>
            <div><dt>Windows / rejects</dt><dd>{result.windowsProcessed.toLocaleString()} / {result.rejectedWindows.toLocaleString()}</dd></div>
            <div><dt>WASM backend</dt><dd>{result.backend}{result.threaded ? ' · threaded' : ' · single-threaded'}</dd></div>
            <div><dt>Zero-signal control</dt><dd>{result.zeroSignalRejected ? 'Rejected' : 'Not rejected'}</dd></div>
            <div><dt>Stable replay top-1</dt><dd>{result.stablePredictionAccuracy === null ? 'Not available' : percentage(result.stablePredictionAccuracy)}</dd></div>
            <div><dt>Measurement date</dt><dd>{result.measuredAt}</dd></div>
          </dl>
          <p className="small muted">{result.note}</p>
        </>
      ) : (
        <p className="muted">No browser latency artifact is staged for this source.</p>
      )}
    </section>
  );
}

function LatencyPanel({ data }: { data: LoadedData }) {
  return (
    <section className="clinician-screen" aria-labelledby="latency-heading">
      <header className="clinician-content-header">
        <h3 id="latency-heading">Browser latency</h3>
        <p className="muted">
          Worker feature extraction plus ONNX inference, measured one window at a
          time. It excludes window fill, messaging, scheduling and user reaction time.
        </p>
      </header>
      <div className="clinician-two-column">
        <LatencyCard title="Recorded NinaPro replay · S10" result={data.recordedLatency} />
        <LatencyCard title="Simulated replay" result={data.simulatedLatency} />
      </div>
    </section>
  );
}

function RecordsPanel({ data }: { data: LoadedData }) {
  const result = data.heldout;
  return (
    <section className="clinician-screen" aria-labelledby="records-heading">
      <header className="clinician-content-header">
        <h3 id="records-heading">Evaluation records and provenance</h3>
        <p className="muted">
          Local research-artifact metadata only. No person-level clinical record
          is collected or stored by this prototype.
        </p>
      </header>
      {result ? (
        <>
          <section className="panel clinician-panel" aria-label="Held-out study record">
            <h3>Held-out evaluation</h3>
            <dl className="clinician-record">
              <div><dt>Source</dt><dd>{result.source === 'ninapro_db2' ? 'NinaPro DB2' : result.source}</dd></div>
              <div><dt>Model</dt><dd>{result.model}</dd></div>
              <div><dt>Training subjects</dt><dd>{result.training_subjects.map((subject) => `S${subject}`).join(', ') || 'None listed'}</dd></div>
              <div><dt>Held-out subject</dt><dd>S{result.test_subject} · excluded from training: {result.subject_excluded_from_training ? 'yes' : 'no'}</dd></div>
              <div><dt>Train / held-out windows</dt><dd>{result.n_train.toLocaleString()} / {result.n_test.toLocaleString()}</dd></div>
              <div><dt>Top-1 accuracy / ECE</dt><dd>{percentage(result.accuracy)} / {result.expected_calibration_error.toFixed(3)}</dd></div>
              <div><dt>Signal gate</dt><dd>{result.quality_gate.method}</dd></div>
              <div><dt>Accepted / rejected</dt><dd>{result.quality_gate.accepted.toLocaleString()} / {result.quality_gate.rejected.toLocaleString()} ({percentage(result.quality_gate.acceptance_rate)} accepted)</dd></div>
              <div><dt>Latency measurement</dt><dd>{result.latency.runtime} · {result.latency.scope} · P95 {result.latency.p95_ms.toFixed(2)} ms</dd></div>
            </dl>
            <p className="small muted">{result.caveat}</p>
          </section>
          <p className="small muted">
            Record provenance is limited to the staged evaluation summary.
            Raw NinaPro signal and identifiers for individual recordings are not
            displayed here.
          </p>
        </>
      ) : (
        <p className="panel muted">
          No held-out evaluation record is staged. Follow the locally licensed
          NinaPro workflow in the walkthrough.
        </p>
      )}
    </section>
  );
}

export function Clinician() {
  const { data, error } = useClinicianData();
  const [view, setView] = useState<ClinicianView>('evaluate');

  return (
    <section className="clinician" aria-labelledby="clinician-title">
      <header className="clinician-header">
        <div>
          <h2 id="clinician-title">Research review</h2>
          <p className="muted">
            Clinician-oriented research views; these are not clinical decisions,
            records, or validated device checks.
          </p>
        </div>
        <p className="notice" role="note">
          Research prototype. No patient records, live electrodes, or clinical
          recommendations are supported.
        </p>
      </header>
      <nav className="clinician-nav" aria-label="Research review sections">
        {VIEWS.map((item) => (
          <button
            type="button"
            key={item.id}
            aria-current={view === item.id ? 'page' : undefined}
            onClick={() => setView(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      {error ? <p className="notice notice-stop" role="alert">{error}</p> : null}
      {view === 'evaluate' ? <Results />
        : !data ? <p className="panel muted" role="status">Loading research reports…</p>
          : view === 'robustness' ? <RobustnessPanel report={data.robustness} />
            : view === 'latency' ? <LatencyPanel data={data} />
              : <RecordsPanel data={data} />}
    </section>
  );
}
