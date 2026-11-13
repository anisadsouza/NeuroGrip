import { useEffect, useState } from 'react';
import { Screen, type ScreenPage } from '../components/Screen.js';
import { MethodNote } from '../components/MethodNote.js';
import { Term } from '../components/Term.js';


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

interface CwerSummary {
  nTrials: number;
  nCorrectLatches: number;
  nWrongLatches: number;
  nSafeFallbacks: number;
  nUncommitted: number;
  errorRate: number | null;
  cwer: number | null;
  confusions: readonly { intended: string; committed: string; count: number; unitCost: number; totalCost: number }[];
}

interface TtumReport {
  source: string;
  model: string;
  nTrials: number;
  hopMs: number;
  anyMotion: {
    nMoved: number;
    nCensored: number;
    p50Ms: number | null;
    p95Ms: number | null;
    histogramMs?: Record<string, number>;
  };
  correctMotion: { nMoved: number; nCensored: number; p50Ms: number | null; p95Ms: number | null };
  latch: { nLatched: number; accuracyAtLatch: number | null; p50Ms: number | null; p95Ms: number | null };
  cwer: {
    riskWeighted: CwerSummary;
    uniformBoundary: CwerSummary & { uniformRisk: number; anyMotionP50Ms: number | null; latchP50Ms: number | null };
    definition: string;
    caveat: string;
  };
  caveat: string;
}

interface LoadedData {
  robustness: RobustnessReport | null;
  heldout: HeldoutRecord | null;
  recordedLatency: BrowserLatency | null;
  simulatedLatency: BrowserLatency | null;
  ttum: TtumReport | null;
}


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
      optionalJson<TtumReport>('/clinician/ttum.json'),
    ])
      .then(([robustness, heldout, recordedLatency, simulatedLatency, ttum]) => {
        setData({ robustness, heldout, recordedLatency, simulatedLatency, ttum });
      })
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === 'AbortError') return;
        setError(cause instanceof Error ? cause.message : 'Could not load clinician reports');
      });

    return () => controller.abort();
  }, []);

  return { data, error };
}

function ms(value: number | null): string {
  return value === null ? '—' : `${value} ms`;
}

function rate(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(2)}%`;
}

function TimingPanel({ report }: { report: TtumReport | null }) {
  if (!report) {
    return <p className="panel muted">The timing results are not included in this build.</p>;
  }
  const weighted = report.cwer.riskWeighted;
  const uniform = report.cwer.uniformBoundary;

  // Motion-onset histogram. Bins are the hop times the replay reports; the
  // axis runs to the slowest trial so nothing is cut off the right edge.
  const bins = Object.entries(report.anyMotion.histogramMs ?? {})
    .map(([key, count]) => ({ ms: Number(key), count }))
    .sort((a, b) => a.ms - b.ms);
  const maxMs = Math.max(report.hopMs * 12, ...bins.map((bin) => bin.ms));
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  const W = 520;
  const H = 180;
  const L = 36;
  const B = 28;
  const plotW = W - L - 8;
  const plotH = H - B - 8;
  const bx = (value: number) => L + (value / maxMs) * plotW;
  const barW = Math.max(4, (report.hopMs / maxMs) * plotW - 3);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => Math.round((maxMs * fraction) / 20) * 20);

  // The ablation: same decoder, same trials, only the boundary asymmetry removed.
  const pairs = [
    { label: 'CWER', weighted: weighted.cwer ?? 0, uniform: uniform.cwer ?? 0 },
    { label: 'Error rate', weighted: weighted.errorRate ?? 0, uniform: uniform.errorRate ?? 0 },
  ];
  const peak = Math.max(...pairs.flatMap((pair) => [pair.weighted, pair.uniform]), 0.0001);

  return (
    <section className="clinician-screen" aria-labelledby="timing-heading">
      <header className="clinician-content-header">
        <h3 id="timing-heading">Decision timing and cost-weighted error</h3>
      </header>
      <div className="clinician-two-column">
        <section className="panel clinician-panel figure-panel" aria-label="Time to useful motion">
          <h3><Term id="ttum">Time to useful motion</Term></h3>
          <dl className="stat-row">
            <div><dt>Motion starts, P50</dt><dd className="ng-num">{ms(report.anyMotion.p50Ms)}</dd></div>
            <div><dt>Latch, P50</dt><dd className="ng-num">{ms(report.latch.p50Ms)}</dd></div>
            <div><dt>Accuracy at latch</dt><dd className="ng-num">{percentage(report.latch.accuracyAtLatch ?? 0)}</dd></div>
          </dl>
          <svg
            className="ttum-histogram"
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label={`Histogram of motion onset over ${report.nTrials} trials; median ${ms(report.anyMotion.p50Ms)}`}
          >
            {ticks.map((tick) => (
              <g key={tick} className="chart-axis">
                <line x1={bx(tick)} x2={bx(tick)} y1={8} y2={H - B} />
                <text x={bx(tick)} y={H - 8} textAnchor="middle">{tick} ms</text>
              </g>
            ))}
            {bins.map((bin) => {
              const h = (bin.count / maxCount) * plotH;
              return (
                <rect
                  key={bin.ms}
                  className="ttum-bar"
                  x={bx(bin.ms) - barW / 2}
                  y={H - B - h}
                  width={barW}
                  height={h}
                />
              );
            })}
            {report.anyMotion.p50Ms !== null ? (
              <g className="chart-marker">
                <line x1={bx(report.anyMotion.p50Ms)} x2={bx(report.anyMotion.p50Ms)} y1={8} y2={H - B} />
                <text x={bx(report.anyMotion.p50Ms) + 6} y={20}>median</text>
              </g>
            ) : null}
          </svg>
          <p className="figure-caption">{report.nTrials} simulated trials, out of fold</p>
          <MethodNote title="Time to useful motion">
            <p>{report.nTrials} simulated out-of-fold trials replayed through the browser&apos;s own evidence accumulator ({report.model}, {report.hopMs} ms hop).</p>
            <p>Correct motion: P50 {ms(report.correctMotion.p50Ms)}, P95 {ms(report.correctMotion.p95Ms)}. {report.correctMotion.nCensored} of {report.nTrials} trials never moved the right way; they are censored, never given a finite time.</p>
            <p>Excludes the 200 ms window fill before the first decision. The simulator has no onset envelope, so this is a lower bound, not reaction time from intent.</p>
          </MethodNote>
        </section>

        <section className="panel clinician-panel figure-panel" aria-label="Cost-weighted error rate">
          <h3><Term id="cwer">Cost-weighted error rate</Term></h3>
          <dl className="stat-row">
            <div><dt>CWER, risk-weighted boundaries</dt><dd className="ng-num">{rate(weighted.cwer)}</dd></div>
            <div><dt>Wrong latches</dt><dd className="ng-num">{weighted.nWrongLatches} <span className="stat-vs">vs {uniform.nWrongLatches} uniform</span></dd></div>
          </dl>
          <div className="ablation" role="img" aria-label={`Risk-weighted boundaries: CWER ${rate(weighted.cwer)}, error rate ${rate(weighted.errorRate)}. Uniform boundaries: CWER ${rate(uniform.cwer)}, error rate ${rate(uniform.errorRate)}.`}>
            {pairs.map((pair) => (
              <div className="ablation-group" key={pair.label}>
                <span className="ablation-label">{pair.label}</span>
                <span className="ablation-bar" data-kind="weighted" style={{ width: `${(pair.weighted / peak) * 100}%` }} />
                <span className="ablation-value ng-num">{rate(pair.weighted)}</span>
                <span className="ablation-bar" data-kind="uniform" style={{ width: `${(pair.uniform / peak) * 100}%` }} />
                <span className="ablation-value ng-num">{rate(pair.uniform)}</span>
              </div>
            ))}
            <p className="ablation-legend">
              <span className="legend-key" data-kind="weighted" /> risk-weighted
              <span className="legend-key" data-kind="uniform" /> uniform (ablation)
            </p>
          </div>
          <MethodNote title="Cost-weighted error rate">
            <p>{report.cwer.definition}</p>
            <p>
              Uncommitted trials: {weighted.nUncommitted} weighted, {uniform.nUncommitted} uniform.
              Safe fallbacks to rest: {weighted.nSafeFallbacks}. Uniform boundaries start moving at
              P50 {ms(uniform.anyMotionP50Ms)}.
            </p>
            {weighted.confusions.length > 0 ? (
              <p>
                Wrong latches:{' '}
                {weighted.confusions
                  .map((item) => `${item.intended.replaceAll('_', ' ')} → ${item.committed.replaceAll('_', ' ')} ×${item.count} (cost ${item.unitCost.toFixed(1)})`)
                  .join('; ')}.
              </p>
            ) : null}
            <p>
              {weighted.nWrongLatches} and {uniform.nWrongLatches} wrong latches are too few to call
              the difference significant. {report.cwer.caveat}
            </p>
          </MethodNote>
        </section>
      </div>
    </section>
  );
}

function RobustnessPanel({ report }: { report: RobustnessReport | null }) {
  if (!report) {
    return (
      <section className="panel clinician-panel" aria-labelledby="robustness-heading">
        <h3 id="robustness-heading">Electrode-shift robustness</h3>
        <p className="muted">
          The robustness results are not included in this build.
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
            Simulated LOSO · {report.model} · {report.corpus.n_subjects} subjects
          </p>
        </div>
        <strong className={report.acceptance.meets_bound ? 'clinician-status' : 'clinician-status is-failed'}>
          {report.acceptance.meets_bound ? 'Within bound' : 'Bound not met'}
        </strong>
      </div>

      <div className="clinician-chart-scroll" tabIndex={0} role="region" aria-label="Robustness chart, scrollable">
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

      <div className="clinician-table-wrap" tabIndex={0} role="region" aria-label="Robustness table, scrollable">
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
      <MethodNote title="Electrode-shift robustness">
        <p>{report.caveat}</p>
        <p>
          The ≤{report.acceptance.max_allowed_degradation_pp.toFixed(0)} percentage-point bound is an
          exploratory simulator gate; it is not a clinical acceptance threshold.
        </p>
      </MethodNote>
    </section>
  );
}

function LatencyCard({ title, result }: { title: string; result: BrowserLatency }) {
  const share = Math.min(1, result.p95Ms / result.budgetMs);
  return (
    <section className="panel clinician-panel figure-panel" aria-label={title}>
      <h3>{title}</h3>
      <dl className="stat-row">
        <div><dt><Term id="latency">Latency</Term>, P95</dt><dd className="ng-num">{result.p95Ms.toFixed(2)} ms</dd></div>
        <div><dt>Budget</dt><dd className="ng-num">{result.budgetMs} ms · {result.meetsBudget ? 'met' : 'exceeded'}</dd></div>
      </dl>
      <div
        className="budget"
        role="img"
        aria-label={`P95 ${result.p95Ms.toFixed(2)} ms of a ${result.budgetMs} ms budget`}
      >
        <span className="budget-used" style={{ width: `${share * 100}%` }} />
        <span className="budget-label ng-num">{Math.round(share * 100)}% of budget</span>
      </div>
      <dl className="clinician-metrics">
        <div><dt>P50 / P99</dt><dd>{result.p50Ms.toFixed(2)} / {result.p99Ms.toFixed(2)} ms</dd></div>
        <div><dt>Features / inference, P95</dt><dd>{result.featureP95Ms.toFixed(2)} / {result.inferenceP95Ms.toFixed(2)} ms</dd></div>
        <div><dt>Windows / rejected</dt><dd>{result.windowsProcessed.toLocaleString()} / {result.rejectedWindows.toLocaleString()}</dd></div>
        <div><dt>Runtime</dt><dd>{result.backend}{result.threaded ? ' · threaded' : ' · single-threaded'}</dd></div>
      </dl>
      <MethodNote title={title}>
        <p>{result.note}</p>
        <p>
          Stable-replay accuracy {result.stablePredictionAccuracy === null ? 'not available' : percentage(result.stablePredictionAccuracy)};
          an all-zero window was {result.zeroSignalRejected ? 'rejected, as required' : 'not rejected'}. Measured {result.measuredAt}.
        </p>
      </MethodNote>
    </section>
  );
}

function LatencyPanel({ data }: { data: LoadedData }) {
  const cards = [
    { title: 'Recorded NinaPro replay · S10', result: data.recordedLatency },
    { title: 'Simulated replay', result: data.simulatedLatency },
  ].filter((card): card is { title: string; result: BrowserLatency } => card.result !== null);
  return (
    <section className="clinician-screen" aria-labelledby="latency-heading">
      <header className="clinician-content-header">
        <h3 id="latency-heading">Browser latency</h3>
      </header>
      {cards.length > 0 ? (
        <div className="clinician-two-column" data-count={cards.length}>
          {cards.map((card) => (
            <LatencyCard key={card.title} title={card.title} result={card.result} />
          ))}
        </div>
      ) : (
        <p className="panel muted">No browser latency measurement is included in this build.</p>
      )}
    </section>
  );
}

function RecordsPanel({ result }: { result: HeldoutRecord }) {
  return (
    <section className="clinician-screen" aria-labelledby="records-heading">
      <header className="clinician-content-header">
        <h3 id="records-heading">Evaluation records and provenance</h3>
      </header>
      <section className="panel clinician-panel" aria-label="Held-out study record">
        <h3>Held-out evaluation</h3>
        <dl className="clinician-record">
          <div><dt>Source</dt><dd>{result.source === 'ninapro_db2' ? 'NinaPro DB2' : result.source}</dd></div>
          <div><dt>Model</dt><dd>{result.model}</dd></div>
          <div><dt>Training subjects</dt><dd>{result.training_subjects.map((subject) => `S${subject}`).join(', ') || 'None listed'}</dd></div>
          <div><dt>Held-out subject</dt><dd>S{result.test_subject} · excluded from training: {result.subject_excluded_from_training ? 'yes' : 'no'}</dd></div>
          <div><dt>Windows, train / test</dt><dd>{result.n_train.toLocaleString()} / {result.n_test.toLocaleString()}</dd></div>
          <div><dt>Accuracy / <Term id="calibration">ECE</Term></dt><dd>{percentage(result.accuracy)} / {result.expected_calibration_error.toFixed(3)}</dd></div>
          <div><dt>Quality gate, accepted / rejected</dt><dd>{result.quality_gate.accepted.toLocaleString()} / {result.quality_gate.rejected.toLocaleString()} ({percentage(result.quality_gate.acceptance_rate)})</dd></div>
          <div><dt>Latency</dt><dd>{result.latency.runtime} · {result.latency.scope} · P95 {result.latency.p95_ms.toFixed(2)} ms</dd></div>
        </dl>
        <MethodNote title="Held-out evaluation">
          <p>{result.caveat}</p>
          <p>Quality gate: {result.quality_gate.method}</p>
          <p>Provenance is limited to the staged evaluation summary. Raw NinaPro signal and recording identifiers are not displayed, and no person-level record is collected or stored.</p>
        </MethodNote>
      </section>
    </section>
  );
}

/** What this prototype has and has not shown. Words only: every number lives on its own page. */
function LimitsPanel() {
  return (
    <section className="clinician-screen" aria-labelledby="limits-heading">
      <header className="clinician-content-header">
        <h3 id="limits-heading">Limits</h3>
      </header>
      <div className="limits">
        <section className="panel" aria-labelledby="limits-built">
          <h3 id="limits-built">Built and working</h3>
          <ul>
            <li>One feature pipeline in Python and TypeScript, pinned together by golden-vector conformance tests.</li>
            <li>The risk-weighted evidence accumulator, running live in the browser worker you used in step 2.</li>
            <li>Browser inference inside the 10 ms budget, with the threading it depends on asserted by a test.</li>
            <li>Every number on these screens traced to an artifact written by a runnable command.</li>
          </ul>
        </section>
        <section className="panel" aria-labelledby="limits-open">
          <h3 id="limits-open">Not yet shown</h3>
          <ul>
            <li>No human recording evaluated end to end. NinaPro results are exploratory, from three subjects.</li>
            <li>Teach-back used a simulated learner. Whether people can learn the new gestures is untested.</li>
            <li>Error routing and coaching were checked on injected simulated faults only. Coaching localisation is an upper bound: the inverse uses the simulator&rsquo;s own forward model.</li>
            <li>Time to useful motion has no onset envelope, so it is a lower bound, not reaction time from intent.</li>
            <li>Robustness covers simulated electrode rotation only: no motion artefact, skin-impedance drift or re-donning.</li>
            <li>No amputee evaluation and no clinical claim. Not a medical device.</li>
          </ul>
        </section>
      </div>
    </section>
  );
}

export function Clinician() {
  const { data, error } = useClinicianData();
  const [page, setPage] = useState(0);

  const status = error ? (
    <p className="notice notice-stop" role="alert">{error}</p>
  ) : !data ? (
    <p className="screen-empty muted" role="status">Loading research reports…</p>
  ) : undefined;

  const pages: ScreenPage[] = data
    ? [
        {
          id: 'timing',
          label: 'Decision timing',
          summary:
            'When the hand starts moving, and what its mistakes cost.',
          content: <TimingPanel report={data.ttum} />,
        },
        {
          id: 'robustness',
          label: 'Robustness',
          summary:
            'Accuracy as the electrode band rotates. Simulated shift only.',
          content: <RobustnessPanel report={data.robustness} />,
        },
        {
          id: 'latency',
          label: 'Latency',
          summary:
            'Features plus inference for one window, in this browser.',
          content: <LatencyPanel data={data} />,
        },
        ...(data.heldout
          ? [
              {
                id: 'records',
                label: 'Records',
                summary: 'Where the held-out result came from.',
                content: <RecordsPanel result={data.heldout} />,
              },
            ]
          : []),
        {
          id: 'limits',
          label: 'Limits',
          summary:
            'What this prototype has shown, and what it has not.',
          content: <LimitsPanel />,
        },
      ]
    : [];

  return (
    <Screen
      title="Speed, safety and limits"
      className="screen-review"
      pages={pages}
      page={page}
      onPage={setPage}
      status={status}
    />
  );
}
