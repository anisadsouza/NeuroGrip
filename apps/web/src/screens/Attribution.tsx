import { useEffect, useState } from 'react';

type Cause = 'physiological' | 'behavioral' | 'model_capacity' | 'drift' | 'inconclusive';
type ScoredCause = Exclude<Cause, 'inconclusive'>;

interface ChannelAttribution {
  channel: number;
  ring_location: string;
  deviation: number;
  direction: 'above' | 'below';
}

interface AttributionCase {
  name: string;
  intended: string;
  decoded: string;
  cause: Cause;
  cause_label: string;
  injected_cause: ScoredCause;
  route_matches_injection: boolean;
  scores: Record<ScoredCause, number>;
  evidence: {
    class_overlap: number;
    behavioral_deviation_ratio: number;
    reference_correct_when_deployed_wrong: boolean;
    centroid_cusum_ratio: number;
    median_frequency_shift_ratio: number;
  };
  routed_action: string;
  coaching: string | null;
  channel_attribution: readonly ChannelAttribution[];
}

interface AttributionReport {
  source: 'synthetic_fault_injections';
  injected_causes_routed: number;
  n_injections: number;
  method: string;
  cases: readonly AttributionCase[];
  caveat: string;
}

const CAUSE_ORDER: readonly ScoredCause[] = [
  'physiological',
  'behavioral',
  'model_capacity',
  'drift',
];

const CAUSE_NAMES: Readonly<Record<ScoredCause, string>> = {
  physiological: 'Physiological overlap',
  behavioral: 'Behavioral deviation',
  model_capacity: 'Model-capacity limitation',
  drift: 'Signal drift',
};

function useReport() {
  const [report, setReport] = useState<AttributionReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetch('/attribution/demo.json', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Attribution demo report returned HTTP ${response.status}`);
        }
        return response.json() as Promise<AttributionReport>;
      })
      .then(setReport)
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === 'AbortError') return;
        setError(cause instanceof Error ? cause.message : 'Could not load attribution report');
      });
    return () => controller.abort();
  }, []);

  return { report, error };
}

function ForearmRing({ channels }: { channels: readonly ChannelAttribution[] }) {
  const marks = new Map(channels.map((item) => [item.channel, item]));
  const points = Array.from({ length: 8 }, (_, index) => {
    const angle = (index / 8) * Math.PI * 2 - Math.PI / 2;
    return {
      channel: index + 1,
      x: 100 + 72 * Math.cos(angle),
      y: 100 + 72 * Math.sin(angle),
    };
  });

  return (
    <figure className="attribution-ring">
      <svg
        viewBox="0 0 200 200"
        role="img"
        aria-label="Forearm electrode ring showing channels with the largest feature deviations"
      >
        <circle cx="100" cy="100" r="72" className="attribution-ring-outline" />
        <circle cx="100" cy="100" r="46" className="attribution-forearm-outline" />
        <text x="100" y="96" textAnchor="middle" className="attribution-ring-centre">
          Channel
        </text>
        <text x="100" y="111" textAnchor="middle" className="attribution-ring-centre">
          deviation
        </text>
        {points.map(({ channel, x, y }) => {
          const attribution = marks.get(channel);
          const radius = attribution ? 7 + attribution.deviation * 5 : 6;
          return (
            <g key={channel} className={attribution ? 'attribution-mark is-highlighted' : 'attribution-mark'}>
              <circle cx={x} cy={y} r={radius} />
              <text x={x} y={y + 3} textAnchor="middle">C{channel}</text>
            </g>
          );
        })}
      </svg>
      <figcaption className="small muted">
        Ring positions are schematic. Highlighted channels are the largest
        standardized deviations from the intended class's repeat cluster.
      </figcaption>
    </figure>
  );
}

function CauseScores({ scores }: { scores: AttributionCase['scores'] }) {
  return (
    <div className="attribution-scores">
      {CAUSE_ORDER.map((cause) => (
        <div className="attribution-score" key={cause}>
          <div className="attribution-score-label">
            <span>{CAUSE_NAMES[cause]}</span>
            <span className="ng-num">{(scores[cause] * 100).toFixed(1)}%</span>
          </div>
          <div
            className="attribution-score-track"
            role="meter"
            aria-label={`${CAUSE_NAMES[cause]} evidence`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(scores[cause] * 100)}
          >
            <span style={{ width: `${Math.min(100, Math.max(0, scores[cause] * 100))}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function Attribution() {
  const { report, error } = useReport();
  const [selectedName, setSelectedName] = useState<string | null>(null);

  if (error) {
    return (
      <section className="attribution-screen" aria-labelledby="attribution-heading">
        <h2 id="attribution-heading">Error attribution and coaching</h2>
        <p className="notice notice-stop" role="alert">{error}</p>
        <p className="small muted">
          Generate the controlled-injection report with{' '}
          <code>python -m neurogrip.experiments attribution</code>, then stage it with{' '}
          <code>npm run stage:attribution</code>.
        </p>
      </section>
    );
  }
  if (!report) {
    return <p className="panel muted" role="status">Loading attribution report…</p>;
  }
  if (report.source !== 'synthetic_fault_injections' || report.cases.length === 0) {
    return <p className="notice notice-stop" role="alert">Attribution report has an unexpected shape.</p>;
  }

  const selected = report.cases.find((item) => item.name === selectedName) ?? report.cases[0]!;

  return (
    <section className="attribution-screen" aria-labelledby="attribution-heading">
      <header className="attribution-header">
        <h2 id="attribution-heading">Error attribution and coaching</h2>
        <p className="muted">
          A research demonstrator that routes injected error patterns to
          different review actions.
        </p>
      </header>

      <p className="notice" role="note">
        Synthetic fault injections only. {report.injected_causes_routed}/{report.n_injections}
        {' '}known injection scenarios reached their expected route. This is a
        logic check—not an accuracy estimate or a diagnosis of a person's error.
      </p>

      <div className="attribution-case-picker" role="group" aria-label="Injected error scenario">
        {report.cases.map((item) => (
          <button
            type="button"
            key={item.name}
            aria-pressed={selected.name === item.name}
            onClick={() => setSelectedName(item.name)}
          >
            <span>{item.name.replaceAll('_', ' ')}</span>
            <span className="small muted">
              {item.cause_label} · {item.intended.replaceAll('_', ' ')} → {item.decoded.replaceAll('_', ' ')}
            </span>
          </button>
        ))}
      </div>

      <section className="panel attribution-detail" aria-labelledby="attribution-detail-heading">
        <div className="attribution-detail-heading">
          <div>
            <h3 id="attribution-detail-heading">{selected.cause_label}</h3>
            <p className="small muted">
              Intended: {selected.intended.replaceAll('_', ' ')} · decoded:{' '}
              {selected.decoded.replaceAll('_', ' ')}
            </p>
          </div>
          <span className={selected.route_matches_injection ? 'attribution-route-check' : 'attribution-route-check is-mismatch'}>
            {selected.route_matches_injection ? 'Expected route' : 'Route mismatch'}
          </span>
        </div>

        <div className="attribution-detail-grid">
          <section aria-labelledby="attribution-evidence-heading">
            <h4 id="attribution-evidence-heading">Cause evidence scores</h4>
            <p className="small muted">
              Scores are normalized across the four candidate causes. The route
              also requires minimum absolute evidence; a weak signal is inconclusive.
            </p>
            <CauseScores scores={selected.scores} />
            <dl className="attribution-evidence">
              <div><dt>Clean-class overlap</dt><dd>{(selected.evidence.class_overlap * 100).toFixed(1)}%</dd></div>
              <div><dt>Behavior deviation / repeat spread</dt><dd>{selected.evidence.behavioral_deviation_ratio.toFixed(2)}×</dd></div>
              <div><dt>Reference correct, deployed wrong</dt><dd>{selected.evidence.reference_correct_when_deployed_wrong ? 'Yes' : 'No'}</dd></div>
              <div><dt>Centroid CUSUM / threshold</dt><dd>{selected.evidence.centroid_cusum_ratio.toFixed(2)}</dd></div>
              <div><dt>Median-frequency shift / threshold</dt><dd>{selected.evidence.median_frequency_shift_ratio.toFixed(2)}</dd></div>
            </dl>
          </section>

          <section aria-labelledby="attribution-anatomy-heading">
            <h4 id="attribution-anatomy-heading">Channel-localized feedback</h4>
            <ForearmRing channels={selected.channel_attribution} />
            {selected.coaching ? (
              <p className="attribution-coaching">{selected.coaching}</p>
            ) : (
              <p className="small muted">
                This route does not prescribe a practice cue. Follow the
                routed review action instead.
              </p>
            )}
          </section>
        </div>

        <div className="attribution-action">
          <h4>Routed next step</h4>
          <p>{selected.routed_action}</p>
        </div>
      </section>

      <details className="attribution-method">
        <summary>How the four evidence scores are formed</summary>
        <p>{report.method}</p>
      </details>
      <p className="small muted attribution-caveat">{report.caveat}</p>
    </section>
  );
}
