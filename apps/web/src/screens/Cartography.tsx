import { useEffect, useMemo, useState } from 'react';
import {
  assignFunctionsByRisk,
  type PrototypeFunctionAssignment,
} from '../cartography/assignment.js';
import { GESTURE_LABEL } from '../decode/commitCost.js';

interface PrototypeComparison {
  prototype: string;
  matched_catalogue_function: string;
  recall: number;
  recall_threshold: number;
  nearest_rival: string;
  separation: number;
  catalogue_nearest_rival: string;
  catalogue_separation: number;
  learnable: boolean;
  beats_catalogue_separation: boolean;
  passes: boolean;
}

interface CartographySubject {
  subject: string;
  map_points: readonly (readonly [number, number])[];
  pose_effort: readonly number[];
  prototypes_map: readonly (readonly [number, number])[];
  prototype_comparisons: readonly PrototypeComparison[];
}

interface CartographyReport {
  criterion: {
    learnable_recall: number;
    description: string;
  };
  cohorts: Readonly<Record<string, {
    wins: number;
    n_subjects: number;
    subjects: readonly CartographySubject[];
    weighted_selection: {
      separability_ratio: number;
      excitation_change: number;
      fatigue_change: number;
    };
  }>>;
  caveat: string;
}

interface DemoAssignment extends PrototypeFunctionAssignment {
  label: string;
}

function useCartographyReport() {
  const [report, setReport] = useState<CartographyReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetch('/cartography/demo.json', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Cartography demo report returned HTTP ${response.status}`);
        }
        return response.json() as Promise<CartographyReport>;
      })
      .then(setReport)
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === 'AbortError') return;
        setError(cause instanceof Error ? cause.message : 'Could not load Cartography report');
      });
    return () => controller.abort();
  }, []);

  return { report, error };
}

function MapPlot({ subject }: { subject: CartographySubject }) {
  const left = 54;
  const top = 18;
  const width = 560;
  const height = 300;
  const x = (value: number) => left + value * width;
  const y = (value: number) => top + (1 - value) * height;

  return (
    <figure className="cartography-map">
      <svg
        viewBox="0 0 640 370"
        role="img"
        aria-labelledby="cartography-map-title cartography-map-desc"
      >
        <title id="cartography-map-title">Simulated reachable activation map</title>
        <desc id="cartography-map-desc">
          A two-dimensional projection of simulated babble poses for {subject.subject}.
          Numbered rings identify synthesized prototype targets.
        </desc>
        <rect x={left} y={top} width={width} height={height} className="cartography-plot" />
        {Array.from({ length: 6 }, (_, index) => {
          const value = index / 5;
          return (
            <g key={value} className="cartography-grid">
              <line x1={x(value)} x2={x(value)} y1={top} y2={top + height} />
              <line x1={left} x2={left + width} y1={y(value)} y2={y(value)} />
              <text x={x(value)} y={top + height + 18} textAnchor="middle">{value.toFixed(1)}</text>
              <text x={left - 10} y={y(value) + 4} textAnchor="end">{value.toFixed(1)}</text>
            </g>
          );
        })}
        <text x={left + width / 2} y="360" textAnchor="middle" className="cartography-axis">
          Synergy axis 1 · display projection only
        </text>
        <text
          x="15"
          y={top + height / 2}
          textAnchor="middle"
          className="cartography-axis"
          transform={`rotate(-90 15 ${top + height / 2})`}
        >
          Synergy axis 2
        </text>
        {subject.map_points.map(([mapX, mapY], index) => {
          const effort = subject.pose_effort[index] ?? 0;
          return (
            <circle
              key={`pose-${index}`}
              cx={x(mapX)}
              cy={y(mapY)}
              r={1.4 + effort * 1.8}
              className="cartography-pose"
            />
          );
        })}
        {subject.prototypes_map.map(([mapX, mapY], index) => (
          <g key={`prototype-${index}`} className="cartography-prototype">
            <circle cx={x(mapX)} cy={y(mapY)} r="8" />
            <text x={x(mapX)} y={y(mapY) + 3.5} textAnchor="middle">{index + 1}</text>
          </g>
        ))}
      </svg>
      <figcaption className="small muted">
        Each dot is one simulated babble pose; ring numbers are selected targets.
        The 2-D map is for display, while synthesis and separation use the
        higher-dimensional representation.
      </figcaption>
    </figure>
  );
}

function PrototypeTable({
  subject,
  recallThreshold,
}: {
  subject: CartographySubject;
  recallThreshold: number;
}) {
  const assignments = useMemo(
    () => assignFunctionsByRisk(
      subject.prototype_comparisons.map((item) => item.prototype),
      subject.prototype_comparisons.map((item) => 1 - item.recall),
    ),
    [subject],
  );
  const assignmentByPrototype = new Map<string, DemoAssignment>(
    assignments.map((assignment) => [
      assignment.prototype,
      { ...assignment, label: GESTURE_LABEL[assignment.function] ?? assignment.function },
    ]),
  );

  return (
    <div className="cartography-table-wrap">
      <table className="cartography-table">
        <caption>
          Per-prototype held-out repetition gate and illustrative function assignment
        </caption>
        <thead>
          <tr>
            <th scope="col">Target</th>
            <th scope="col">Matched catalogue</th>
            <th scope="col">Held-out recall</th>
            <th scope="col">Nearest-rival separation</th>
            <th scope="col">Assignment</th>
            <th scope="col">Gate</th>
          </tr>
        </thead>
        <tbody>
          {subject.prototype_comparisons.map((item) => {
            const assignment = assignmentByPrototype.get(item.prototype);
            return (
              <tr key={item.prototype}>
                <th scope="row">{item.prototype.replace('_', ' ')}</th>
                <td>{GESTURE_LABEL[item.matched_catalogue_function] ?? item.matched_catalogue_function}</td>
                <td>{(item.recall * 100).toFixed(1)}% (≥{(recallThreshold * 100).toFixed(0)}%)</td>
                <td>
                  {item.separation.toFixed(2)} vs {item.catalogue_separation.toFixed(2)}
                  <span className="muted"> · {item.nearest_rival} / {item.catalogue_nearest_rival}</span>
                </td>
                <td>
                  {assignment?.label ?? 'Unassigned'}
                  {assignment ? ` · expected cost ${assignment.expectedCommitCost.toFixed(3)}` : ''}
                </td>
                <td>{item.passes ? 'Pass' : 'Not met'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Cartography() {
  const { report, error } = useCartographyReport();

  if (error) {
    return (
      <section className="cartography-screen" aria-labelledby="cartography-heading">
        <h2 id="cartography-heading">Neuromotor Cartography</h2>
        <p className="notice notice-stop" role="alert">{error}</p>
        <p className="small muted">
          Generate the simulator-only report with{' '}
          <code>python -m neurogrip.experiments cartography</code>, then stage it with{' '}
          <code>npm run stage:cartography</code>.
        </p>
      </section>
    );
  }
  if (!report) {
    return <p className="panel muted" role="status">Loading Cartography report…</p>;
  }

  const intact = report.cohorts.intact;
  const demo = intact?.subjects[0];
  if (!demo) {
    return <p className="notice notice-stop" role="alert">Cartography report has no intact-cohort demo.</p>;
  }
  const passes = demo.prototype_comparisons.filter((item) => item.passes).length;

  return (
    <section className="cartography-screen" aria-labelledby="cartography-heading">
      <header className="cartography-header">
        <h2 id="cartography-heading">Neuromotor Cartography</h2>
        <p className="muted">
          A simulator-only demonstration of the babble map, prototype comparison,
          and risk-weighted function assignment.
        </p>
      </header>

      <p className="notice" role="note">
        Research prototype. This view uses synthetic subjects, not a wearer’s
        electrode recording. It is not a teaching session, clinical assessment,
        or prosthesis configuration.
      </p>

      <section className="panel cartography-gate" aria-labelledby="cartography-gate-heading">
        <div>
          <h3 id="cartography-gate-heading">Per-gesture synthesis gate</h3>
          <p>
            Pass requires held-out repetition recall ≥
            {(report.criterion.learnable_recall * 100).toFixed(0)}% and better
            nearest-rival separation than the one-to-one matched catalogue gesture.
          </p>
        </div>
        <strong className="cartography-readout">
          {passes}/{demo.prototype_comparisons.length}
          <span>targets pass · {demo.subject}</span>
        </strong>
      </section>

      <section className="panel" aria-labelledby="cartography-map-heading">
        <h3 id="cartography-map-heading">Reachable activation map</h3>
        <MapPlot subject={demo} />
      </section>

      <section className="panel" aria-labelledby="cartography-prototypes-heading">
        <h3 id="cartography-prototypes-heading">Prototype evaluation and assignment</h3>
        <p className="small muted">
          Assignment minimizes held-out misclassification risk × function commit
          cost using the same commit-cost table as the decoder. It is an
          illustrative mapping, not a clinical recommendation.
        </p>
        <PrototypeTable
          subject={demo}
          recallThreshold={report.criterion.learnable_recall}
        />
      </section>

      <section className="panel cartography-cohorts" aria-labelledby="cartography-cohorts-heading">
        <h3 id="cartography-cohorts-heading">Simulation gate across cohorts</h3>
        <table className="cartography-table">
          <thead>
            <tr>
              <th scope="col">Cohort</th>
              <th scope="col">Subjects passing any prototype</th>
              <th scope="col">Weighted catalogue separability</th>
              <th scope="col">Effort change</th>
              <th scope="col">Fatigue change</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.cohorts).map(([name, cohort]) => (
              <tr key={name}>
                <th scope="row">{name.replace('_', ' ')}</th>
                <td>{cohort.wins}/{cohort.n_subjects}</td>
                <td>{(cohort.weighted_selection.separability_ratio * 100).toFixed(1)}% of max-min</td>
                <td>{(cohort.weighted_selection.excitation_change * 100).toFixed(1)} pp</td>
                <td>{(cohort.weighted_selection.fatigue_change * 100).toFixed(1)} pp</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <p className="small muted cartography-caveat">{report.caveat}</p>
    </section>
  );
}
