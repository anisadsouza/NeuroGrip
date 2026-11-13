import { useEffect, useMemo, useState } from 'react';
import { MethodNote } from '../components/MethodNote.js';
import { Screen, type ScreenPage } from '../components/Screen.js';
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

interface TeachBackTarget {
  prototype: string;
  attempts_to_criterion: number | null;
  attempts_made: number;
  distances: readonly number[];
  on_target: readonly boolean[];
  trajectory_map: readonly (readonly [number, number])[];
  target_map: readonly [number, number];
  recall_at_exact_target: number;
  recall_after_teach_back: number;
  passes_gate: boolean;
}

interface TeachBackSummary {
  n_targets: number;
  n_learned: number;
  n_censored: number;
  median_attempts_to_criterion: number | null;
  median_recall_at_exact_target: number;
  median_recall_after_teach_back: number;
  n_passing_gate: number;
  n_passing_gate_and_retained: number;
}

interface CartographySubject {
  subject: string;
  teach_back: {
    config: { max_attempts: number; window: number; hits: number };
    targets: readonly TeachBackTarget[];
  };
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
    teach_back: TeachBackSummary;
  }>>;
  teach_back_definition: string;
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
          Muscle pattern, direction 1
        </text>
        <text
          x="15"
          y={top + height / 2}
          textAnchor="middle"
          className="cartography-axis"
          transform={`rotate(-90 15 ${top + height / 2})`}
        >
          Muscle pattern, direction 2
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
            <th scope="col">New gesture</th>
            <th scope="col">Standard gesture it replaces</th>
            <th scope="col">Recognised on new tries</th>
            <th scope="col">Gap to nearest look-alike (new vs standard)</th>
            <th scope="col">Safest hand function</th>
            <th scope="col">Better than standard?</th>
          </tr>
        </thead>
        <tbody>
          {subject.prototype_comparisons.map((item) => {
            const assignment = assignmentByPrototype.get(item.prototype);
            return (
              <tr key={item.prototype}>
                <th scope="row">{item.prototype.replace('proto_', 'Target ')}</th>
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
                <td>{item.passes ? 'Yes' : 'No'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const PLOT = { left: 54, top: 18, width: 560, height: 300 } as const;
const plotX = (value: number) => PLOT.left + value * PLOT.width;
const plotY = (value: number) => PLOT.top + (1 - value) * PLOT.height;

/** Distance to the target, attempt by attempt. A real scale, so a graticule. */
function DistanceTrace({ target, upTo }: { target: TeachBackTarget; upTo: number }) {
  const width = 560;
  const height = 120;
  const left = 44;
  const top = 10;
  const max = Math.max(...target.distances, 0.1);
  const ceiling = Math.ceil(max * 4) / 4;
  const n = target.attempts_made;
  const x = (index: number) => left + (n <= 1 ? 0 : (index / (n - 1)) * (width - left - 8));
  const y = (value: number) => top + (1 - value / ceiling) * height;
  const shown = target.distances.slice(0, upTo);
  const ticks = [0, ceiling / 2, ceiling];
  return (
    <svg
      className="teach-trace"
      viewBox={`0 0 ${width} ${height + 40}`}
      role="img"
      aria-label={`Distance to target over ${upTo} of ${n} attempts`}
    >
      {ticks.map((tick) => (
        <g key={tick} className="cartography-grid">
          <line x1={left} x2={width - 8} y1={y(tick)} y2={y(tick)} />
          <text x={left - 8} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text>
        </g>
      ))}
      <polyline
        className="teach-trace-line"
        points={shown.map((value, index) => `${x(index)},${y(value)}`).join(' ')}
      />
      {shown.map((value, index) => (
        <circle
          key={index}
          cx={x(index)}
          cy={y(value)}
          r={3}
          className={target.on_target[index] ? 'teach-hit' : 'teach-miss'}
        />
      ))}
      <text x={left + (width - left) / 2} y={height + 34} textAnchor="middle" className="cartography-axis">
        Each try, and how far it landed from the target
      </text>
    </svg>
  );
}

function TeachBackPanel({
  subject,
  definition,
}: {
  subject: CartographySubject;
  definition: string;
}) {
  const targets = subject.teach_back.targets;
  const [selectedName, setSelectedName] = useState(targets[0]?.prototype ?? '');
  const target = targets.find((item) => item.prototype === selectedName) ?? targets[0]!;
  const [attempt, setAttempt] = useState(target.attempts_made);
  const [names, setNames] = useState<Record<string, string>>({});
  const upTo = Math.min(attempt, target.attempts_made);
  const path = target.trajectory_map.slice(0, upTo);
  const current = path[path.length - 1];
  const { config } = subject.teach_back;

  const choose = (name: string) => {
    const next = targets.find((item) => item.prototype === name);
    setSelectedName(name);
    if (next) setAttempt(next.attempts_made);
  };

  return (
    <section className="panel teach-back" aria-labelledby="teach-heading">
      <h3 id="teach-heading" className="visually-hidden">Teach-back</h3>

      <div className="attribution-case-picker" role="group" aria-label="Target to teach back">
        {targets.map((item) => (
          <button
            type="button"
            key={item.prototype}
            aria-pressed={item.prototype === target.prototype}
            onClick={() => choose(item.prototype)}
          >
            <span>{names[item.prototype] || item.prototype.replace('proto_', 'Target ')}</span>
            <span className="small muted">
              {item.attempts_to_criterion === null
                ? `not learned in ${item.attempts_made}`
                : `learned in ${item.attempts_to_criterion}`}
            </span>
          </button>
        ))}
      </div>

      <div className="teach-grid">
        <figure className="cartography-map">
          <svg viewBox="0 0 640 340" role="img" aria-labelledby="teach-map-title">
            <title id="teach-map-title">
              Attempts 1 to {upTo} toward {target.prototype.replace('proto_', 'target ')} on the activation map
            </title>
            <rect x={PLOT.left} y={PLOT.top} width={PLOT.width} height={PLOT.height} className="cartography-plot" />
            {subject.map_points.map(([mapX, mapY], index) => (
              <circle key={`pose-${index}`} cx={plotX(mapX)} cy={plotY(mapY)} r={1.4} className="cartography-pose" />
            ))}
            <polyline
              className="teach-path"
              points={path.map(([mapX, mapY]) => `${plotX(mapX)},${plotY(mapY)}`).join(' ')}
            />
            {path.map(([mapX, mapY], index) => (
              <circle
                key={`attempt-${index}`}
                cx={plotX(mapX)}
                cy={plotY(mapY)}
                r={index === path.length - 1 ? 5 : 3}
                className={target.on_target[index] ? 'teach-hit' : 'teach-miss'}
              />
            ))}
            <g className="cartography-prototype">
              <circle cx={plotX(target.target_map[0])} cy={plotY(target.target_map[1])} r="10" />
            </g>
            {current ? (
              <text x={plotX(current[0]) + 9} y={plotY(current[1]) - 8} className="cartography-axis">
                {upTo}
              </text>
            ) : null}
          </svg>
          <figcaption className="small muted">Ring: target · filled: on target</figcaption>
        </figure>

        <div className="teach-side">
          <label className="teach-scrub">
            <span>Attempt {upTo} of {target.attempts_made}</span>
            <input
              type="range"
              min={1}
              max={target.attempts_made}
              value={upTo}
              onChange={(event) => setAttempt(Number(event.target.value))}
            />
          </label>
          <dl className="attribution-evidence">
            <div>
              <dt>Result</dt>
              <dd>
                {target.attempts_to_criterion === null
                  ? `Censored at ${target.attempts_made}`
                  : `Learned at attempt ${target.attempts_to_criterion}`}
              </dd>
            </div>
            <div><dt>Counts as learned</dt><dd>{config.hits} of last {config.window} on target</dd></div>
            <div><dt>Recognised, perfect attempt</dt><dd>{(target.recall_at_exact_target * 100).toFixed(1)}%</dd></div>
            <div><dt>Recognised, as learned</dt><dd>{(target.recall_after_teach_back * 100).toFixed(1)}%</dd></div>
            <div><dt>Better than standard?</dt><dd>{target.passes_gate ? 'Yes' : 'No'}</dd></div>
          </dl>
          <label className="teach-name">
            <span>Name this target</span>
            <input
              type="text"
              maxLength={32}
              value={names[target.prototype] ?? ''}
              placeholder={target.prototype.replace('proto_', 'Target ')}
              onChange={(event) => setNames({ ...names, [target.prototype]: event.target.value })}
            />
          </label>
          <p className="small muted">Not saved or sent.</p>
          <DistanceTrace target={target} upTo={upTo} />
        </div>
      </div>
      <div className="teach-definition">
        <MethodNote>
          <p>
            Learned means 3 of the last 5 tries landed closer to this target than to
            any other. &ldquo;Recognised, as learned&rdquo; is how often the decoder
            then recognises the gesture the way the learner ended up making it.
            Simulated learner only.
          </p>
        </MethodNote>
      </div>
    </section>
  );
}

const CARTOGRAPHY_SCOPE = 'Simulated people only. Not a clinical recommendation.';

export function Cartography() {
  const { report, error } = useCartographyReport();
  const [page, setPage] = useState(0);

  const intact = report?.cohorts.intact;
  const demo = intact?.subjects[0];

  const status = error ? (
    <div className="screen-empty">
      <p className="notice notice-stop" role="alert">{error}</p>
      <p className="small muted">This part of the research is not included in this build. Every other screen still works.</p>
    </div>
  ) : !report ? (
    <p className="screen-empty muted" role="status">Loading Cartography report…</p>
  ) : !demo ? (
    <p className="notice notice-stop" role="alert">Cartography report has no intact-cohort demo.</p>
  ) : undefined;

  const pages: ScreenPage[] =
    report && demo
      ? [
          {
            id: 'map',
            label: 'Map',
            summary: 'Dots: movements tried. Rings: gestures chosen.',
            content: (
              <div className="carto-map-page">
                <section className="panel" aria-labelledby="cartography-map-heading">
                  <h3 id="cartography-map-heading" className="visually-hidden">Reachable activation map</h3>
                  <MapPlot subject={demo} />
                </section>
                <aside className="carto-aside">
                  <section className="cartography-gate" aria-labelledby="cartography-gate-heading">
                    <h3 id="cartography-gate-heading">Do the new gestures beat the standard ones?</h3>
                    <strong className="cartography-readout">
                      {demo.prototype_comparisons.filter((item) => item.passes).length}/
                      {demo.prototype_comparisons.length}
                      <span>targets pass · {demo.subject}</span>
                    </strong>
                    <MethodNote label="What counts as a pass" title="Passing target">
                      <p>
                        Recognised in at least{' '}
                        {(report.criterion.learnable_recall * 100).toFixed(0)}% of fresh
                        attempts, and easier to tell apart than the standard gesture it
                        replaces.
                      </p>
                    </MethodNote>
                  </section>
                  <p className="notice small" role="note">{CARTOGRAPHY_SCOPE}</p>
                </aside>
              </div>
            ),
          },
          {
            id: 'teach',
            label: 'Teach-back',
            summary:
              'A simulated learner finds a new gesture.',
            content: <TeachBackPanel subject={demo} definition={report.teach_back_definition} />,
          },
          {
            id: 'targets',
            label: 'Targets',
            summary:
              'Each new gesture, scored.',
            content: (
              <section className="panel" aria-labelledby="cartography-prototypes-heading">
                <h3 id="cartography-prototypes-heading">The new gestures, one by one</h3>
                <PrototypeTable subject={demo} recallThreshold={report.criterion.learnable_recall} />
                <MethodNote>
                  <p>
                    Assignment minimizes held-out misclassification risk × function commit
                    cost using the same commit-cost table as the decoder. It is an
                    illustrative mapping, not a clinical recommendation.
                  </p>
                </MethodNote>
              </section>
            ),
          },
          {
            id: 'cohorts',
            label: 'Cohorts',
            summary:
              'Sixteen simulated people, two groups.',
            content: (
              <section className="panel cartography-cohorts" aria-labelledby="cartography-cohorts-heading">
                <h3 id="cartography-cohorts-heading">Across sixteen simulated people</h3>
                <div className="cartography-table-wrap">
                  <table className="cartography-table">
                    <thead>
                      <tr>
                        <th scope="col">Group</th>
                        <th scope="col">People with a better gesture</th>
                        <th scope="col">Easier set: separation kept</th>
                        <th scope="col">Effort saved</th>
                        <th scope="col">Tiredness saved</th>
                        <th scope="col">New gestures learned</th>
                        <th scope="col">Winners still recognised after learning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(report.cohorts).map(([name, cohort]) => (
                        <tr key={name}>
                          <th scope="row">{name === 'finger_atrophy' ? 'Weak finger muscles' : 'Typical forearm'}</th>
                          <td>{cohort.wins}/{cohort.n_subjects}</td>
                          <td>{(cohort.weighted_selection.separability_ratio * 100).toFixed(1)}% of max-min</td>
                          <td>{(cohort.weighted_selection.excitation_change * 100).toFixed(1)} pp</td>
                          <td>{(cohort.weighted_selection.fatigue_change * 100).toFixed(1)} pp</td>
                          <td>
                            {cohort.teach_back.n_learned}/{cohort.teach_back.n_targets}
                            {cohort.teach_back.median_attempts_to_criterion !== null
                              ? ` · median ${cohort.teach_back.median_attempts_to_criterion} attempts`
                              : ''}
                          </td>
                          <td>{cohort.teach_back.n_passing_gate_and_retained}/{cohort.teach_back.n_passing_gate}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <MethodNote><p>{report.caveat}</p></MethodNote>
              </section>
            ),
          },
        ]
      : [];

  return (
    <Screen
      title="Neuromotor Cartography"
      className="screen-cartography"
      pages={pages}
      page={page}
      onPage={setPage}
      status={status}
    />
  );
}
