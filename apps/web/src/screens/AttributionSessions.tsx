/**
 * End-to-end routing on simulated sessions, and the anatomical coaching check.
 *
 * Both panels read Python artifacts. Neither re-runs the router in the
 * browser: a TypeScript copy of the router would be a second implementation
 * owing a second conformance gate, for a research display.
 */

import { useEffect, useState } from 'react';
import { MethodNote } from '../components/MethodNote.js';
import { GESTURE_LABEL } from '../decode/commitCost.js';

type RoutedCause = 'physiological' | 'behavioral' | 'model_capacity' | 'drift' | 'inconclusive';
type Fault = Exclude<RoutedCause, 'inconclusive'>;
type Group = 'digit_flexor' | 'digit_extensor' | 'wrist_flexor' | 'wrist_extensor';

interface Coaching {
  confident: boolean;
  group: Group | null;
  direction: 'excess' | 'deficit' | null;
  share: number;
  instruction: string;
  scope: string;
}

interface SessionExample {
  intended: string;
  decoded: string;
  cause: RoutedCause;
  cause_label: string;
  scores: Record<Fault, number>;
  routed_action: string;
  window: number;
  anatomy?: {
    muscle_groups: Record<Group, number>;
    coaching: Coaching;
  };
}

interface Session {
  fault: Fault;
  injected_cause: Fault;
  injection: string;
  deployed_model: string;
  reference_model: string;
  n_windows: number;
  n_error_windows: number;
  deployed_accuracy: number;
  reference_accuracy: number;
  routed: Record<RoutedCause, number>;
  plurality_route: RoutedCause | null;
  routes_to_injected_cause: boolean;
  examples: readonly SessionExample[];
}

interface SessionReport {
  source: 'simulated_sessions';
  seeds: readonly number[];
  sessions_routed: number;
  n_sessions: number;
  per_fault: Record<Fault, {
    sessions_routed: number;
    n_sessions: number;
    n_error_windows: number;
    n_errors_routed_to_injected_cause: number;
  }>;
  detail_seed: number;
  sessions: readonly Session[];
  method: string;
  history: string;
  caveat: string;
}

interface CoachingReport {
  source: 'simulator';
  n_subjects: number;
  n_error_windows: number;
  n_recovered: number;
  n_wrong_group_or_direction: number;
  n_declined: number;
  recovery_rate: number;
  chance_rate: number;
  per_group: Record<Group, { error_windows: number; recovered: number; recovery_rate: number | null }>;
  per_subject: readonly { subject: string; electrode_shift_rad: number; n_error_windows: number; recovery_rate: number | null }[];
  caveat: string;
}

const FAULT_ORDER: readonly Fault[] = ['physiological', 'behavioral', 'model_capacity', 'drift'];
const ROUTE_ORDER: readonly RoutedCause[] = [...FAULT_ORDER, 'inconclusive'];

const CAUSE_NAME: Readonly<Record<RoutedCause, string>> = {
  physiological: 'Body: gestures look alike',
  behavioral: 'Technique: done differently',
  model_capacity: 'Decoder: too simple',
  drift: 'Signal: changed over time',
  inconclusive: 'Not sure',
};

export const GROUP_NAME: Readonly<Record<Group, string>> = {
  digit_flexor: 'Finger flexors',
  digit_extensor: 'Finger extensors',
  wrist_flexor: 'Wrist flexors',
  wrist_extensor: 'Wrist extensors',
};
const GROUP_ORDER: readonly Group[] = ['digit_flexor', 'digit_extensor', 'wrist_flexor', 'wrist_extensor'];

function label(gesture: string): string {
  return GESTURE_LABEL[gesture] ?? gesture.replaceAll('_', ' ');
}

function percent(value: number | null, digits = 1): string {
  return value === null ? '—' : `${(value * 100).toFixed(digits)}%`;
}

function useJson<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void fetch(url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
        return response.json() as Promise<T>;
      })
      .then(setData)
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === 'AbortError') return;
        setError(cause instanceof Error ? cause.message : `Could not load ${url}`);
      });
    return () => controller.abort();
  }, [url]);
  return { data, error };
}

/**
 * Signed muscle-group evidence. Carmine, because the quantity is muscle
 * activity: bars to the right are more activity than the wearer's own
 * template, to the left less. The scale is the largest magnitude shown.
 */
export function MuscleGroupBars({ groups }: { groups: Record<Group, number> }) {
  const largest = Math.max(1e-9, ...GROUP_ORDER.map((group) => Math.abs(groups[group])));
  // Raw back-projected units have no meaning to a reader; each group's share
  // of the total evidence does, and it is what the coaching rule thresholds.
  const total = Math.max(1e-9, GROUP_ORDER.reduce((sum, group) => sum + Math.abs(groups[group]), 0));
  return (
    <div className="muscle-groups" role="list" aria-label="Estimated muscle-group deviation">
      {GROUP_ORDER.map((group) => {
        const value = groups[group];
        const width = (Math.abs(value) / largest) * 50;
        return (
          <div className="muscle-group" role="listitem" key={group}>
            <span className="muscle-group-name">{GROUP_NAME[group]}</span>
            <span className="muscle-group-track" aria-hidden="true">
              <span
                className="muscle-group-bar"
                style={value >= 0 ? { left: '50%', width: `${width}%` } : { left: `${50 - width}%`, width: `${width}%` }}
              />
            </span>
            <span className="muscle-group-value ng-num">
              {value >= 0 ? 'more' : 'less'} · {Math.round((Math.abs(value) / total) * 100)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function RouteBars({ session }: { session: Session }) {
  const total = Math.max(1, session.n_error_windows);
  return (
    <div className="attribution-scores">
      {ROUTE_ORDER.map((cause) => {
        const count = session.routed[cause] ?? 0;
        return (
          <div className="attribution-score" key={cause} data-cause={cause}>
            <div className="attribution-score-label">
              <span>{CAUSE_NAME[cause]}</span>
              <span className="ng-num">{count}/{session.n_error_windows}</span>
            </div>
            <div
              className="attribution-score-track"
              role="meter"
              aria-label={`Errors routed to ${CAUSE_NAME[cause]}`}
              aria-valuemin={0}
              aria-valuemax={session.n_error_windows}
              aria-valuenow={count}
            >
              <span style={{ width: `${(count / total) * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function SessionPanel({ view }: { view: 'summary' | 'detail' }) {
  const { data: report, error } = useJson<SessionReport>('/attribution/session.json');
  const [selected, setSelected] = useState<Fault>('behavioral');

  if (error) {
    return (
      <section className="panel" aria-labelledby="session-heading">
        <h3 id="session-heading">Simulated sessions</h3>
        <p className="notice notice-stop" role="alert">{error}</p>
        <p className="small muted">This part of the research is not included in this build. Every other screen still works.</p>
      </section>
    );
  }
  if (!report) return <p className="screen-empty muted" role="status">Loading simulated sessions…</p>;

  if (view === 'summary') {
    return (
      <section className="panel attribution-sessions" aria-labelledby="session-heading">
        <div className="attribution-detail-heading">
          <div>
            <h3 id="session-heading">Simulated sessions, end to end</h3>
            <p className="small muted">One planted fault per session.</p>
          </div>
          <strong className="cartography-readout">
            {report.sessions_routed}/{report.n_sessions}
            <span>sessions diagnosed correctly, over {report.seeds.length} runs</span>
          </strong>
        </div>

        <div className="cartography-table-wrap">
          <table className="cartography-table">
            <caption>Routing by injected fault, across all seeds</caption>
            <thead>
              <tr>
                <th scope="col">Planted cause</th>
                <th scope="col">Sessions diagnosed correctly</th>
                <th scope="col">Mistakes traced to it</th>
              </tr>
            </thead>
            <tbody>
              {FAULT_ORDER.map((fault) => {
                const row = report.per_fault[fault];
                return (
                  <tr key={fault}>
                    <th scope="row">{CAUSE_NAME[fault]}</th>
                    <td>{row.sessions_routed}/{row.n_sessions}</td>
                    <td>{row.n_errors_routed_to_injected_cause}/{row.n_error_windows}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <MethodNote>
          <p>{report.method}</p>
          <p>{report.history}</p>
          <p>{report.caveat}</p>
        </MethodNote>
      </section>
    );
  }

  const session = report.sessions.find((item) => item.fault === selected) ?? report.sessions[0]!;
  const example =
    session.examples.find((item) => item.cause === session.injected_cause && (item.anatomy || item.cause !== 'behavioral')) ??
    session.examples[0];

  return (
    <section className="attribution-sessions" aria-labelledby="session-detail-heading">
      <h3 id="session-detail-heading" className="visually-hidden">One simulated session</h3>
      <div className="attribution-case-picker" role="group" aria-label={`Simulated session, seed ${report.detail_seed}`}>
        {FAULT_ORDER.map((fault) => {
          const item = report.sessions.find((candidate) => candidate.fault === fault);
          if (!item) return null;
          return (
            <button type="button" key={fault} data-cause={fault} aria-pressed={session.fault === fault} onClick={() => setSelected(fault)}>
              <span>{CAUSE_NAME[fault]}</span>
              <span className="small muted">
                {item.n_error_windows} errors · {item.routes_to_injected_cause ? 'reached' : 'missed'}
              </span>
            </button>
          );
        })}
      </div>

      <div className="panel attribution-detail-grid">
        <section aria-labelledby="session-routes-heading">
          <h4 id="session-routes-heading">What the mistakes were blamed on</h4>
          <p className="small muted">{session.injection}</p>
          <RouteBars session={session} />
          <dl className="attribution-evidence">
            <div><dt>Decoder in use</dt><dd>{session.deployed_model} · {percent(session.deployed_accuracy)}</dd></div>
            <div><dt>Stronger comparison decoder</dt><dd>{session.reference_model} · {percent(session.reference_accuracy)}</dd></div>
            <div><dt>Most common diagnosis</dt><dd>{session.plurality_route ? CAUSE_NAME[session.plurality_route] : 'No errors'}</dd></div>
          </dl>
        </section>

        <section aria-labelledby="session-example-heading">
          <h4 id="session-example-heading">One mistake, explained</h4>
          {example ? (
            <>
              <p className="small">
                Cued <strong>{label(example.intended)}</strong>, decoded{' '}
                <strong>{label(example.decoded)}</strong> · routed to {example.cause_label.toLowerCase()}.
              </p>
              {example.anatomy ? (
                <>
                  <MuscleGroupBars groups={example.anatomy.muscle_groups} />
                  <p className="attribution-coaching">{example.anatomy.coaching.instruction}</p>
                  <p className="small muted">{example.anatomy.coaching.scope}</p>
                </>
              ) : (
                <p className="small muted">{example.routed_action}</p>
              )}
            </>
          ) : (
            <p className="small muted">This session produced no misclassified windows.</p>
          )}
        </section>
      </div>
    </section>
  );
}

export function CoachingCheck() {
  const { data: report, error } = useJson<CoachingReport>('/attribution/coaching.json');
  if (error) {
    return <p className="notice notice-stop" role="alert">{error}</p>;
  }
  if (!report) return <p className="panel muted" role="status">Loading coaching check…</p>;

  return (
    <section className="panel" aria-labelledby="coaching-heading">
      <div className="attribution-detail-heading">
        <div>
          <h3 id="coaching-heading">Does the coaching name the right muscles?</h3>
          <MethodNote>
            <p>
              A known muscle-group deviation is injected into each gesture. Every
              window the decoder then gets wrong is explained, and counts as
              recovered only if the sentence names the injected group and
              direction. Declined windows count against it.
            </p>
          </MethodNote>
        </div>
        <strong className="cartography-readout">
          {percent(report.recovery_rate)}
          <span>
            {report.n_recovered}/{report.n_error_windows} error windows · chance{' '}
            {percent(report.chance_rate)}
          </span>
        </strong>
      </div>
      <div className="cartography-table-wrap">
        <table className="cartography-table">
          <caption>Recovery by injected muscle group · {report.n_subjects} simulated subjects</caption>
          <thead>
            <tr>
              <th scope="col">Muscles we disturbed</th>
              <th scope="col">Mistakes</th>
              <th scope="col">Advice named them</th>
            </tr>
          </thead>
          <tbody>
            {GROUP_ORDER.map((group) => {
              const row = report.per_group[group];
              return (
                <tr key={group}>
                  <th scope="row">{GROUP_NAME[group]}</th>
                  <td>{row.error_windows}</td>
                  <td>{row.recovered} · {percent(row.recovery_rate)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Wrong group or direction: {report.n_wrong_group_or_direction} · declined:{' '}
        {report.n_declined} · simulated only
      </p>
      <MethodNote><p>{report.caveat}</p></MethodNote>
    </section>
  );
}
