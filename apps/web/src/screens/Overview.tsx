/**
 * Step 1, the idea: what NeuroGrip is, shown rather than described, and the
 * one thing an examiner should leave with - that it is more than a gesture
 * classifier.
 *
 * Three parts. The claim and the four numbers that back it; the hero film,
 * one gesture travelling muscle to hand; and the four mechanisms, each with
 * its own measured result and an honest status (built into the live app, or
 * demonstrated in simulation only).
 *
 * Every number is read from a staged artifact, never typed in. While an
 * artifact is missing the value shows an ellipsis rather than a guess.
 */

import { useEffect, useState, type MouseEvent } from 'react';
import { Icon } from '../components/Icon.js';
import { Term } from '../components/Term.js';
import { HeroPlayer } from '../hero/HeroPlayer.js';
import type { ScreenId } from '../App.js';

interface Facts {
  accuracy: number | null;
  ece: number | null;
  browserP95Ms: number | null;
  ttumP50Ms: number | null;
  cwer: number | null;
  errorRate: number | null;
  cartographyIntact: string | null;
  cartographyAtrophy: string | null;
  sessionsRouted: string | null;
  coachingRate: number | null;
  coachingChance: number | null;
}

async function optionalJson<T>(url: string, signal: AbortSignal): Promise<T | null> {
  try {
    const response = await fetch(url, { signal });
    if (!response.ok) return null;
    if (response.headers.get('content-type')?.includes('text/html')) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

type Comparison = {
  best_model: string;
  models: Record<string, { mean_accuracy: number; expected_calibration_error: number }>;
};
type Ttum = {
  anyMotion: { p50Ms: number | null };
  cwer: { riskWeighted: { cwer: number; errorRate: number } };
};
type CartographyDemo = { cohorts: Record<string, { wins: number; n_subjects: number }> };
type Sessions = { sessions_routed: number; n_sessions: number };
type Coaching = { recovery_rate: number; chance_rate: number };

function useFacts(): Facts | null {
  const [facts, setFacts] = useState<Facts | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    void Promise.all([
      optionalJson<Comparison>('/results/model_comparison.json', signal),
      optionalJson<{ p95Ms: number }>('/results/browser_latency.json', signal),
      optionalJson<Ttum>('/clinician/ttum.json', signal),
      optionalJson<CartographyDemo>('/cartography/demo.json', signal),
      optionalJson<Sessions>('/attribution/session.json', signal),
      optionalJson<Coaching>('/attribution/coaching.json', signal),
    ]).then(([comparison, latency, ttum, cartography, sessions, coaching]) => {
      if (signal.aborted) return;
      const best = comparison?.models[comparison.best_model];
      const cohort = (name: string) => {
        const entry = cartography?.cohorts[name];
        return entry ? `${entry.wins}/${entry.n_subjects}` : null;
      };
      setFacts({
        accuracy: best?.mean_accuracy ?? null,
        ece: best?.expected_calibration_error ?? null,
        browserP95Ms: latency?.p95Ms ?? null,
        ttumP50Ms: ttum?.anyMotion.p50Ms ?? null,
        cwer: ttum?.cwer.riskWeighted.cwer ?? null,
        errorRate: ttum?.cwer.riskWeighted.errorRate ?? null,
        cartographyIntact: cohort('intact'),
        cartographyAtrophy: cohort('finger_atrophy'),
        sessionsRouted: sessions ? `${sessions.sessions_routed}/${sessions.n_sessions}` : null,
        coachingRate: coaching?.recovery_rate ?? null,
        coachingChance: coaching?.chance_rate ?? null,
      });
    });
    return () => controller.abort();
  }, []);
  return facts;
}

const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const show = <T,>(value: T | null | undefined, format: (v: T) => string) =>
  value === null || value === undefined ? '…' : format(value);

interface Mechanism {
  readonly id: string;
  readonly name: string;
  readonly term: string;
  readonly plain: string;
  readonly value: string;
  readonly unit: string;
  readonly status: 'Built' | 'Simulation';
  readonly target: ScreenId;
}

function mechanisms(facts: Facts | null): readonly Mechanism[] {
  return [
    {
      id: 'evidence',
      name: 'Risk-weighted evidence',
      term: 'evidence',
      plain: 'Risky grips must clear a higher bar. The hand moves early and can retract until it latches.',
      value: show(facts?.cwer, (v) => percent(v, 2)),
      unit: `cost-weighted error (plain ${show(facts?.errorRate, (v) => percent(v, 2))})`,
      status: 'Built',
      target: 'live',
    },
    {
      id: 'cartography',
      name: 'Neuromotor Cartography',
      term: 'cartography',
      plain: "Maps a person's reachable muscle patterns and designs gestures that fit them.",
      value: show(facts?.cartographyAtrophy, (v) => v),
      unit: `weak-finger subjects helped (intact ${show(facts?.cartographyIntact, (v) => v)})`,
      status: 'Simulation',
      target: 'cartography',
    },
    {
      id: 'router',
      name: 'Error-attribution router',
      term: 'attribution',
      plain: 'Sends each mistake to its cause: the body, the technique, the decoder, or drift.',
      value: show(facts?.sessionsRouted, (v) => v),
      unit: 'sessions routed to the right cause',
      status: 'Simulation',
      target: 'attribution',
    },
    {
      id: 'coaching',
      name: 'Anatomical coaching',
      term: 'attribution',
      plain: 'Names the muscle group behind an error, in one sentence the wearer can act on.',
      value: show(facts?.coachingRate, (v) => percent(v)),
      unit: `errors traced to the right muscle (chance ${show(facts?.coachingChance, (v) => percent(v))})`,
      status: 'Simulation',
      target: 'attribution',
    },
  ];
}

export function Overview({ onOpen }: { onOpen: (screen: ScreenId) => void }) {
  const facts = useFacts();

  const go = (screen: ScreenId) => (event: MouseEvent) => {
    event.preventDefault();
    onOpen(screen);
  };

  return (
    <section className="overview" aria-labelledby="overview-title">
      <div className="overview-intro">
        <h2 id="overview-title">Reads the forearm. Moves only when sure.</h2>
        <p className="overview-lede">
          <Term id="semg">sEMG</Term> decoding for prosthetic hands, plus four
          mechanisms a classifier lacks.
        </p>
        <a className="button-primary" href="#live" onClick={go('live')}>
          Start the live demo
          <Icon name="chevronRight" size={18} />
        </a>

        <dl className="facts">
          <div>
            <dt><Term id="unseen">LOSO</Term> accuracy</dt>
            <dd className="ng-num">{show(facts?.accuracy, (v) => percent(v))}</dd>
          </div>
          <div>
            <dt><Term id="calibration">ECE</Term></dt>
            <dd className="ng-num">{show(facts?.ece, (v) => v.toFixed(3))}</dd>
          </div>
          <div>
            <dt>P95 <Term id="latency">latency</Term></dt>
            <dd className="ng-num">{show(facts?.browserP95Ms, (v) => `${v.toFixed(1)} ms`)}</dd>
          </div>
          <div>
            <dt>Median <Term id="ttum">TTUM</Term></dt>
            <dd className="ng-num">{show(facts?.ttumP50Ms, (v) => `${v} ms`)}</dd>
          </div>
        </dl>
        <p className="facts-note">
          Simulated subjects. Not yet tested on people.
        </p>
      </div>

      <HeroPlayer />

      <section className="mechanisms" aria-labelledby="mechanisms-title">
        <h3 id="mechanisms-title">Beyond a classifier</h3>
        <ol>
          {mechanisms(facts).map((item, index) => (
            <li key={item.id}>
              <a href={`#${item.target}`} onClick={go(item.target)} className="mechanism" title={item.plain}>
                <span className="mechanism-head">
                  <span className="mechanism-index ng-num" aria-hidden="true">{index + 1}</span>
                  <span className="mechanism-name">{item.name}</span>
                  <span className="mechanism-status" data-status={item.status}>
                    {item.status}
                  </span>
                </span>
                <span className="mechanism-metric">
                  <span className="mechanism-value ng-num">{item.value}</span>
                  <span className="mechanism-unit">{item.unit}</span>
                </span>
              </a>
            </li>
          ))}
        </ol>
      </section>
    </section>
  );
}
