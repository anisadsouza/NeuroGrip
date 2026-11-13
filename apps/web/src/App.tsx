/**
 * Application shell.
 *
 * One frame for the whole instrument: a top bar carrying the guided path, a
 * stage that fills the rest of the window, and a slim footer. On a desktop
 * the frame is exactly one viewport tall and never scrolls; a screen whose
 * content would not fit is split into pages instead.
 *
 * The path is the presentation order an examiner should follow: the idea,
 * the live demo, how accurate it is, then the research mechanisms, ending on
 * speed, safety and limits. Every step is also directly reachable, and the
 * footer always offers the next one, so nobody has to guess where to go.
 *
 * The safety statement is not dismissible and is not a toast. The project's
 * ethical requirement (ER-4) is that any public-facing surface states plainly
 * that this is a prototype, and a notice you can dismiss is a notice that will
 * be dismissed. It sits in the footer on every screen.
 *
 * The current screen lives in the URL hash, so a link can open a screen
 * directly and the browser's back button moves between screens.
 */

import { useEffect, useState } from 'react';
import { Icon } from './components/Icon.js';
import { BrandLockup } from './components/Mark.js';
import { Attribution } from './screens/Attribution.js';
import { Cartography } from './screens/Cartography.js';
import { Clinician } from './screens/Clinician.js';
import { Glossary } from './screens/Glossary.js';
import { Live } from './screens/Live.js';
import { Overview } from './screens/Overview.js';
import { Results } from './screens/Results.js';

type Theme = 'light' | 'dark';

export type ScreenId =
  | 'overview'
  | 'live'
  | 'results'
  | 'cartography'
  | 'attribution'
  | 'review'
  | 'glossary';

interface Step {
  readonly id: ScreenId;
  /** Short label in the path. */
  readonly label: string;
  /** Full name, used for the accessible name and the Next button. */
  readonly name: string;
}

/** The guided path, in presentation order. The glossary sits off the path. */
export const PATH: readonly Step[] = [
  { id: 'overview', label: 'Idea', name: 'The idea' },
  { id: 'live', label: 'Live', name: 'Live demo' },
  { id: 'results', label: 'Accuracy', name: 'Accuracy' },
  { id: 'cartography', label: 'Gestures', name: 'Personal gestures' },
  { id: 'attribution', label: 'Errors', name: 'Why errors happen' },
  { id: 'review', label: 'Safety', name: 'Speed, safety and limits' },
];

const ALL: readonly ScreenId[] = [...PATH.map((step) => step.id), 'glossary'];

const THEME_KEY = 'neurogrip.theme';

/** Whose work this is. Order as on the project report. */
const TEAM = ['Harsh Bavaskar', "Anisa D'Souza", 'Shruti Shanklesha'];

function readStoredTheme(): Theme | null {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === 'light' || stored === 'dark' ? stored : null;
  } catch {
    // Private windows and blocked site data both throw here. A missing
    // preference is not an error; the system setting takes over.
    return null;
  }
}

function screenFromHash(): ScreenId {
  const value = window.location.hash.replace(/^#\/?/, '');
  return (ALL as readonly string[]).includes(value) ? (value as ScreenId) : 'overview';
}

export function App() {
  const [theme, setTheme] = useState<Theme | null>(readStoredTheme);
  const [screen, setScreen] = useState<ScreenId>(screenFromHash);

  useEffect(() => {
    const onHash = () => setScreen(screenFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme) {
      root.setAttribute('data-theme', theme);
      try {
        localStorage.setItem(THEME_KEY, theme);
      } catch {
        // Preference simply will not persist. Nothing else changes.
      }
    } else {
      root.removeAttribute('data-theme');
    }
  }, [theme]);

  useEffect(() => {
    const step = PATH.find((entry) => entry.id === screen);
    const name = step ? step.name : screen === 'glossary' ? 'Words explained' : '';
    document.title = name ? `NeuroGrip · ${name}` : 'NeuroGrip';
  }, [screen]);

  const open = (id: ScreenId) => {
    if (window.location.hash !== `#${id}`) window.location.hash = id;
    setScreen(id);
  };

  const resolved: Theme =
    theme ??
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

  const index = PATH.findIndex((step) => step.id === screen);
  const previous = index > 0 ? PATH[index - 1] : undefined;
  const next = index >= 0 && index < PATH.length - 1 ? PATH[index + 1] : undefined;

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="topbar-brand">
          <a
            href="#overview"
            onClick={(event) => {
              event.preventDefault();
              open('overview');
            }}
            aria-label="NeuroGrip, the idea"
          >
            <BrandLockup />
          </a>
        </h1>

        <nav className="path" aria-label="Guided path">
          <ol>
            {PATH.map((step, position) => {
              const state =
                position === index ? 'current' : index >= 0 && position < index ? 'done' : 'todo';
              return (
                <li key={step.id} data-state={state}>
                  <button
                    type="button"
                    aria-label={`Step ${position + 1}: ${step.name}`}
                    aria-current={state === 'current' ? 'step' : undefined}
                    onClick={() => open(step.id)}
                  >
                    <span className="path-index ng-num" aria-hidden="true">
                      {position + 1}
                    </span>
                    <span className="path-label">{step.label}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>

        <div className="topbar-actions">
          <button
            type="button"
            className="icon-button"
            aria-label="Words explained"
            aria-current={screen === 'glossary' ? 'page' : undefined}
            title="Every term, explained"
            onClick={() => open('glossary')}
          >
            <Icon name="glossary" />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => setTheme(resolved === 'dark' ? 'light' : 'dark')}
            aria-label={`Switch to ${resolved === 'dark' ? 'light' : 'dark'} theme`}
            title={`Switch to ${resolved === 'dark' ? 'light' : 'dark'} theme`}
          >
            <Icon name="theme" />
          </button>
        </div>
      </header>

      <main className="stage">
        {screen === 'overview' ? <Overview onOpen={open} />
          : screen === 'live' ? <Live />
            : screen === 'results' ? <Results />
              : screen === 'cartography' ? <Cartography />
                : screen === 'attribution' ? <Attribution />
                  : screen === 'glossary' ? <Glossary />
                    : <Clinician />}
      </main>

      <footer className="footline">
        <p role="note" className="footline-notice">
          Research prototype · not a medical device · no live electrodes
        </p>
        <p className="footline-team">
          <span className="visually-hidden">Built by </span>
          {TEAM.join(', ')}
        </p>
        <nav className="stepper" aria-label="Step through the path">
          {previous ? (
            <button type="button" className="button-quiet" onClick={() => open(previous.id)}>
              <Icon name="chevronLeft" size={18} />
              <span className="visually-hidden">Previous: </span>
              {previous.label}
            </button>
          ) : null}
          {next ? (
            <button type="button" className="button-primary" onClick={() => open(next.id)}>
              <span>
                <span className="visually-hidden">Next: </span>
                <span className="ng-num stepper-index" aria-hidden="true">{index + 2}</span>
                {next.name}
              </span>
              <Icon name="chevronRight" size={18} />
            </button>
          ) : screen === 'glossary' ? (
            <button type="button" className="button-quiet" onClick={() => open('overview')}>
              Back to the idea
            </button>
          ) : null}
        </nav>
      </footer>
    </div>
  );
}
