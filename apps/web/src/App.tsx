/**
 * Application shell.
 *
 * The safety statement is not dismissible and is not a toast. The project's
 * ethical requirement (ER-4) is that any public-facing surface states plainly
 * that this is a prototype, and a notice you can dismiss is a notice that will
 * be dismissed.
 *
 * It lives in the colophon, with the credits and the course, where a reader
 * looking for provenance finds it. It is never dismissible and never a toast.
 */

import { useEffect, useState } from 'react';
import { Icon } from './components/Icon.js';
import { Wordmark } from './components/Wordmark.js';
import { Live } from './screens/Live.js';

type Theme = 'light' | 'dark';

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

export function App() {
  const [theme, setTheme] = useState<Theme | null>(readStoredTheme);

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

  const resolved: Theme =
    theme ??
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead-title">
          <h1>
            <Wordmark />
          </h1>
          <p className="masthead-sub">Live decoding</p>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme(resolved === 'dark' ? 'light' : 'dark')}
          aria-label={`Switch to ${resolved === 'dark' ? 'light' : 'dark'} theme`}
        >
          <Icon name="theme" />
        </button>
      </header>

      <main>
        <Live />
      </main>

      <footer className="colophon">
        <div className="colophon-mark">
          <Wordmark height="1.5rem" />
          <p className="colophon-course">
            BTECH CAP 501, Project Life Cycle Management
            <br />
            ATLAS SkillTech University / uGDX
          </p>
        </div>

        <div className="colophon-team">
          <h2>Built by</h2>
          <ul>
            {TEAM.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        </div>

        <p className="colophon-safety" role="note">
          Research prototype. Not a medical device. The signal shown is
          simulated, not a recording of a person.
        </p>
      </footer>
    </div>
  );
}
