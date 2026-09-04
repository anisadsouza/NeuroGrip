/**
 * Application shell.
 *
 * The safety banner is not dismissible and is not a toast. It is a permanent
 * part of the frame, because the project's own ethical requirement (ER-4) is
 * that any public-facing surface states plainly that this is a prototype. A
 * notice you can dismiss is a notice that will be dismissed.
 */

import { useEffect, useState } from 'react';
import { Icon } from './components/Icon.js';
import { Live } from './screens/Live.js';

type Theme = 'light' | 'dark';

const THEME_KEY = 'neurogrip.theme';

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
      <p className="safety" role="note">
        Research prototype. Not a medical device. The signal shown is simulated,
        not a recording of a person.
      </p>

      <header className="masthead">
        <div className="masthead-title">
          <Icon name="waveform" size={22} />
          <h1>NeuroGrip</h1>
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
    </div>
  );
}
