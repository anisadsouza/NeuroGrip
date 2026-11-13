/**
 * A technical word with its plain meaning one hover, tap or focus away.
 *
 * The definition comes from the shared glossary, so the inline explanation
 * and the glossary screen always say the same thing. The trigger is a real
 * button: it is reachable by keyboard, and on a touch screen a tap opens it.
 */

import { useId, useState, type ReactNode } from 'react';
import { glossaryTerm } from '../glossary.js';

export function Term({ id, children }: { id: string; children?: ReactNode }) {
  const entry = glossaryTerm(id);
  const tipId = useId();
  const [open, setOpen] = useState(false);
  if (!entry) return <>{children}</>;

  return (
    <span
      className="term"
      data-story={entry.story}
      data-open={open || undefined}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        className="term-trigger"
        aria-describedby={tipId}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        onBlur={() => setOpen(false)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') setOpen(false);
        }}
      >
        {children ?? entry.term}
      </button>
      <span role="tooltip" id={tipId} className="term-tip">
        <strong>{entry.term}</strong> {entry.plain}
      </span>
    </span>
  );
}
