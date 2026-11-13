/**
 * The caveat, one click away.
 *
 * Every result in this interface carries its method and its limits, and none
 * of them may be dropped: the project's rule is that a number never travels
 * without the sentence that qualifies it. But a wall of caveats on every page
 * buries the result it qualifies. So the page shows one short line - what
 * kind of evidence this is - and the full text opens on demand.
 *
 * Built on the native Popover API: the note renders in the top layer, so it
 * is never clipped by a panel, never pushes the layout, and light-dismisses
 * on Escape or an outside click with no script of ours.
 */

import { useId, type ReactNode } from 'react';
import { Icon } from './Icon.js';

export function MethodNote({
  label = 'Method and limits',
  title,
  children,
}: {
  /** The trigger's visible text. Short: it sits under a result. */
  readonly label?: string;
  /** Heading inside the note. Defaults to the label. */
  readonly title?: string;
  readonly children: ReactNode;
}) {
  const id = useId().replace(/:/g, '');
  return (
    <>
      <button type="button" className="method-trigger" popoverTarget={`note-${id}`}>
        <Icon name="info" size={16} />
        {label}
      </button>
      <div id={`note-${id}`} popover="auto" className="method-note" role="dialog" aria-labelledby={`note-${id}-title`}>
        <div className="method-note-head">
          <h3 id={`note-${id}-title`}>{title ?? label}</h3>
          <button
            type="button"
            className="icon-button icon-button-small"
            popoverTarget={`note-${id}`}
            popoverTargetAction="hide"
            aria-label="Close"
          >
            <Icon name="close" size={16} />
          </button>
        </div>
        <div className="method-note-body">{children}</div>
      </div>
    </>
  );
}
