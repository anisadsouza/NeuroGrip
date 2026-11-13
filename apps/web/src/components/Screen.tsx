/**
 * A screen: one title, one plain-language sentence, and its content split into
 * pages that each fit the viewport.
 *
 * Pages, not scrolling, because each page answers one question and the
 * question should be readable whole. The page control is a real tab list:
 * one tab stop, arrow keys move between pages, Home and End jump to the ends.
 * Previous and Next sit beside it for anyone who would rather step through.
 *
 * The sentence under the title belongs to the page, not the screen: it changes
 * with the page so that it always says what the reader is looking at now.
 */

import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react';
import { Icon } from './Icon.js';

export interface ScreenPage {
  readonly id: string;
  readonly label: string;
  /** One sentence, in plain words, about what this page shows. */
  readonly summary: string;
  readonly content: ReactNode;
}

interface ScreenProps {
  readonly title: string;
  readonly pages: readonly ScreenPage[];
  readonly page: number;
  readonly onPage: (index: number) => void;
  /** Replaces the page body while data loads or after it fails. */
  readonly status?: ReactNode;
  /** Controls that belong to the whole screen, such as Start. */
  readonly actions?: ReactNode;
  readonly className?: string;
}

export function Screen({
  title,
  pages,
  page,
  onPage,
  status,
  actions,
  className,
}: ScreenProps) {
  const id = useId();
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  const current = Math.min(Math.max(page, 0), Math.max(pages.length - 1, 0));
  const active = pages[current];
  const paged = pages.length > 1;

  const go = (index: number, focus = false) => {
    const next = (index + pages.length) % pages.length;
    onPage(next);
    if (focus) tabs.current[next]?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const keys: Record<string, number> = {
      ArrowRight: current + 1,
      ArrowLeft: current - 1,
      Home: 0,
      End: pages.length - 1,
    };
    const target = keys[event.key];
    if (target === undefined) return;
    event.preventDefault();
    go(target, true);
  };

  return (
    <section className={`screen ${className ?? ''}`} aria-labelledby={`${id}-title`}>
      <header className="screen-head">
        <div className="screen-heading">
          <h2 id={`${id}-title`}>{title}</h2>
          {active ? <p className="screen-summary">{active.summary}</p> : null}
        </div>
        <div className="screen-controls">
          {paged ? (
            <>
              <div
                className="pager-tabs"
                role="tablist"
                aria-label={`${title} pages`}
                onKeyDown={onKeyDown}
              >
                {pages.map((item, index) => (
                  <button
                    key={item.id}
                    ref={(element) => {
                      tabs.current[index] = element;
                    }}
                    type="button"
                    role="tab"
                    id={`${id}-tab-${item.id}`}
                    aria-selected={index === current}
                    aria-controls={`${id}-panel`}
                    tabIndex={index === current ? 0 : -1}
                    onClick={() => go(index)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <div className="pager-step">
                <button
                  type="button"
                  className="pager-arrow"
                  onClick={() => go(current - 1)}
                  disabled={current === 0}
                  aria-label="Previous page"
                >
                  <Icon name="chevronLeft" size={18} />
                </button>
                <span className="ng-num pager-count" aria-live="polite">
                  {current + 1} / {pages.length}
                </span>
                <button
                  type="button"
                  className="pager-arrow"
                  onClick={() => go(current + 1)}
                  disabled={current === pages.length - 1}
                  aria-label="Next page"
                >
                  <Icon name="chevronRight" size={18} />
                </button>
              </div>
            </>
          ) : null}
          {actions}
        </div>
      </header>

      <div
        className="screen-body"
        id={`${id}-panel`}
        {...(paged
          ? { role: 'tabpanel', 'aria-labelledby': `${id}-tab-${active?.id ?? ''}` }
          : {})}
      >
        {status ?? (
          <div className="screen-page" key={active?.id}>
            {active?.content}
          </div>
        )}
      </div>
    </section>
  );
}
