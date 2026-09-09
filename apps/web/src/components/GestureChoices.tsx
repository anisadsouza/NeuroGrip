/**
 * The wearer's declaration of intent.
 *
 * A radio group, and it has to behave like one. ARIA's radiogroup role is a
 * promise to assistive technology that arrow keys will move between the
 * options and that the group is a single tab stop; markup that claims the role
 * without honouring the contract is worse than plain buttons, because a screen
 * reader announces "radio, 1 of 10" and then the keys that would move through
 * them do nothing.
 *
 * Selection follows focus, per the ARIA authoring practices. One consequence
 * is worth naming rather than hiding: choosing a gesture clears the decoder's
 * accumulated evidence, so arrowing across the whole list resets the
 * accumulator once per option. That is correct on both sides -- the wearer did
 * change their intent each time -- but it looks like a fault when watched.
 */

import { useEffect, useRef } from 'react';

export interface GestureChoicesProps {
  readonly labels: readonly string[];
  readonly selected: number;
  readonly onSelect: (index: number) => void;
  readonly legend: string;
}

export function GestureChoices({ labels, selected, onSelect, legend }: GestureChoicesProps) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  // Only move focus for a key the wearer pressed. Focusing on every render
  // would drag focus back here whenever a decision arrives, twenty times a
  // hop, which would make the rest of the page unreachable.
  const shouldFocus = useRef(false);

  useEffect(() => {
    if (!shouldFocus.current) return;
    shouldFocus.current = false;
    buttons.current[selected]?.focus();
  }, [selected]);

  /** Move to `index`, wrapping, and select it: selection follows focus here. */
  const move = (index: number): void => {
    const wrapped = ((index % labels.length) + labels.length) % labels.length;
    shouldFocus.current = true;
    onSelect(wrapped);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>): void => {
    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        move(selected + 1);
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        move(selected - 1);
        break;
      case 'Home':
        move(0);
        break;
      case 'End':
        move(labels.length - 1);
        break;
      default:
        return;
    }
    // Otherwise the arrow keys scroll the page out from under the group.
    event.preventDefault();
  };

  return (
    <div
      className="gesture-choices"
      role="radiogroup"
      aria-label={legend}
      onKeyDown={onKeyDown}
    >
      {labels.map((label, index) => (
        <button
          key={label}
          type="button"
          role="radio"
          aria-checked={index === selected}
          // Roving tabindex: the group is one tab stop, so tabbing past it
          // does not mean pressing tab ten times.
          tabIndex={index === selected ? 0 : -1}
          ref={(element) => {
            buttons.current[index] = element;
          }}
          className="gesture-choice"
          data-selected={index === selected || undefined}
          onClick={() => onSelect(index)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
