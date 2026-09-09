/**
 * docs/walkthrough.md tells a reader that every control is reachable by
 * keyboard. Before this file existed that sentence was half true: the group
 * announced itself as a radiogroup, which promises arrow-key navigation and a
 * single tab stop, and implemented neither. A screen-reader user heard "radio,
 * 1 of 10" and then found the keys that would move through them did nothing.
 *
 * The events come from user-event rather than fireEvent deliberately. A
 * synthesised keydown would satisfy a handler while leaving focus where it
 * was, so a broken roving tabindex would pass -- and the roving tabindex is
 * the half of the contract a keyboard user actually feels.
 */

import { cleanup, render, screen } from '@testing-library/react';
import { useState } from 'react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import { GestureChoices } from '../../src/components/GestureChoices.js';

afterEach(cleanup);

const LABELS = ['Rest', 'Close fist', 'Open hand', 'Pinch'];

/** A stateful host, so selection drives a real re-render the way Live does. */
function Host({ initial, chosen }: { initial: number; chosen: number[] }) {
  const [selected, setSelected] = useState(initial);
  return (
    <GestureChoices
      labels={LABELS}
      selected={selected}
      onSelect={(index) => {
        chosen.push(index);
        setSelected(index);
      }}
      legend="Intended gesture"
    />
  );
}

function renderGroup(initial = 0) {
  const chosen: number[] = [];
  render(<Host initial={initial} chosen={chosen} />);
  return { chosen };
}

describe('GestureChoices: the radiogroup contract', () => {
  /**
   * Ten gestures behind ten tab stops means a keyboard user pressing tab ten
   * times to reach the transport buttons. The roving tabindex is what makes
   * the group one stop, and it is invisible until someone tries to tab past.
   */
  it('presents the group as one tab stop rather than one per option', () => {
    renderGroup(1);
    const radios = screen.getAllByRole('radio');

    const tabbable = radios.filter((radio) => radio.getAttribute('tabindex') === '0');
    expect(tabbable.length).toBe(1);
    expect(tabbable[0]!.textContent).toBe(LABELS[1]);
    for (const radio of radios) {
      if (radio !== tabbable[0]) {
        expect(radio.getAttribute('tabindex'), radio.textContent ?? '').toBe('-1');
      }
    }
  });

  it('moves and selects on the arrow keys rather than only moving focus', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(0);

    await user.tab();
    await user.keyboard('{ArrowDown}');

    expect(chosen).toEqual([1]);
    expect(document.activeElement?.textContent).toBe(LABELS[1]);
    expect(screen.getAllByRole('radio')[1]!.getAttribute('aria-checked')).toBe('true');
  });

  it('treats the horizontal arrows the same as the vertical ones', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(0);

    await user.tab();
    await user.keyboard('{ArrowRight}');
    expect(chosen.at(-1)).toBe(1);

    await user.keyboard('{ArrowLeft}');
    expect(chosen.at(-1)).toBe(0);
  });

  /**
   * Wrapping, not stopping. A wearer holding the arrow key to scan the list
   * should not have to notice which end they started from.
   */
  it('wraps from the last option to the first rather than stopping at the end', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(LABELS.length - 1);

    await user.tab();
    await user.keyboard('{ArrowDown}');

    expect(chosen).toEqual([0]);
    expect(document.activeElement?.textContent).toBe(LABELS[0]);
  });

  it('wraps backwards from the first option to the last', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(0);

    await user.tab();
    await user.keyboard('{ArrowUp}');

    expect(chosen).toEqual([LABELS.length - 1]);
  });

  it('selects on Home and End rather than only focusing them', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(1);

    await user.tab();
    await user.keyboard('{End}');
    expect(chosen.at(-1)).toBe(LABELS.length - 1);
    expect(document.activeElement?.textContent).toBe(LABELS.at(-1));

    await user.keyboard('{Home}');
    expect(chosen.at(-1)).toBe(0);
    expect(document.activeElement?.textContent).toBe(LABELS[0]);
  });

  /**
   * Each key press is one change of intent. Firing twice would clear the
   * decoder's accumulated evidence twice, which is harmless here and would not
   * be if the handler ever did something less reversible.
   */
  it('reports one selection per key press rather than one per render', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(0);

    await user.tab();
    await user.keyboard('{ArrowDown}');

    expect(chosen.length).toBe(1);
  });

  it('leaves an unhandled key to the page rather than swallowing it', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(0);

    await user.tab();
    await user.keyboard('{Escape}');

    expect(chosen).toEqual([]);
  });

  it('still selects on a click, for the wearer who is not using a keyboard', async () => {
    const user = userEvent.setup();
    const { chosen } = renderGroup(0);

    await user.click(screen.getByText(LABELS[2]!));

    expect(chosen).toEqual([2]);
  });
});
