/**
 * An icon beside text that announces itself is a defect, not a nicety: the
 * screen reader says "alert Signal rejected" where the sighted reader sees
 * one warning. The branch that decides this is three lines long and has no
 * visible symptom when it breaks, which is exactly why it wants a gate.
 */

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { Icon } from '../../src/components/Icon.js';

afterEach(cleanup);

describe('Icon', () => {
  it('hides itself from assistive technology when it carries no label', () => {
    const { container } = render(<Icon name="alert" />);
    const svg = container.querySelector('svg');

    expect(svg?.getAttribute('aria-hidden')).toBe('true');
    expect(svg?.getAttribute('role')).toBeNull();
  });

  it('announces itself as an image when it is the sole carrier of meaning', () => {
    const { container } = render(<Icon name="alert" label="Signal rejected" />);
    const svg = container.querySelector('svg');

    expect(svg?.getAttribute('role')).toBe('img');
    expect(svg?.getAttribute('aria-label')).toBe('Signal rejected');
    expect(svg?.getAttribute('aria-hidden')).toBeNull();
  });
});
