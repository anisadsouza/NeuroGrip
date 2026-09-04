import { describe, expect, it } from 'vitest';
import { DEFAULT_SIGNAL_QUALITY_CONFIG, checkSignalQuality } from '../src/signalQuality.js';

const good = () => [new Float64Array(400).fill(0.01), new Float64Array(400).fill(0.02)];

describe('checkSignalQuality', () => {
  it('accepts a healthy window', () => {
    expect(checkSignalQuality(good(), DEFAULT_SIGNAL_QUALITY_CONFIG).ok).toBe(true);
  });

  it('accepts a rest window carrying only baseline noise', () => {
    // ~15 uV baseline. The detachment gate must sit below this, not above it.
    const rest = [new Float64Array(400).fill(1.5e-5), new Float64Array(400).fill(1.5e-5)];
    expect(checkSignalQuality(rest, DEFAULT_SIGNAL_QUALITY_CONFIG).ok).toBe(true);
  });

  it('rejects a detached electrode and names it', () => {
    const channels = good();
    channels[1] = new Float64Array(400);
    const result = checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('channel RMS below noise floor');
    expect(result.failedChannel).toBe(1);
  });

  it('rejects a channel exceeding the expected EMG range', () => {
    const channels = good();
    channels[0]![10] = 99;
    const result = checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('channel amplitude exceeds expected EMG range');
    expect(result.failedChannel).toBe(0);
  });

  it('rejects non-finite samples before any other check', () => {
    const channels = good();
    channels[0]![5] = Number.NaN;
    const result = checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('channel contains non-finite values');
  });

  it('rejects an empty channel list', () => {
    expect(checkSignalQuality([], DEFAULT_SIGNAL_QUALITY_CONFIG).ok).toBe(false);
  });
});
