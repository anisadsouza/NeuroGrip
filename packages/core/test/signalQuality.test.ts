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

  it('uses per-channel recorded-rest floors without weakening the shared default', () => {
    const channels = [
      new Float64Array(400).fill(0.8e-6),
      new Float64Array(400).fill(0),
    ];
    const calibrated = {
      ...DEFAULT_SIGNAL_QUALITY_CONFIG,
      minChannelRmsByChannel: [0.5e-6, 0.75e-6],
    };

    expect(checkSignalQuality(channels, calibrated)).toMatchObject({
      ok: false,
      reason: 'channel RMS below calibrated minimum',
      failedChannel: 1,
    });
    expect(checkSignalQuality([channels[0]!], DEFAULT_SIGNAL_QUALITY_CONFIG)).toMatchObject({
      ok: false,
      failedChannel: 0,
    });
  });

  it('rejects a nonzero channel attenuated below its recorded-rest floor', () => {
    const result = checkSignalQuality(
      [new Float64Array(400).fill(0.4e-6), new Float64Array(400).fill(1e-6)],
      {
        ...DEFAULT_SIGNAL_QUALITY_CONFIG,
        minChannelRmsByChannel: [0.5e-6, 0.75e-6],
      },
    );

    expect(result).toMatchObject({
      ok: false,
      reason: 'channel RMS below calibrated minimum',
      failedChannel: 0,
    });
  });

  it('rejects a per-channel calibration with the wrong number of channels', () => {
    const result = checkSignalQuality(good(), {
      ...DEFAULT_SIGNAL_QUALITY_CONFIG,
      minChannelRmsByChannel: [1e-6],
    });
    expect(result).toMatchObject({
      ok: false,
      reason: 'quality calibration channel count mismatch',
    });
  });

  it('rejects a channel exceeding the expected EMG range', () => {
    const channels = good();
    channels[0]![10] = 99;
    const result = checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe('channel amplitude exceeds expected EMG range');
    expect(result.failedChannel).toBe(0);
  });

  it('rejects repeated extreme samples suggesting clipping', () => {
    const channels = Array.from({ length: 8 }, (_, channel) =>
      Float64Array.from({ length: 400 }, (_, sample) =>
        100e-6 * Math.sin((2 * Math.PI * 35 * sample) / 2000 + channel),
      ),
    );
    for (const channel of channels) {
      for (let i = 0; i < channel.length; i++) {
        channel[i] = Math.max(-50e-6, Math.min(50e-6, channel[i]!));
      }
    }

    expect(checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG)).toMatchObject({
      ok: false,
      reason: 'repeated extreme samples suggest clipping',
    });
  });

  it('rejects a repeated large-step burst', () => {
    const channels = Array.from({ length: 8 }, (_, channel) =>
      Float64Array.from({ length: 400 }, (_, sample) =>
        100e-6 * Math.sin((2 * Math.PI * 35 * sample) / 2000 + channel),
      ),
    );
    for (let i = 100; i < 120; i++) {
      channels[0]![i] = channels[0]![i]! + (i % 2 ? 1e-3 : -1e-3);
    }

    expect(checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG)).toMatchObject({
      ok: false,
      reason: 'repeated large sample steps suggest transient noise',
      failedChannel: 0,
    });
  });

  it('rejects an abrupt half-window RMS change', () => {
    const channels = Array.from({ length: 8 }, (_, channel) =>
      Float64Array.from({ length: 400 }, (_, sample) => {
        const scale = sample < 200 ? 1 : 10;
        return scale * 100e-6 * Math.sin((2 * Math.PI * 35 * sample) / 2000 + channel);
      }),
    );

    expect(checkSignalQuality(channels, DEFAULT_SIGNAL_QUALITY_CONFIG)).toMatchObject({
      ok: false,
      reason: 'abrupt half-window RMS change suggests gain shift',
    });
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
