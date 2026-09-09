import { describe, expect, it } from 'vitest';
import { assertFeatureSpecCompatible } from '../src/compatibility.js';
import { FEATURE_SPEC_VERSION, featureCount } from '../src/spec.js';

describe('assertFeatureSpecCompatible', () => {
  it('accepts an asset built against the version this build implements', () => {
    expect(() =>
      assertFeatureSpecCompatible({
        source: 'decoder.json',
        featureSpecVersion: FEATURE_SPEC_VERSION,
        nChannels: 8,
        nFeatures: featureCount(8),
      }),
    ).not.toThrow();
  });

  /**
   * The message has to name both numbers. A reader who hits this is looking at
   * an app that will not start, and the only useful thing to tell them is which
   * two versions disagree and which command reconciles them.
   */
  it('names both versions and the remedy rather than failing opaquely', () => {
    let message = '';
    try {
      assertFeatureSpecCompatible({
        source: 'decoder.json',
        featureSpecVersion: FEATURE_SPEC_VERSION + 1,
      });
    } catch (error) {
      message = (error as Error).message;
    }

    expect(message).toMatch('decoder.json');
    expect(message).toMatch(String(FEATURE_SPEC_VERSION + 1));
    expect(message).toMatch(String(FEATURE_SPEC_VERSION));
    expect(message).toMatch('npm run build:assets');
  });

  /**
   * A corpus rebuilt at eight channels and a model exported at twelve agree on
   * the spec version and still feed the decoder a differently shaped vector.
   * The version check alone would pass this.
   */
  it('rejects a channel count that disagrees with the feature count at a matching version', () => {
    expect(() =>
      assertFeatureSpecCompatible({
        source: 'decoder.json',
        featureSpecVersion: FEATURE_SPEC_VERSION,
        nChannels: 8,
        nFeatures: featureCount(12),
      }),
    ).toThrow();
  });

  it('accepts an asset that declares no shape rather than inventing one', () => {
    expect(() =>
      assertFeatureSpecCompatible({
        source: 'emg-replay.json',
        featureSpecVersion: FEATURE_SPEC_VERSION,
      }),
    ).not.toThrow();
  });
});
