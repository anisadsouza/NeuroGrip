/**
 * The committed assets, checked against the code that consumes them.
 *
 * Every other test in this suite proves the code is self-consistent. This one
 * proves the built artifacts in the repository still match it. Bump
 * FEATURE_SPEC_VERSION without regenerating, and nothing else in the suite
 * notices — the app simply starts feeding a decoder a vector it was not fitted
 * on. This is the test that observes that.
 *
 * It reads artifacts/decoder.json rather than the copy under
 * apps/web/public/models/, because that directory is gitignored and absent on
 * a fresh clone.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { assertFeatureSpecCompatible } from '../src/compatibility.js';
import { FEATURE_SPEC_VERSION, featureCount, featureNames } from '../src/spec.js';

/** Read a JSON artifact by its path relative to the repository root. */
function readArtifact(relativePath: string): Record<string, unknown> {
  const url = new URL(`../../../${relativePath}`, import.meta.url);
  return JSON.parse(readFileSync(fileURLToPath(url), 'utf-8'));
}

const decoder = readArtifact('artifacts/decoder.json');
const replay = readArtifact('apps/web/public/replay/emg-replay.json');

describe('the committed decoder', () => {
  it('was exported against the feature spec this build implements', () => {
    expect(() =>
      assertFeatureSpecCompatible({
        source: 'artifacts/decoder.json',
        featureSpecVersion: decoder.feature_spec_version as number,
        nChannels: decoder.n_channels as number,
        nFeatures: decoder.n_features as number,
      }),
    ).not.toThrow();
  });

  /**
   * Ordering is the half of the contract a version number cannot carry. The
   * decoder stores the names it was fitted on; if this build would generate a
   * different sequence, every column is offset and the model still runs.
   */
  it('lists its feature names in the order this build generates them', () => {
    const expected = featureNames(decoder.n_channels as number);
    expect(decoder.feature_names).toEqual(expected);
    expect(expected.length).toBe(featureCount(decoder.n_channels as number));
  });
});

describe('the committed replay bundle', () => {
  it('was generated against the feature spec this build implements', () => {
    expect(() =>
      assertFeatureSpecCompatible({
        source: 'apps/web/public/replay/emg-replay.json',
        featureSpecVersion: replay.featureSpecVersion as number,
      }),
    ).not.toThrow();
  });

  /**
   * The bundle feeds the decoder. A bundle at eight channels and a decoder at
   * twelve would be caught inside the worker at init, but only on the machine
   * that ran it; here it fails in CI on the commit that introduced it.
   */
  it('carries the channel count the decoder expects', () => {
    expect(replay.nChannels).toBe(decoder.n_channels);
    expect(replay.featureSpecVersion).toBe(FEATURE_SPEC_VERSION);
  });
});
