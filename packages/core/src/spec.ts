/**
 * Canonical feature specification. Mirrors services/research/neurogrip/spec.py.
 * Bump FEATURE_SPEC_VERSION in BOTH files and regenerate fixtures on any change.
 */

export const FEATURE_SPEC_VERSION = 1;
export const AR_ORDER = 4;
export const PSD_BINS = 6;

export const PER_CHANNEL_FEATURES = [
  'rms', 'mav', 'wl', 'zc', 'ssc',
  'ar1', 'ar2', 'ar3', 'ar4',
  'psd1', 'psd2', 'psd3', 'psd4', 'psd5', 'psd6',
  'mdf', 'mnf',
] as const satisfies readonly string[];

/**
 * Channel-major ordering with a fixed intra-channel order.
 * Deliberately not lexicographic: sorting places ch10_* before ch2_*, which
 * silently permutes the vector at ten or more channels. NinaPro DB2 has twelve.
 */
export function featureNames(nChannels: number): string[] {
  if (nChannels < 1) throw new RangeError('nChannels must be at least 1');
  const names: string[] = [];
  for (let channel = 1; channel <= nChannels; channel++) {
    for (const feature of PER_CHANNEL_FEATURES) names.push(`ch${channel}_${feature}`);
  }
  return names;
}

export function featureCount(nChannels: number): number {
  if (nChannels < 1) throw new RangeError('nChannels must be at least 1');
  return nChannels * PER_CHANNEL_FEATURES.length;
}
