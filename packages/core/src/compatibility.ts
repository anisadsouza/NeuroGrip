/**
 * The gate against a stale asset.
 *
 * Python trains the model and TypeScript feeds it at inference, so the feature
 * vector's contents and its ordering are a contract between two codebases.
 * FEATURE_SPEC_VERSION is how that contract is versioned, and it is carried in
 * three places: the decoder metadata, the replay manifest, and this package's
 * own constant. Nothing compared any pair of them until this file existed.
 *
 * The failure it prevents has no symptom. A decoder exported under an older
 * spec loads without complaint, receives a vector whose columns mean something
 * different from what it was fitted on, and returns confident nonsense.
 * Accuracy degrades and no error is raised anywhere. That is the same failure
 * the conformance suite exists to prevent, arriving by a different route: not
 * two implementations disagreeing, but one implementation moving on and
 * leaving a built artifact behind.
 */

import { FEATURE_SPEC_VERSION, featureCount } from './spec.js';

export interface AssetDescriptor {
  /** Where this came from, named as the reader would recognise it. */
  readonly source: string;
  readonly featureSpecVersion: number;
  readonly nChannels?: number;
  readonly nFeatures?: number;
}

/**
 * Throws if the asset was built against a different feature specification than
 * this package implements, or if its own declared shape is self-inconsistent.
 *
 * The channel check is not redundant with the version check. A corpus rebuilt
 * at eight channels and a model exported at twelve share a spec version and
 * still produce the same silent degradation, so both are refused here.
 */
export function assertFeatureSpecCompatible(asset: AssetDescriptor): void {
  if (asset.featureSpecVersion !== FEATURE_SPEC_VERSION) {
    throw new Error(
      `${asset.source} was built against feature spec version ` +
        `${asset.featureSpecVersion}, but this build implements ` +
        `${FEATURE_SPEC_VERSION}. The feature vector would mean something ` +
        `different from what the decoder was fitted on, with no other symptom. ` +
        `Regenerate the assets: npm run build:assets`,
    );
  }

  if (asset.nChannels !== undefined && asset.nFeatures !== undefined) {
    const expected = featureCount(asset.nChannels);
    if (asset.nFeatures !== expected) {
      throw new Error(
        `${asset.source} declares ${asset.nFeatures} features for ` +
          `${asset.nChannels} channels, but the feature set gives ${expected}. ` +
          `Regenerate the assets: npm run build:assets`,
      );
    }
  }
}
