# Data, licensing, and release review

**Review date:** 2026-11-04  
**Status:** Source correction received from the project owner: local DB2 files
were obtained from Kaggle. The exact listing is identified and Kaggle currently
labels it CC0, but the uploader's authority to license NinaPro's original data
and downstream redistribution permission remain unverified.

This is a technical inventory, not legal advice. It does not replace the
applicable upstream data terms, institutional review, or counsel review.

## What was reviewed

- The project owner's provided
  [Kaggle dataset URL](https://www.kaggle.com/datasets/quddusikashaf/ninapro-db2)
  and public Kaggle dataset metadata API. The API identifies the listing title
  as “Ninapro DB2,” owner/creator as Quddusi Kashaf, current version 2, last
  updated 2025-12-21, and `licenseName` as “CC0: Public Domain.” Its description
  field is empty. Metadata reviewed 2026-11-04.
- The local project tree and `data/ninapro/` for a downloaded license or
  permission statement. The local DB2 directory contains nine recordings (S1,
  S10, and S11; E1-E3), but no Kaggle terms snapshot, uploader authorization,
  or upstream permission statement was found.
- The [NinaPro home page](https://ninapro.hevs.ch/), which describes NinaPro as
  a publicly available research database and asks users to cite the relevant
  publication.
- The [NinaPro DB2 instructions](https://ninapro.hevs.ch/instructions/DB2.html),
  which describe DB2's 40 intact subjects, 49 movements, acquisition protocol,
  and data fields.
- The repository's `.gitignore`, which excludes `data/`, MATLAB `.mat` files,
  NinaPro replay/model directories, and NinaPro-named JSON artifacts.
- The build workflow and README warning that local NinaPro replay/model assets
  can be copied into static web output when building the application.
- The current GitHub repository metadata. `anisadsouza/NeuroGrip` is public as
  of this review. The current local Git index contains no matching raw `.mat`,
  NinaPro replay/model, or NinaPro-named JSON paths; this check does not audit
  every historical commit or release.

## Finding

The project owner reports that the local files came from the identified Kaggle
listing rather than directly from a NinaPro registration. Kaggle's current
metadata labels the listing CC0 and names Quddusi Kashaf as owner/creator. The
listing has no description field. This establishes what Kaggle currently
displays; it does **not** establish that the uploader owned or had permission
to license the NinaPro source data, that the local files are the exact listing
version, or which terms were displayed when the files were downloaded.

The public pages reviewed provide dataset descriptions and citation guidance.
They did not state a license or grant permission to redistribute the raw
recordings, a replay bundle, trained model weights, or other derived assets.
“Publicly available” is not treated here as a redistribution license.

**The listing currently displays a CC0 label, but the uploader's authority and
any upstream redistribution grant have not been verified. Therefore, the CC0
label alone is not sufficient evidence that the underlying NinaPro data, or
derived replay/model files, may be redistributed.** This does not imply
wrongdoing by the uploader; NinaPro's applicable terms or written permission
are needed to assess the rights chain.
The local `.gitignore` is a useful safeguard, but does not prevent a developer
from force-adding ignored files or publishing a build directory.

The NinaPro pages reviewed request citation. If the project reports results
using DB2, include the dataset citation listed in the README and any additional
attribution required by the applicable upstream terms.

The GitHub repository is currently public. Do not commit the local
`docs/patent_disclosure_drafts.md` working document or other new invention
details to this public repository. Existing public technical material may
already count as a disclosure; patent counsel must review the public history
and applicable jurisdictions. No repository visibility change was made.

## Release status and required checks

Until the applicable rights and terms are confirmed:

- Keep raw `.mat` files, NinaPro-derived replay binaries, model weights, and
  derived artifacts private and out of commits, releases, public storage, and
  deployed static bundles.
- Preserve the Kaggle dataset URL, uploader identity, dataset version, download
  date, a copy/screenshot of the listing and its CC0 label, and any
  accompanying source/permission statements.
- Verify the uploader's redistribution authority against NinaPro's own terms
  or a written grant from the rights holder; a Kaggle license label by itself
  does not prove that the uploader could license NinaPro source data.
- Do not assume that aggregate metrics, feature explanations, or model weights
  are exempt from the agreement; check the exact terms for derived data and
  trained models.
- Before a web build intended for publication, verify whether local ignored
  NinaPro assets are staged under `apps/web/public/` or copied into `dist/`.
- Check the Git remote's visibility and the contents of prior commits/releases;
  `.gitignore` does not remove previously tracked data.

Before any public source or binary release, also:

1. Retain the exact Kaggle listing and metadata snapshot, including version,
   license label, owner, and download date; compare the local files against the
   listing's manifest/version if possible.
2. Obtain evidence that Quddusi Kashaf was authorized by NinaPro/the relevant
   rights holder to redistribute the original DB2 data under CC0. A Kaggle
   license label alone does not establish this rights chain.
3. Review the upstream NinaPro terms that applied to this dataset and ask
   NinaPro/the data provider or institution for written clarification if those
   terms do not clearly permit third-party redistribution or derived models.
4. Ask the Kaggle uploader for the source/provenance of the upload and any
   written permission, if appropriate.
5. Audit the final commit, release archive, static output, and hosted assets for
   raw or derived data.
6. Confirm the repository's own source-code license. No root `LICENSE` file
   was present in the reviewed tree, so this review does not grant a license to
   reuse NeuroGrip code.
7. Review third-party package licenses and notices as a separate release task;
   this was not a complete dependency/SBOM audit.
8. Confirm team, course, and institutional ownership/disclosure requirements
   before publicizing technical details or filing IP.

## Sources

- [NinaPro](https://ninapro.hevs.ch/) — public dataset overview and citation
  request; reviewed 2026-11-04.
- [NinaPro DB2 instructions](https://ninapro.hevs.ch/instructions/DB2.html) —
  dataset composition and acquisition details; reviewed 2026-11-04.
- [Kaggle listing: Ninapro DB2](https://www.kaggle.com/datasets/quddusikashaf/ninapro-db2)
  — metadata API reports Quddusi Kashaf, version 2, CC0: Public Domain, and no
  description; reviewed 2026-11-04. The listing does not establish the
  uploader's authority over the upstream data.
- [Kaggle dataset metadata API](https://www.kaggle.com/api/v1/datasets/view/quddusikashaf/ninapro-db2)
  — listing metadata; reviewed 2026-11-04.
- [GitHub repository](https://github.com/anisadsouza/NeuroGrip) — metadata
  reports public visibility; reviewed 2026-11-04.
- Atzori et al., “Electromyography data for non-invasive naturally-controlled
  robotic hand prostheses,” *Scientific Data* 1, 140053 (2014),
  [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53).
