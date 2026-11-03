# Data, licensing, and release review

**Review date:** 2026-11-04  
**Status:** Internal review; NinaPro redistribution permission is unverified.

This is a technical inventory, not legal advice. It does not replace the
agreement supplied to the registered downloader, institutional review, or
counsel review.

## What was reviewed

- The local project tree and `data/ninapro/` for a registered-download
  agreement, license, or terms file. The local DB2 directory contains nine
  recordings (S1, S10, and S11; E1-E3) and no accompanying agreement was found.
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

The public pages reviewed provide dataset descriptions and citation guidance.
They did not state a license or grant permission to redistribute the raw
recordings, a replay bundle, trained model weights, or other derived assets.
“Publicly available” is not treated here as a redistribution license.

**The registered-download terms were not available for review. Therefore,
permission to publish or redistribute DB2-derived files is not verified.**
The local `.gitignore` is a useful safeguard, but does not prevent a developer
from force-adding ignored files or publishing a build directory.

The NinaPro pages reviewed request citation. If the project reports results
using DB2, include the dataset citation listed in the README and any additional
attribution required by the downloader agreement.

The GitHub repository is currently public. Do not commit the local
`docs/patent_disclosure_drafts.md` working document or other new invention
details to this public repository. Existing public technical material may
already count as a disclosure; patent counsel must review the public history
and applicable jurisdictions. No repository visibility change was made.

## Release status and required checks

Until the agreement is obtained and reviewed:

- Keep raw `.mat` files, NinaPro-derived replay binaries, model weights, and
  derived artifacts private and out of commits, releases, public storage, and
  deployed static bundles.
- Do not assume that aggregate metrics, feature explanations, or model weights
  are exempt from the agreement; check the exact terms for derived data and
  trained models.
- Before a web build intended for publication, verify whether local ignored
  NinaPro assets are staged under `apps/web/public/` or copied into `dist/`.
- Check the Git remote's visibility and the contents of prior commits/releases;
  `.gitignore` does not remove previously tracked data.

Before any public source or binary release, also:

1. Obtain the registered-download agreement and record its version/date.
2. Ask the data provider or institution if it does not clearly address
   redistribution of derived replay and model files.
3. Audit the final commit, release archive, static output, and hosted assets for
   raw or derived data.
4. Confirm the repository's own source-code license. No root `LICENSE` file
   was present in the reviewed tree, so this review does not grant a license to
   reuse NeuroGrip code.
5. Review third-party package licenses and notices as a separate release task;
   this was not a complete dependency/SBOM audit.
6. Confirm team, course, and institutional ownership/disclosure requirements
   before publicizing technical details or filing IP.

## Sources

- [NinaPro](https://ninapro.hevs.ch/) — public dataset overview and citation
  request; reviewed 2026-11-04.
- [NinaPro DB2 instructions](https://ninapro.hevs.ch/instructions/DB2.html) —
  dataset composition and acquisition details; reviewed 2026-11-04.
- [GitHub repository](https://github.com/anisadsouza/NeuroGrip) — metadata
  reports public visibility; reviewed 2026-11-04.
- Atzori et al., “Electromyography data for non-invasive naturally-controlled
  robotic hand prostheses,” *Scientific Data* 1, 140053 (2014),
  [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53).
