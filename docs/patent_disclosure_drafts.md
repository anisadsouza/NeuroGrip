# NeuroGrip — internal patent-disclosure working drafts

**Status:** Drafts for inventor/team and qualified patent-counsel review only.  
**Date prepared:** 2026-11-04  
**Not a patent application, novelty opinion, or legal advice.**

Do not treat this document as confidential merely because it is labeled
“internal.” Keep it and related technical material private. Avoid public
repositories, presentations, posters, papers, demos to unrestricted audiences,
or other disclosures until the team/institution and patent counsel decide what
to file and when. Public-disclosure consequences vary by jurisdiction.

The project design specification contains preliminary prior-art notes, not a
professional search. Nothing here states that an idea is novel, patentable, or
owned by any particular person.

## Before submission to counsel

- Confirm the actual inventors by contribution to conception, not by project
  membership or authorship alone.
- Confirm each contributor's institution, employment, course, grant, and
  assignment obligations.
- Record conception dates, implementation dates, repository history, and any
  prior public or semi-public disclosures.
- Review the NinaPro agreement and do not include restricted dataset samples or
  derived assets in an invention submission without permission.
- Ask counsel to search prior art and determine claim scope, filing strategy,
  ownership, inventorship, and disclosure timing.

The names on the project README are project contributors, not a determination
of inventorship. Fill in the inventor and ownership fields only after review.

---

## Draft A — Risk-weighted evidence accumulation with reversible actuation

### Administrative fields

- **Working title:** Risk-weighted sequential evidence control with reversible
  progressive actuator commands
- **Inventor(s):** To be determined by the team and counsel
- **Owner/assignee:** To be determined under course and institutional policies
- **First conception / reduction to practice:** Verify from project records
- **Known public disclosures:** Team to identify all presentations, repositories,
  reports, and demonstrations before counsel review

### Technical problem

A single-window classifier threshold gives a binary decision and cannot
represent (a) different consequences for confusing different gestures and
(b) a period of observable, reversible movement while the classifier is still
accumulating evidence. A late binary command can be abrupt; an early incorrect
command can be difficult to undo.

### Proposed technical approach

For each gesture class, accumulate evidence over successive posterior
observations. The implemented accumulator compares each class's probability
with its strongest rival, applies leakage and a non-negative floor, and uses a
class-specific boundary of the form:

`boundary_k = base_threshold + risk_weight * commit_cost_k`

The leading class's evidence is converted to a bounded commitment value. A
virtual actuator begins moving at a configured fraction of the boundary; the
motion remains retractable while evidence changes and latches only at full
commitment. An urgency schedule reduces the boundary over time. If no class
commits before a maximum duration, the system selects the configured safe
fallback rather than forcing a gesture guess.

### Prototype status and evidence

The mechanism is implemented in [packages/core/src/evidence.ts](../packages/core/src/evidence.ts)
and wired to the browser worker and virtual hand. The current default settings
include a 0.2 motion-onset fraction, a base threshold of 24, a risk weight of
12, leakage of 0.05, and a 75-hop timeout at a 20 ms hop. Values are
configurations of this research prototype, not validated device settings.

The TTUM analysis replays simulator-generated posteriors through the same
accumulator and measures when the virtual hand begins to move. The mechanism
has not been validated with recorded-data posteriors, a physical actuator, or
human prosthesis users. No clinical safety or reaction-time claim is made.

### Possible embodiments for counsel to consider

- Other posterior or score sources, provided class evidence is accumulated
  against competing hypotheses.
- Risk costs supplied by a user, clinician, task context, or a stored functional
  risk matrix.
- Different leakage, urgency, motion-onset, and safe-fallback policies.
- An actuator command constrained to be continuous and retractable until a
  class-specific commit condition is met.

These are discussion prompts, not claims that every variant is implemented.

### Potential technical distinction to investigate

The candidate combination is per-class risk-derived evidence boundaries plus
continuous reversible actuation before commitment, followed by bounded
urgency and a safe fallback. The team should compare this complete combination
against sequential probability ratio methods, shared-control prosthesis
controllers, confidence-triggered abstention, and progressive myoelectric
control before asserting any distinction.

---

## Draft B — Reachability-informed gesture vocabulary selection

### Administrative fields

- **Working title:** User-reachable activation mapping for effort-aware
  myoelectric gesture vocabulary selection
- **Inventor(s):** To be determined by the team and counsel
- **Owner/assignee:** To be determined under course and institutional policies
- **First conception / reduction to practice:** Verify from project records
- **Known public disclosures:** Team to identify all presentations, repositories,
  reports, and demonstrations before counsel review

### Technical problem

A fixed gesture catalogue may include activations that are difficult for a
particular user to reproduce or that are poorly separated in the user's
available EMG patterns. Selecting a gesture only by a population-level
confusion matrix does not account for that user's reachable activation
patterns, repeatability, effort, or fatigue.

### Proposed technical approach

Collect a rest anchor and a sequence of exploratory activation patterns (“babble”).
Represent channel-level activation using a low-dimensional synergy model,
estimate which regions are reachable and repeatable, and associate candidate
gestures with effort and separation from already selected gestures. Select a
vocabulary that trades off reachability, separation, and effort while
respecting a minimum spacing constraint. A later teach-back stage could train
the user to reproduce selected targets and use those observations to update
the map.

### Prototype status and limits

The research implementation in
[services/research/neurogrip/cartography.py](../services/research/neurogrip/cartography.py)
uses simulated babble, NMF synergy features, reachability/effort/separability
estimates, prototype synthesis experiments, and weighted catalogue selection
as a fallback. The synthesis gate has not demonstrated an advantage over
catalogue selection in the simulator. The browser display is a simulated
demonstration. Human babble, real-user teach-back, and clinical evaluation are
not implemented.

The current implementation does not establish that the generated vocabulary
is learnable, improves recognition, or is safe. Do not describe the proposed
teach-back workflow as completed.

### Potential technical distinction to investigate

The candidate combination is a user-specific reachable-set estimate coupled
to a selection objective that includes repeatability, separation, and effort,
with a catalogue fallback when synthesized prototypes fail a gate. Counsel
should assess prior work in myoelectric calibration, motor babbling, synergy
mapping, gesture selection, and user adaptation.

---

## Draft C — Error decomposition and remedy routing for myoelectric decoding

### Administrative fields

- **Working title:** Evidence-based routing of myoelectric decoding errors to
  distinct corrective actions
- **Inventor(s):** To be determined by the team and counsel
- **Owner/assignee:** To be determined under course and institutional policies
- **First conception / reduction to practice:** Verify from project records
- **Known public disclosures:** Team to identify all presentations, repositories,
  reports, and demonstrations before counsel review

### Technical problem

A misclassification alone does not distinguish an ambiguous pair of muscle
patterns, a user attempt outside their own repeat distribution, a model
capacity limit, or signal drift. Retuning the decoder for every error can
address the wrong cause and can reduce performance elsewhere.

### Proposed technical approach

For an observed error, derive separate evidence from: overlap between repeated
class clusters; deviation from the intended class's own repeat distribution;
whether a separately trained reference classifier corrects the deployed
classifier's error; and temporal/spectral drift indicators. Normalize the
evidence, require minimum evidence and separation before routing, and otherwise
report an inconclusive case. Map supported causes to distinct follow-up
actions, such as vocabulary review, repeat practice, model comparison, or
recalibration.

### Prototype status and limits

The current router in
[services/research/neurogrip/attribution.py](../services/research/neurogrip/attribution.py)
uses synthetic injected cases for the browser demonstration. Its channel ring
shows electrode positions, not individual muscles. A separate offline
analysis computes exact standardized LDA score-margin contributions and
compares held-out errors with LinearSVC, RBF SVM, and random-forest models; it
is not integrated into the router. These results use only S1, S10, and S11,
which have already informed development. The coaching has not been validated
for usefulness, comprehensibility, or safety with people.

### Potential technical distinction to investigate

The candidate combination is multi-source error evidence with an abstention
state and remedy-specific routing, rather than a single confidence score or
generic retraining action. Counsel should review prior work on error
diagnostics, model disagreement, concept drift, myoelectric training, and
clinical decision support.

---

## Supporting project records

- Current architecture and implementation status: [README.md](../README.md),
  [context.md](../context.md), and [walkthrough.md](walkthrough.md).
- Preliminary design and prior-art notes:
  [NeuroGrip design specification](superpowers/specs/2026-09-04-neurogrip-design.md).
- Data permission status:
  [data and release review](data_and_release_review.md).

The design specification's prior-art discussion is not a legal search. No
patentability or freedom-to-operate conclusion has been made.
