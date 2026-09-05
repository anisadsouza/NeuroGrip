# Using NeuroGrip

**Research prototype. Not a medical device. The signal it plays back is
simulated, not a recording of a person.**

This walks through what the application does, screen by screen, and what each
number on it means. It assumes nothing about the codebase.

---

## Getting it running

You need Python 3.12+ and Node 24+.

```bash
pip install -e "./services/research[dev,ml]"
npm install
npm run dev:web
```

Open the address it prints. Nothing else is required — no dataset download, no
NinaPro registration, no account.

If the app reports **"The decoder could not start"**, the model or replay bundle
is missing. Rebuild both:

```bash
npm run build:assets
```

That takes about a minute: it regenerates the replay signal, retrains the
decoder on the simulator, re-exports it to ONNX, verifies the exported model
still agrees with scikit-learn, and copies both into the app.

---

## What the application is doing

Every 20 milliseconds it takes the most recent 200 milliseconds of
eight-channel muscle signal, checks the electrodes are still attached, reduces
that window to 136 numbers, runs them through a classifier, and updates its
belief about which of ten gestures you are attempting.

It does **not** decide immediately. It accumulates evidence across successive
windows and moves the hand progressively as its confidence builds. That is the
part worth watching.

All of this happens inside your browser. Nothing is uploaded.

---

## The Live screen

### Intended gesture

Pick what you are attempting. Ten options, in plain language rather than the
model's internal names.

**The decoder is not told your choice.** It only sees the muscle signal that
choice produces. That is what makes the "Decoded" readout meaningful: if it
matches your selection, the decoder worked it out from the signal alone.

Changing your selection clears the decoder's accumulated evidence, so each
attempt starts fresh.

### Start decoding / Pause

Starts and stops the signal stream. **Release** clears a held gesture without
stopping the stream — the equivalent of relaxing your hand so the prosthesis
opens.

### Signal

Eight lanes of muscle activity, one per electrode, drawn on a graticule.

The label at the top right — for example `600 µV · 200 ms` — is the scale. A
trace without a stated scale is a decoration, so it is always shown. The grid
lines are what you count by.

All eight lanes share one vertical scale, so their amplitudes are directly
comparable. A quiet channel looks quiet.

If you see a flat line where the others are active, that electrode has lost
contact.

### Activation

The forearm in cross-section, viewed as if looking down your own arm, with the
eight electrodes around it.

The ring sits around the **forearm just below the elbow**, level with the elbow
joint itself. That is where the muscles that move your fingers actually are:
they sit near the elbow and pull on long tendons running down to the hand. An
armband further down the forearm would be sitting over tendon, and one on the
upper arm would be reading biceps and triceps, which say nothing about what the
fingers are doing.

Each electrode fills toward deep red in proportion to how hard the muscle
beneath it is working. This is the one place in the interface where colour is a
measurement rather than a label.

Try `Close fist` and then `Open hand` and watch the pattern move from one side
of the ring to the other. Fist is finger flexion; open hand is finger extension;
those muscles sit on opposite sides of the forearm. If the ring did not move,
something is wrong.

An electrode outlined in red has been rejected by the signal-quality check.

### Commitment

The most important panel, and the thing that makes this system different from a
conventional gesture classifier.

A conventional decoder waits until it is confident, then jumps. This one starts
moving early and can change its mind.

Reading the bar, left to right:

| Mark | Meaning |
| --- | --- |
| Filled region | How much evidence has accumulated for the leading gesture |
| First thin line | Where the hand starts to move. Before this, nothing has moved |
| Short vertical marks | Each gesture's commitment point |
| Tall dark mark | The leading gesture's commitment point |
| Low short marks | Competing gestures, on the same scale |

The state label at the top right reads:

- **listening** — evidence is accumulating, the hand has not moved
- **moving, reversible** — the hand is moving and can still be taken back
- **held** — committed; the gesture stays until you press Release
- **timed out** — no clear intent was found, so it fell back to rest

**The commitment points are not all in the same place, and that is the point.**
Closing a fist sits further right than opening a hand. Both require the same
kind of evidence, but a fist closing on something fragile is much harder to undo
than a hand opening, so it has to be more certain before it commits.

You can see this directly. Select `Close fist`, start, and watch how far the bar
travels before it locks. Then do the same with `Open hand`. The open hand
commits sooner from the same quality of evidence.

### Hand

What the prosthesis is actually doing. The commitment bar is the evidence; this
is the consequence. They are driven by the same number.

The hand does not move at all until the bar passes its first mark. From there it
travels continuously, reaching the full gesture exactly when the bar reaches
that gesture's post. So the two panels always agree — if the bar is at half, the
hand is halfway there.

**While the hand is still moving, a faint dashed outline sits behind it.** That
is the resting hand: where it returns to if the evidence turns. It disappears
the moment the gesture locks, because at that moment the hand is no longer going
back on its own. That appearing-and-vanishing outline is the clearest single
signal of whether you can still change your mind.

When the gesture locks, the whole hand turns green.

The hand is drawn from the palm side, turned slightly so you can see fingers
curl toward you rather than straight at you. Wrist bending is shown at a
somewhat smaller angle than a real wrist reaches; at the true angle the hand
turns edge-on to you and becomes impossible to read.

### Decoder

| Row | Meaning |
| --- | --- |
| **Decoded** | The gesture it committed to. Green when it matches your selection, amber when it does not |
| **Latency, this window** | Feature extraction plus inference for the most recent window |
| **Latency, P95** | The 95th percentile over the session. The safety-relevant number |
| **WASM threads** | Whether multi-threaded inference is available |

The budget is 10 ms at P95. If **WASM threads** reads `single`, the page is not
cross-origin isolated and inference is roughly twice as slow; the number shown
is still honest, just measured on a slower path.

---

## Things worth trying

**Watch a decision form.** Pick `Hold a ball`, press start, and watch the
commitment bar rather than the readout. You are seeing evidence accumulate in
real time.

**See the risk asymmetry.** Compare how far `Close fist` travels before locking
against `Open hand`. Same decoder, different cost of being wrong.

**Watch the hand and the bar together.** Pick any gesture and watch both at
once. The hand stays completely still while the bar fills to its first mark,
then starts closing. Nothing on a conventional prosthesis does this: it would
sit still through the whole deliberation and then jump.

**Change your mind mid-gesture.** Start one gesture, and while the bar is still
filling — before it reads *held* — switch to another. The accumulated evidence
is discarded and the new one starts from zero. On a real prosthesis that is the
difference between a recoverable mistake and a dropped cup.

**Watch the anatomy.** Alternate `Bend wrist in` and `Bend wrist back` and watch
the activation ring flip sides.

**Confuse it deliberately.** `Pinch` and `Two-finger grip` use similar muscles.
The decoder confuses them more often than it confuses either with `Open hand`.
That is a real property of surface EMG with eight channels, not a bug — and it
is exactly the kind of error the error-attribution router (Phase 3) is designed
to diagnose.

---

## What the numbers mean, and what they do not

The decoder scores **95.6%** under leave-one-subject-out evaluation: trained on
seven simulated subjects, tested on an eighth it has never seen. The full
breakdown, including per-gesture recall and the comparison against the two
models that lost, is in [`model_card.md`](model_card.md).

**That figure is optimistic and should not be quoted as a clinical result.**

The signal is simulated. The simulator produces physiologically shaped muscle
activity — real motor-unit recruitment, real fatigue behaviour, real
volume-conduction crosstalk between channels — but it has no motion artefact, no
sweat changing skin impedance, no electrode peeling off mid-session, and no
variation from taking the sleeve off and putting it back on tomorrow. Those are
the things that break myoelectric control in practice.

Published results on recorded human sEMG for a ten-gesture vocabulary sit well
below this. Expect a drop when real data arrives, and report both numbers.

There is also **no amputee stratum**. Every simulated subject is able-bodied by
construction. The project treats honest amputee-specific reporting as a
first-class requirement, and it cannot be satisfied until recorded amputee data
is available.

---

## Accessibility

- Every control is reachable by keyboard, with a visible focus ring.
- The commitment bar exposes its state to screen readers as a meter, and
  announces committed and abandoned gestures — not every 20 ms update, which
  would be unusable.
- The activation ring has an equivalent data table for assistive technology.
- The interface text is set in Atkinson Hyperlegible, designed for maximum
  character distinction at low vision.
- Light and dark themes are separate palettes, both tested to WCAG contrast
  minimums by an automated check that fails the build.
- Animation respects `prefers-reduced-motion`.

If something here does not work with your assistive technology, that is a bug.

---

## Not built yet

The Live screen is one screen of a planned application. Still to come:

- **Cartography** — mapping your reachable activation space and synthesising
  gestures suited to it
- **Coaching** — diagnosing whether an error was physiological, behavioural, a
  model limitation, or drift, and prescribing the matching fix
- **Clinician mode** — subject-independent evaluation, amputee-stratum
  reporting, drift monitoring, electrode-shift robustness

See `CLAUDE.md` for the phase plan and
`docs/superpowers/specs/2026-09-04-neurogrip-design.md` for the full design.
