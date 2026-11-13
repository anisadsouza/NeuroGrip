# NeuroGrip UI: clinical redesign

Date: 2026-11-08. Owner: Harsh Bavaskar. Supersedes the "image-led, colourful"
visual layer of the same morning.

## Design read

Research instrument for project examiners (BTECH CAP 501), presented on a
projector and then explored on a laptop, in a clinical-precision language
(Apple Health / medical-device calm), built on native CSS custom properties
over the existing token file. Dials: VARIANCE 4, MOTION 3, DENSITY 5.

The one thing an examiner must leave with: **NeuroGrip is novel** - four
mechanisms beyond a standard gesture classifier.

## Why the current UI fails (audit)

| Problem | Where |
| --- | --- |
| Reads as AI-generic: rounded bento tiles, pale ground, black pills, generated studio photos | Overview, every screen |
| Three stacked stylesheets (`app.css`, `shell.css`, `theme.css`) overriding each other with `!important` | `apps/web/src` |
| No path: five equal nav pills, nothing says where to start or what comes next | Top bar |
| Results opens on an empty "real recording" tab | Results |
| Live: hand is small and secondary; a large empty graticule dominates before Start | Live |
| Four story hues compete with state colours; amber/green/carmine everywhere | All |
| Monospace numbers dropped into prose rows, centred table cells | Review |

## Decisions

| Area | Decision |
| --- | --- |
| Colour | Greys + one medical blue accent (`#0B5FD9` light). State colours (commit, caution, stop) keep their meaning. Categorical data colours live only inside charts and the hand. |
| Type | IBM Plex Sans (UI, headlines) + IBM Plex Mono (numerals), self-hosted. |
| Theme | Light default; dark toggle, its own palette. |
| Shape | Radius 4 / 8 / 12. Panels are hairline-bordered surfaces, not floating tiles. No shadows. |
| Imagery | Generated photos removed. The only visuals are the Remotion hero and real figures. |
| Structure | Guided path of six numbered steps plus free navigation: 1 Idea, 2 Live, 3 Accuracy, 4 Personal gestures, 5 Why errors happen, 6 Speed, safety & limits. Glossary off-path. Each step ends with a Next button. |
| Scroll | One viewport, no page scroll on desktop (unchanged; `layout.spec.ts`). |
| Live | The hand dominates; signal and commitment bar support it. |
| Hand | Technical line drawing: outlines, joint markers, ghosted rest pose. |
| Words | Correct technical terms with a plain `<Term>` hint. |
| Headline | "Reads the forearm. Moves only when sure." |
| Logo | Redesign: eight electrodes around a forearm cross-section. |
| Motion | Functional only. Remotion hero autoplays muted in a loop with a pause control; a static frame under reduced motion. |
| Notice | Slim footer on every screen, never dismissible, with the team names. |
| Icons | Custom 24-grid set and its tests kept. |

## Remotion hero

A ~12 s loop at 30 fps composed in `apps/web/src/hero/`, embedded with
`@remotion/player`:

1. Forearm cross-section, eight electrodes (0-2 s).
2. Muscle groups activate; electrodes nearest light up (2-4 s).
3. Eight traces draw (4-7 s).
4. Commitment bar fills; motion-onset mark crossed (7-9.5 s).
5. Hand closes and latches; label "Latched: fist" (9.5-12 s).

Drawn from the same `posture.ts`/`handPaths.ts` the app uses, so the hand in
the film is the app's hand, not an illustration. Simulated, and says so.

## Gates changed

- `contrast.test.ts`: the "exactly the story hue families" assertion becomes
  "chrome hue families are accent + state; data colours are declared under
  `--ng-data-*` only". All WCAG pair checks kept, retargeted to the new tokens.
- `layout.spec.ts`: unchanged rule; updated selectors.
- CLAUDE.md design-system section rewritten.
