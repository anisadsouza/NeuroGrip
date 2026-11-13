/**
 * Every technical word the interface uses, in plain language.
 *
 * Written so that anyone on the team can read a definition aloud and have
 * explained the idea: no formulas, no acronyms inside a definition, one or two
 * sentences each. The same text appears in the glossary screen and in the
 * inline definition behind each underlined term, so the two can never
 * disagree.
 *
 * Grouped by the four story colours, in the order a gesture travels: the body
 * makes a signal, the signal is measured, a decision is made, the result is
 * judged. Research ideas come last.
 */

export type Story = 'muscle' | 'signal' | 'evidence' | 'grip' | 'research';

export interface GlossaryTerm {
  readonly id: string;
  readonly term: string;
  /** The short form people will see on screen, if different. */
  readonly aka?: string;
  readonly plain: string;
  /** One sentence to say out loud when someone asks. */
  readonly sayIt: string;
}

export interface GlossaryGroup {
  readonly story: Story;
  readonly title: string;
  readonly intro: string;
  readonly terms: readonly GlossaryTerm[];
}

export const GLOSSARY: readonly GlossaryGroup[] = [
  {
    story: 'muscle',
    title: 'The body',
    intro: 'Where the signal comes from.',
    terms: [
      {
        id: 'semg',
        term: 'Surface EMG',
        aka: 'sEMG',
        plain:
          'The tiny electrical signal muscles give off when they work, picked up by sensors resting on the skin. No needles.',
        sayIt: 'We listen to the electricity your forearm muscles make when you try to move your hand.',
      },
      {
        id: 'electrode',
        term: 'Electrode',
        plain:
          'A small sensor on the skin. NeuroGrip uses eight of them in a band around the upper forearm, where the finger muscles are.',
        sayIt: 'Eight skin sensors in a band around the forearm.',
      },
      {
        id: 'flexor',
        term: 'Flexors and extensors',
        plain:
          'Flexors bend the fingers or wrist inward; extensors straighten the fingers or bend the wrist back. The band can tell these groups apart, not single fingers.',
        sayIt: 'Bending muscles and straightening muscles.',
      },
      {
        id: 'activation',
        term: 'Muscle activation',
        plain:
          'How hard a muscle is working right now. Shown in red throughout the app.',
        sayIt: 'Red always means muscle effort.',
      },
    ],
  },
  {
    story: 'signal',
    title: 'The signal',
    intro: 'How the raw electricity becomes something a computer can read.',
    terms: [
      {
        id: 'window',
        term: 'Window',
        plain:
          'A 0.2 second slice of the signal. A fresh slice is examined every 0.02 seconds, so decisions keep coming fifty times a second.',
        sayIt: 'We look at the last fifth of a second, fifty times a second.',
      },
      {
        id: 'feature',
        term: 'Features',
        plain:
          'Numbers that summarise a slice of signal: how strong it is, how jittery, which pitches it contains. Each slice becomes 136 of them.',
        sayIt: 'We turn each slice of signal into a short list of measurements.',
      },
      {
        id: 'simulator',
        term: 'Simulator',
        plain:
          'A computer model of forearm muscles that produces realistic signals, so the whole system can be built and tested without recording people.',
        sayIt: 'A virtual forearm that produces realistic muscle signals.',
      },
      {
        id: 'ninapro',
        term: 'NinaPro',
        aka: 'NinaPro DB2',
        plain:
          'A public research collection of real forearm muscle recordings. Three of its people are used here, separately from the simulator.',
        sayIt: 'Real recordings from a public research dataset, kept apart from simulated results.',
      },
    ],
  },
  {
    story: 'evidence',
    title: 'The decision',
    intro: 'How NeuroGrip decides when the hand should move.',
    terms: [
      {
        id: 'decoder',
        term: 'Decoder',
        aka: 'model',
        plain:
          'The program that looks at the measurements and says which gesture it thinks you are making, and how sure it is.',
        sayIt: 'The part that guesses your gesture from the signal.',
      },
      {
        id: 'evidence',
        term: 'Evidence',
        aka: 'commitment',
        plain:
          'Confidence built up over several slices instead of trusting one. It fills like a bar, and drains if the decoder changes its mind.',
        sayIt: 'Confidence that has to build up before the hand acts.',
      },
      {
        id: 'risk',
        term: 'Risk-weighted bar',
        plain:
          'Risky gestures need more evidence than safe ones. Closing a strong grip by mistake is worse than opening the hand, so its bar is set higher.',
        sayIt: 'Dangerous moves need more certainty than harmless ones.',
      },
      {
        id: 'reversible',
        term: 'Reversible motion',
        plain:
          'The hand starts moving early, while it can still go back if the evidence turns. It only locks in (latches) once the bar is full.',
        sayIt: 'It starts early, but can still change its mind until it is sure.',
      },
    ],
  },
  {
    story: 'grip',
    title: 'How we judge it',
    intro: 'The numbers used to say whether it works.',
    terms: [
      {
        id: 'unseen',
        term: 'Accuracy on unseen people',
        aka: 'leave-one-subject-out',
        plain:
          'The decoder is always tested on someone it never learned from. That is the honest test of whether it would work for a new user.',
        sayIt: 'We always test on a person the system has never seen.',
      },
      {
        id: 'calibration',
        term: 'Confidence error',
        aka: 'calibration, ECE',
        plain:
          'Whether the decoder’s confidence can be trusted. If it says it is 90% sure, it should be right about 90% of the time. Lower is better.',
        sayIt: 'How honest the decoder is about its own certainty.',
      },
      {
        id: 'latency',
        term: 'Decision time',
        aka: 'latency, P95',
        plain:
          'How long one decision takes on the computer. P95 means 95 out of 100 decisions were at least this fast. The limit is 10 ms.',
        sayIt: 'Each decision takes a couple of thousandths of a second.',
      },
      {
        id: 'ttum',
        term: 'Time to first movement',
        aka: 'TTUM',
        plain:
          'How long after the signal starts before the hand begins to move toward the gesture.',
        sayIt: 'How quickly the hand starts to respond.',
      },
      {
        id: 'cwer',
        term: 'Cost-weighted mistakes',
        aka: 'CWER',
        plain:
          'Mistakes counted by how harmful they are: a wrong strong grip counts as a whole mistake, a wrong hand-open only a tenth of one.',
        sayIt: 'Mistakes weighted by how much harm they could do.',
      },
    ],
  },
  {
    story: 'research',
    title: 'The research ideas',
    intro: 'The new methods this project is testing.',
    terms: [
      {
        id: 'cartography',
        term: 'Cartography',
        plain:
          'Mapping which muscle patterns a particular person can make comfortably, then choosing gestures that suit that person instead of a fixed list.',
        sayIt: 'Gestures chosen to fit your muscles, not the other way round.',
      },
      {
        id: 'teachback',
        term: 'Teach-back',
        plain:
          'Showing someone a new gesture as a target on the map and letting them practise until they can reach it reliably.',
        sayIt: 'Practising a new gesture by aiming at it on a map.',
      },
      {
        id: 'attribution',
        term: 'Attribution',
        plain:
          'Working out why a mistake happened (the body, the technique, the decoder or a changing signal) so the right fix is applied.',
        sayIt: 'Finding the cause of a mistake before fixing it.',
      },
      {
        id: 'drift',
        term: 'Drift',
        plain:
          'The signal slowly changing during use, from tired muscles or a sensor band that has shifted.',
        sayIt: 'The signal changing over time, so the decoder needs a top-up.',
      },
    ],
  },
];

const BY_ID = new Map(GLOSSARY.flatMap((group) => group.terms.map((term) => [term.id, { ...term, story: group.story }] as const)));

export function glossaryTerm(id: string): (GlossaryTerm & { story: Story }) | undefined {
  return BY_ID.get(id);
}
