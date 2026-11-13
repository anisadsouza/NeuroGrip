/**
 * The hero film, embedded with Remotion's Player.
 *
 * Autoplays muted and loops, with one visible pause control (WCAG 2.2.2: any
 * moving content that lasts more than five seconds must be pausable). Under
 * prefers-reduced-motion it never starts: it shows the latched frame as a
 * still, and the button offers to play it.
 */

import { useEffect, useRef, useState } from 'react';
import { Player, type PlayerRef } from '@remotion/player';
import { Icon } from '../components/Icon.js';
import {
  HERO_FPS,
  HERO_FRAMES,
  HERO_HEIGHT,
  HERO_POSTER_FRAME,
  HERO_WIDTH,
  HeroFilm,
} from './HeroFilm.js';

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function HeroPlayer() {
  const player = useRef<PlayerRef>(null);
  const [reduced] = useState(prefersReducedMotion);
  const [playing, setPlaying] = useState(!reduced);

  useEffect(() => {
    const current = player.current;
    if (!current) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    current.addEventListener('play', onPlay);
    current.addEventListener('pause', onPause);
    return () => {
      current.removeEventListener('play', onPlay);
      current.removeEventListener('pause', onPause);
    };
  }, []);

  return (
    <figure className="hero-player">
      <div className="hero-frame">
        <Player
          ref={player}
          component={HeroFilm}
          durationInFrames={HERO_FRAMES}
          fps={HERO_FPS}
          compositionWidth={HERO_WIDTH}
          compositionHeight={HERO_HEIGHT}
          initialFrame={reduced ? HERO_POSTER_FRAME : 0}
          autoPlay={!reduced}
          loop
          controls={false}
          clickToPlay={false}
          doubleClickToFullscreen={false}
          spaceKeyToPlayOrPause={false}
          acknowledgeRemotionLicense
          style={{ width: '100%', height: '100%' }}
        />
      </div>
      <figcaption className="hero-caption">
        <button
          type="button"
          className="icon-button icon-button-small"
          onClick={() => player.current?.toggle()}
          aria-label={playing ? 'Pause the film' : 'Play the film'}
          title={playing ? 'Pause' : 'Play'}
        >
          <Icon name={playing ? 'pause' : 'play'} size={16} />
        </button>
        <span>
          One gesture, end to end · simulated
        </span>
      </figcaption>
    </figure>
  );
}
