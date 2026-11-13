/**
 * Words explained.
 *
 * One page per stage of the story, each term with a plain definition and one
 * sentence to say out loud. Written for the team as much as for visitors:
 * presenting the project should never require explaining a word the slide
 * does not.
 */

import { useState } from 'react';
import { Screen, type ScreenPage } from '../components/Screen.js';
import { GLOSSARY } from '../glossary.js';

export function Glossary() {
  const [page, setPage] = useState(0);

  const pages: ScreenPage[] = GLOSSARY.map((group) => ({
    id: group.story,
    label: group.title,
    summary: group.intro,
    content: (
      <ul className="glossary" data-story={group.story}>
        {group.terms.map((term) => (
          <li key={term.id} className="glossary-card" data-story={group.story}>
            <h3>
              {term.term}
              {term.aka ? <span className="glossary-aka">{term.aka}</span> : null}
            </h3>
            <p>{term.plain}</p>
            <p className="glossary-say">
              <span className="glossary-say-label">In one line</span>
              {term.sayIt}
            </p>
          </li>
        ))}
      </ul>
    ),
  }));

  return (
    <Screen
      title="Words explained"
      className="screen-glossary"
      pages={pages}
      page={page}
      onPage={setPage}
    />
  );
}
