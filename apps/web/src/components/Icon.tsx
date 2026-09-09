/**
 * Renders an icon from the design system's construction grid.
 *
 * The icon is decorative by default and hidden from assistive technology. Pass
 * `label` only when the icon is the sole carrier of meaning; if there is text
 * beside it, the text is the label and the icon must stay hidden or it will be
 * announced twice.
 */

import { ICONS, ICON_CANVAS, ICON_STROKE, type IconName } from '@neurogrip/design';

export interface IconProps {
  readonly name: IconName;
  readonly size?: number;
  readonly label?: string;
}

export function Icon({ name, size = 20, label }: IconProps) {
  const accessibility = label
    ? ({ role: 'img', 'aria-label': label } as const)
    : ({ 'aria-hidden': true, focusable: false } as const);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${ICON_CANVAS} ${ICON_CANVAS}`}
      strokeWidth={ICON_STROKE}
      strokeLinecap="butt"
      strokeLinejoin="miter"
      className="icon"
      {...accessibility}
    >
      {ICONS[name].elements.map((element, index) => {
        const fill = element.filled ? 'currentColor' : 'none';
        const stroke = element.filled ? 'none' : 'currentColor';
        return element.kind === 'circle' ? (
          <circle key={index} cx={element.cx} cy={element.cy} r={element.r} fill={fill} stroke={stroke} />
        ) : (
          <path key={index} d={element.d} fill={fill} stroke={stroke} />
        );
      })}
    </svg>
  );
}
