"use client";

import { useEffect, useRef, useState } from "react";

export interface Series {
  name: string;
  color: string; // CSS colour (validated palette)
  values: number[];
}

interface Props {
  x: number[];
  series: Series[];
  yLabel: string;
  digits?: number;
  markers?: { x: number; label: string }[];
  height?: number;
}

const M = { top: 16, right: 16, bottom: 32, left: 52 };

function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min || 1;
  const raw = span / count;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => span / s <= count) ?? raw;
  // Domain rounds outwards so every data point lies inside [first tick, last tick].
  const out: number[] = [];
  const lo = Math.floor(min / step - 1e-9) * step;
  const hi = Math.ceil(max / step + 1e-9) * step;
  for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}

export function LineChart({ x, series, yLabel, digits = 4, markers = [], height = 240 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const all = series.flatMap((s) => s.values);
  const pad = (Math.max(...all) - Math.min(...all)) * 0.1 || 0.01;
  const ticks = niceTicks(Math.min(...all) - pad, Math.max(...all) + pad, 4);
  const y0 = ticks[0];
  const y1 = ticks[ticks.length - 1];
  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const sx = (v: number) => M.left + ((v - x[0]) / (x[x.length - 1] - x[0])) * iw;
  const sy = (v: number) => M.top + (1 - (v - y0) / (y1 - y0)) * ih;

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const f = (e.clientX - rect.left) / rect.width;
    setHover(Math.max(0, Math.min(x.length - 1, Math.round(f * (x.length - 1)))));
  }

  return (
    <div ref={ref} className="relative">
      {series.length > 1 ? (
        <ul className="mb-2 flex flex-wrap gap-4 text-xs text-ink-2">
          {series.map((s) => (
            <li key={s.name} className="flex items-center gap-1.5">
              <span className="h-0.5 w-4 rounded" style={{ background: s.color }} aria-hidden="true" />
              {s.name}
            </li>
          ))}
        </ul>
      ) : null}
      <svg width={width} height={height} role="img" aria-label={`${yLabel} by epoch`} className="block max-w-full">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={width - M.right} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={M.left - 8} y={sy(t)} dy="0.32em" textAnchor="end" className="tabular fill-[var(--text-muted)] text-[11px]">
              {t.toFixed(digits > 2 ? 3 : 2)}
            </text>
          </g>
        ))}
        {x.map((v) => (
          <text key={v} x={sx(v)} y={height - 10} textAnchor="middle" className="tabular fill-[var(--text-muted)] text-[11px]">
            {v}
          </text>
        ))}
        <text x={width - M.right} y={height - 10} textAnchor="end" className="fill-[var(--text-muted)] text-[11px]" dx="0" dy="-14">
          epoch
        </text>
        {markers.map((m) => {
          const nearRight = sx(m.x) > M.left + iw * 0.75;
          return (
            <g key={m.label}>
              <line x1={sx(m.x)} x2={sx(m.x)} y1={M.top} y2={M.top + ih} stroke="var(--border-strong)" />
              <text
                x={sx(m.x) + (nearRight ? -4 : 4)}
                y={M.top + 10}
                textAnchor={nearRight ? "end" : "start"}
                className="fill-[var(--text-secondary)] text-[11px]"
              >
                {m.label}
              </text>
            </g>
          );
        })}
        {series.map((s) => (
          <g key={s.name}>
            <polyline
              points={s.values.map((v, i) => `${sx(x[i])},${sy(v)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {hover !== null ? (
              <circle cx={sx(x[hover])} cy={sy(s.values[hover])} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
            ) : null}
          </g>
        ))}
        {hover !== null ? (
          <line x1={sx(x[hover])} x2={sx(x[hover])} y1={M.top} y2={M.top + ih} stroke="var(--text-muted)" strokeWidth={1} />
        ) : null}
        <rect
          x={M.left}
          y={M.top}
          width={iw}
          height={ih}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover !== null ? (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(sx(x[hover]) + 12, width - 170),
            top: M.top + (series.length > 1 ? 28 : 4),
          }}
        >
          <p className="mb-1 text-ink-3">Epoch {x[hover]}</p>
          {series.map((s) => (
            <p key={s.name} className="tabular flex items-center gap-2 text-ink">
              <span className="h-0.5 w-3 rounded" style={{ background: s.color }} aria-hidden="true" />
              {s.name}: {s.values[hover].toFixed(digits)}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
