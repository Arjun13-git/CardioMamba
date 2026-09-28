"use client";

import { useMemo, useRef, useState } from "react";

const WINDOWS = [10, 5, 2.5] as const;
const AMP_MV = 1.5; // each row spans -1.5 … +1.5 mV (same scale for every lead)
const VB_W = 1000;
const VB_H = 100;

interface Props {
  data: number[][]; // [12][T] in mV
  leads: string[];
  fs: number;
}

function tracePath(values: number[], i0: number, i1: number): string {
  const n = i1 - i0;
  let d = "";
  for (let i = i0; i <= i1 && i < values.length; i++) {
    const x = ((i - i0) / n) * VB_W;
    const v = Math.max(-AMP_MV * 1.2, Math.min(AMP_MV * 1.2, values[i]));
    const y = ((AMP_MV - v) / (2 * AMP_MV)) * VB_H;
    d += `${i === i0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }
  return d;
}

export function EcgViewer({ data, leads, fs }: Props) {
  const total = data[0].length / fs; // seconds
  const [win, setWin] = useState<number>(10);
  const [start, setStart] = useState(0);
  const [hover, setHover] = useState<number | null>(null); // sample index
  const plotRef = useRef<HTMLDivElement>(null);

  const s0 = Math.min(start, total - win);
  const i0 = Math.round(s0 * fs);
  const i1 = Math.min(data[0].length - 1, Math.round((s0 + win) * fs));

  const paths = useMemo(() => data.map((lead) => tracePath(lead, i0, i1)), [data, i0, i1]);
  const vGrid = useMemo(() => {
    const major: number[] = [];
    const minor: number[] = [];
    for (let t = Math.ceil(s0 / 0.04) * 0.04; t <= s0 + win + 1e-9; t += 0.04) {
      const x = ((t - s0) / win) * VB_W;
      const isMajor = Math.abs(t / 0.2 - Math.round(t / 0.2)) < 1e-6;
      (isMajor ? major : minor).push(x);
    }
    return { major, minor: win <= 5 ? minor : [] };
  }, [s0, win]);
  const ticks = useMemo(() => {
    const step = win >= 10 ? 1 : win >= 5 ? 0.5 : 0.25;
    const out: number[] = [];
    for (let t = Math.ceil(s0 / step) * step; t <= s0 + win + 1e-9; t += step) out.push(t);
    return out;
  }, [s0, win]);

  function onMove(e: React.PointerEvent) {
    const rect = plotRef.current?.getBoundingClientRect();
    if (!rect) return;
    const f = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    setHover(i0 + Math.round(f * (i1 - i0)));
  }

  const hoverX = hover === null ? null : ((hover - i0) / (i1 - i0)) * 100;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <div role="group" aria-label="Time window" className="flex rounded-lg border border-line p-0.5">
          {WINDOWS.map((w) => (
            <button
              key={w}
              type="button"
              onClick={() => {
                setWin(w);
                setStart((s) => Math.min(s, total - w));
              }}
              aria-pressed={win === w}
              className={`rounded-md px-2.5 py-1 transition-colors ${
                win === w ? "bg-surface-2 text-ink" : "text-ink-3 hover:text-ink"
              }`}
            >
              {w} s
            </button>
          ))}
        </div>
        {win < total ? (
          <label className="flex flex-1 items-center gap-2 text-ink-3">
            <span className="whitespace-nowrap">Pan</span>
            <input
              type="range"
              min={0}
              max={total - win}
              step={0.1}
              value={s0}
              onChange={(e) => setStart(Number(e.target.value))}
              className="w-full accent-[var(--accent)]"
              aria-label="Window start (seconds)"
            />
            <span className="tabular w-24 text-right">
              {s0.toFixed(1)}–{(s0 + win).toFixed(1)} s
            </span>
          </label>
        ) : (
          <span className="text-ink-3">Hover to inspect all leads at one instant</span>
        )}
        <span className="tabular ml-auto text-ink-3">
          {hover === null ? "" : `t = ${(hover / fs).toFixed(2)} s`}
        </span>
      </div>

      <div className="relative select-none rounded-xl border border-line bg-bg/70">
        <div
          ref={plotRef}
          className="absolute bottom-6 left-12 right-16 top-0 z-10 cursor-crosshair"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          aria-hidden="true"
        >
          {hoverX !== null ? (
            <div
              className="pointer-events-none absolute bottom-0 top-0 w-px bg-accent/70"
              style={{ left: `${hoverX}%` }}
            />
          ) : null}
        </div>
        <div className="divide-y divide-line/60">
          {leads.map((lead, li) => (
            <div key={lead} className="flex h-14 items-stretch">
              <div className="flex w-12 shrink-0 items-center justify-center font-mono text-xs text-ink-2">
                {lead}
              </div>
              <svg
                viewBox={`0 0 ${VB_W} ${VB_H}`}
                preserveAspectRatio="none"
                className="h-full min-w-0 flex-1"
                role="img"
                aria-label={`Lead ${lead}, ${s0.toFixed(1)} to ${(s0 + win).toFixed(1)} seconds`}
              >
                {vGrid.minor.map((x) => (
                  <line key={`m${x}`} x1={x} x2={x} y1={0} y2={VB_H} stroke="var(--grid)" strokeWidth={1} vectorEffect="non-scaling-stroke" opacity={0.5} />
                ))}
                {vGrid.major.map((x) => (
                  <line key={`M${x}`} x1={x} x2={x} y1={0} y2={VB_H} stroke="var(--grid)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                ))}
                {[-1, -0.5, 0.5, 1].map((mv) => (
                  <line key={mv} x1={0} x2={VB_W} y1={((AMP_MV - mv) / (2 * AMP_MV)) * VB_H} y2={((AMP_MV - mv) / (2 * AMP_MV)) * VB_H} stroke="var(--grid)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                ))}
                <line x1={0} x2={VB_W} y1={VB_H / 2} y2={VB_H / 2} stroke="var(--border-strong)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                <path d={paths[li]} fill="none" stroke="var(--data-teal)" strokeWidth={1.5} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
              </svg>
              <div className="tabular flex w-16 shrink-0 items-center justify-end pr-2 font-mono text-[11px] text-ink-2">
                {hover === null ? "" : `${data[li][hover].toFixed(2)}`}
              </div>
            </div>
          ))}
        </div>
        <div className="relative ml-12 mr-16 h-6 border-t border-line text-[11px] text-ink-3">
          {ticks.map((t) => (
            <span
              key={t}
              className="tabular absolute top-1 -translate-x-1/2"
              style={{ left: `${((t - s0) / win) * 100}%` }}
            >
              {Number.isInteger(t) ? t : t.toFixed(2).replace(/0$/, "")}s
            </span>
          ))}
        </div>
      </div>
      <p className="text-xs text-ink-3">
        Preprocessed model input (before normalization): 100 Hz, 10 s, 12 leads, 1000 samples per
        lead — shape [1000, 12]. Each row spans ±{AMP_MV} mV; grid 0.2 s × 0.5 mV (0.04 s minor
        lines when zoomed). Hover readout in mV.
      </p>
    </div>
  );
}
