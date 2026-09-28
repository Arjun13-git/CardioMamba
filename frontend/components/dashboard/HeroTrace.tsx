/** Decorative-but-real ECG trace: lead II of a PTB-XL demo record, drawn once on load. */
export function HeroTrace({ values, caption }: { values: number[]; caption: string }) {
  const w = 800;
  const h = 160;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 12;
  const d = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = pad + (1 - (v - min) / (max - min || 1)) * (h - 2 * pad);
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join("");
  return (
    <figure className="relative overflow-hidden rounded-2xl border border-line bg-surface/70">
      <div className="ecg-grid absolute inset-0" aria-hidden="true" />
      <svg viewBox={`0 0 ${w} ${h}`} className="relative h-40 w-full" role="img" aria-label={caption}>
        <path
          d={d}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
          pathLength={2000}
          style={{ strokeDasharray: 2000, animation: "trace 2.6s ease-out both" }}
        />
      </svg>
      <figcaption className="relative border-t border-line px-4 py-2 text-xs text-ink-3">
        {caption}
      </figcaption>
    </figure>
  );
}
