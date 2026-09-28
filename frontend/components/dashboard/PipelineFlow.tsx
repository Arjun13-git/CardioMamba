const STEPS = [
  { title: "12-lead ECG", detail: "10 s · PTB-XL WFDB / CSV", shape: "5000 × 12" },
  { title: "Resample", detail: "500 → 100 Hz · resample_poly", shape: "1000 × 12" },
  { title: "Normalize", detail: "per-lead, train-fold stats", shape: "1000 × 12" },
  { title: "Conv1D patch stem", detail: "12 → 128 · k = 4, s = 4", shape: "250 × 128" },
  { title: "Bi-Mamba × 4", detail: "selective SSM, both directions", shape: "250 × 128" },
  { title: "RMSNorm + mean pool", detail: "sequence → vector", shape: "128" },
  { title: "Linear classifier", detail: "128 → 5 logits", shape: "5" },
  { title: "Sigmoid + thresholds", detail: "5 independent labels", shape: "multi-label" },
];

export function PipelineFlow() {
  return (
    <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {STEPS.map((s, i) => (
        <li
          key={s.title}
          className="group relative rounded-xl border border-line bg-surface-2/50 p-4 transition-colors hover:border-accent/40"
        >
          <div className="flex items-center justify-between">
            <span className="tabular text-xs font-medium text-ink-3">
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="tabular rounded-md border border-line bg-bg/60 px-1.5 py-0.5 font-mono text-[11px] text-accent">
              {s.shape}
            </span>
          </div>
          <p className="mt-3 font-medium">{s.title}</p>
          <p className="mt-0.5 text-sm text-ink-3">{s.detail}</p>
          {(i + 1) % 4 !== 0 ? (
            <span
              aria-hidden="true"
              className="absolute -right-2.5 top-1/2 z-10 hidden h-5 w-5 -translate-y-1/2 items-center justify-center rounded-full border border-line bg-bg text-[10px] text-ink-3 lg:flex"
            >
              →
            </span>
          ) : null}
        </li>
      ))}
    </ol>
  );
}
