import { CLASS_INFO } from "@/lib/constants";
import type { ClassName, LabelPrediction } from "@/lib/types";

interface Props {
  predictions: LabelPrediction[];
  referenceLabels?: ClassName[];
}

export function PredictionPanel({ predictions, referenceLabels }: Props) {
  const positives = predictions.filter((p) => p.positive).map((p) => p.label);
  return (
    <div className="space-y-4">
      <ul className="space-y-3">
        {predictions.map((p) => {
          const pct = p.probability * 100;
          const isRef = referenceLabels?.includes(p.label);
          return (
            <li
              key={p.label}
              className={`rounded-xl border p-3 transition-colors ${
                p.positive ? "border-accent/40 bg-accent/5" : "border-line bg-surface-2/40"
              }`}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-sm font-semibold">{p.label}</span>
                  <span className="text-sm text-ink-3">{CLASS_INFO[p.label].name}</span>
                  {isRef ? (
                    <span className="rounded border border-line-strong px-1.5 text-[11px] text-ink-2">
                      PTB-XL reference label
                    </span>
                  ) : null}
                </div>
                <div className="flex items-baseline gap-3">
                  <span className="tabular text-lg font-semibold">{pct.toFixed(1)}%</span>
                  <span
                    className={`inline-flex items-center gap-1 text-xs font-medium ${
                      p.positive ? "text-accent" : "text-ink-3"
                    }`}
                  >
                    <span aria-hidden="true">{p.positive ? "✓" : "—"}</span>
                    {p.positive ? "Above threshold" : "Below threshold"}
                  </span>
                </div>
              </div>
              <div className="relative mt-2 h-2.5 rounded-full bg-[var(--grid)]" aria-hidden="true">
                <div
                  className="h-full rounded-full transition-[width] duration-700"
                  style={{
                    width: `${Math.max(pct, 0.5)}%`,
                    background: "var(--data-teal)",
                    opacity: p.positive ? 1 : 0.55,
                  }}
                />
                <div
                  className="absolute -bottom-1 -top-1 w-0.5 rounded bg-ink-2"
                  style={{ left: `calc(${p.threshold * 100}% - 1px)` }}
                  title={`Threshold ${p.threshold}`}
                />
              </div>
              <p className="tabular mt-1.5 text-[11px] text-ink-3">
                Model probability {p.probability.toFixed(4)} · frozen threshold {p.threshold.toFixed(2)}
              </p>
            </li>
          );
        })}
      </ul>
      <div className="rounded-xl border border-line bg-bg/60 p-3 text-sm">
        <p className="text-ink-2">
          Thresholded model output:{" "}
          <span className="font-medium text-ink">
            {positives.length ? positives.join(" · ") : "no class above its threshold"}
          </span>
        </p>
        {referenceLabels ? (
          <p className="mt-1 text-ink-3">
            PTB-XL reference annotation for this record: {referenceLabels.join(" · ")}
          </p>
        ) : null}
      </div>
      <p className="text-xs text-ink-3">
        Labels are independent (multi-label): probabilities need not sum to 100%. The vertical
        tick marks each class&apos;s threshold, chosen on the validation fold and frozen before
        the held-out test evaluation. These are model outputs, not diagnoses.
      </p>
    </div>
  );
}
