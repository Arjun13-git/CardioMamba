import type { ReactNode } from "react";

function Node({
  title,
  detail,
  shape,
  tone = "default",
}: {
  title: ReactNode;
  detail?: ReactNode;
  shape?: string;
  tone?: "default" | "block" | "io";
}) {
  const tones = {
    default: "border-line bg-surface-2/60",
    block: "border-accent/40 bg-accent/5",
    io: "border-line-strong bg-bg/70",
  };
  return (
    <div className={`relative w-full rounded-xl border px-4 py-3 ${tones[tone]}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-medium">{title}</p>
        {shape ? (
          <span className="tabular rounded-md border border-line bg-bg/60 px-1.5 py-0.5 font-mono text-[11px] text-accent">
            {shape}
          </span>
        ) : null}
      </div>
      {detail ? <p className="mt-0.5 text-sm text-ink-3">{detail}</p> : null}
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-1" aria-hidden="true">
      <svg viewBox="0 0 12 18" className="h-4 w-3 text-ink-3">
        <path d="M6 0v15m-4-4 4 5 4-5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    </div>
  );
}

export function ArchitectureDiagram({ nLayers, dModel, numClasses }: { nLayers: number; dModel: number; numClasses: number }) {
  return (
    <div className="mx-auto max-w-md" role="img" aria-label="CardioMamba tensor flow from a 1000 by 12 ECG to 5 class logits">
      <Node tone="io" title="Input ECG" detail="100 Hz · 10 s · 12 leads, normalized" shape="1000 × 12" />
      <Arrow />
      <Node title="Conv1D patch stem" detail={`12 → ${dModel} channels · kernel 4 · stride 4 · no padding`} shape={`250 × ${dModel}`} />
      <Arrow />
      {Array.from({ length: nLayers }, (_, i) => (
        <div key={i}>
          <Node tone="block" title={`Bidirectional Mamba block ${i + 1}`} detail="RMSNorm → forward S6 + reversed S6 → sum → dropout → residual" shape={`250 × ${dModel}`} />
          <Arrow />
        </div>
      ))}
      <Node title="RMSNorm" shape={`250 × ${dModel}`} />
      <Arrow />
      <Node title="Mean pool over tokens" shape={String(dModel)} />
      <Arrow />
      <Node title="Dropout + linear head" detail={`${dModel} → ${numClasses} logits`} shape={String(numClasses)} />
      <Arrow />
      <Node tone="io" title="NORM · MI · STTC · CD · HYP" detail="sigmoid per class → frozen per-class thresholds" shape="multi-label" />
    </div>
  );
}

export function BiMambaBlockDiagram() {
  return (
    <div className="space-y-2 text-sm" role="img" aria-label="Bidirectional Mamba block: normalized input goes through a forward mixer and a time-reversed mixer; outputs are summed and added to the residual">
      <div className="rounded-lg border border-line bg-bg/60 px-3 py-2 text-center font-mono text-xs">x [250 × 128]</div>
      <Arrow />
      <div className="rounded-lg border border-line bg-surface-2/60 px-3 py-2 text-center">RMSNorm</div>
      <Arrow />
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-lg border border-accent/40 bg-accent/5 p-3">
          <p className="font-medium">Forward S6 mixer</p>
          <p className="mt-1 text-xs text-ink-3">reads tokens left → right (causal)</p>
        </div>
        <div className="rounded-lg border border-accent/40 bg-accent/5 p-3">
          <p className="font-medium">Reverse S6 mixer</p>
          <p className="mt-1 text-xs text-ink-3">flip → mixer → flip back (right → left), separate weights</p>
        </div>
      </div>
      <Arrow />
      <div className="rounded-lg border border-line bg-surface-2/60 px-3 py-2 text-center">Sum → dropout (0.1) → + x (residual)</div>
    </div>
  );
}

export function MixerSteps() {
  const steps: [string, string][] = [
    ["in_proj", "128 → 2 × 256, split into x and gate z"],
    ["Causal depthwise Conv1D", "kernel 4 over time, then SiLU"],
    ["x_proj → Δ, B, C", "input-dependent step size and projections (the “selective” part)"],
    ["Discretize", "Ā = exp(Δ·A), B̄x = Δ·B·x, with A = −exp(A_log), 16 states per channel"],
    ["Selective scan (FP32)", "hₜ = Āₜ ⊙ hₜ₋₁ + B̄ₜxₜ ;  yₜ = hₜ · Cₜ + D ⊙ xₜ"],
    ["Gate + out_proj", "y ⊙ SiLU(z), 256 → 128"],
  ];
  return (
    <ol className="space-y-2">
      {steps.map(([t, d], i) => (
        <li key={t} className="flex gap-3 rounded-lg border border-line bg-surface-2/40 p-3">
          <span className="tabular mt-0.5 text-xs text-ink-3">{i + 1}</span>
          <div>
            <p className="font-medium">{t}</p>
            <p className="font-mono text-xs text-ink-2">{d}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}
