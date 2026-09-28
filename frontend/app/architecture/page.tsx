import type { Metadata } from "next";
import { ArchitectureDiagram, BiMambaBlockDiagram, MixerSteps } from "@/components/model/ArchitectureDiagram";
import { ButtonLink, Card, CardTitle, PageHeading } from "@/components/ui/primitives";
import model from "@/data/model.json";

export const metadata: Metadata = { title: "Model architecture — CardioMamba" };

export default function ArchitecturePage() {
  const c = model.config;
  const specs: [string, string][] = [
    ["Parameters", model.parameters.toLocaleString("en-US")],
    ["d_model", String(c.d_model)],
    ["d_state", String(c.d_state)],
    ["d_conv", String(c.d_conv)],
    ["expand", `${c.expand} (inner width ${c.expand * c.d_model})`],
    ["dt_rank", String(c.dt_rank)],
    ["Layers", String(c.n_layers)],
    ["Bidirectional", c.bidirectional ? "Yes" : "No"],
    ["Dropout", String(c.dropout)],
    ["Positional encoding", "None"],
    ["Patch stem", `kernel ${c.patch_size}, stride ${c.patch_stride} → 250 tokens`],
    ["Selective scan", "pure PyTorch, always FP32"],
  ];
  const why: [string, string][] = [
    ["Selective sequence modeling", "Mamba’s state-space recurrence has input-dependent Δ, B and C, so it can choose what to keep or forget along the ECG — with cost linear in sequence length."],
    ["Both temporal directions", "A classifier sees the whole 10 s record, so each block also reads the sequence right-to-left; every token gets context from before and after."],
    ["Short token sequence", "The Conv1D patch stem turns 1000 samples into 250 tokens of 40 ms (a QRS complex spans ~2–3 tokens), keeping memory within a 4 GB GPU."],
    ["Multi-label output", "Five independent sigmoids let one record carry several superclasses at once (e.g. MI and CD), matching how PTB-XL is annotated."],
  ];
  return (
    <div className="space-y-8">
      <PageHeading eyebrow="How it works" title="Model architecture">
        CardioMamba is a pure-PyTorch implementation of bidirectional Mamba-1 (S6) blocks — no
        external Mamba kernels. The shapes below are the model&apos;s actual tensor shapes.
      </PageHeading>

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardTitle sub="One forward pass for a batch of one recording.">Tensor flow</CardTitle>
          <ArchitectureDiagram nLayers={c.n_layers} dModel={c.d_model} numClasses={c.num_classes} />
        </Card>
        <div className="space-y-6">
          <Card>
            <CardTitle sub={`From the frozen checkpoint (epoch ${model.checkpoint_epoch}).`}>Model specification</CardTitle>
            <dl className="grid grid-cols-1 gap-x-6 sm:grid-cols-2">
              {specs.map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between gap-4 border-b border-line/70 py-2 text-sm">
                  <dt className="text-ink-3">{k}</dt>
                  <dd className="tabular text-right font-medium">{v}</dd>
                </div>
              ))}
            </dl>
          </Card>
          <Card>
            <CardTitle>Inside one bidirectional block</CardTitle>
            <BiMambaBlockDiagram />
          </Card>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardTitle sub="Per direction; shapes per token (d_model 128, inner width 256, 16 states).">The S6 selective state-space mixer</CardTitle>
          <MixerSteps />
        </Card>
        <Card>
          <CardTitle>Why this design</CardTitle>
          <ul className="space-y-4">
            {why.map(([t, d]) => (
              <li key={t}>
                <p className="font-medium">{t}</p>
                <p className="mt-0.5 text-sm text-ink-2">{d}</p>
              </li>
            ))}
          </ul>
          <div className="mt-6 flex flex-wrap gap-3">
            <ButtonLink href="/analyze">Run it on an ECG →</ButtonLink>
            <ButtonLink href="/results" variant="ghost">See held-out results</ButtonLink>
          </div>
        </Card>
      </div>
    </div>
  );
}
