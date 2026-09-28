import { HeroTrace } from "@/components/dashboard/HeroTrace";
import { PipelineFlow } from "@/components/dashboard/PipelineFlow";
import { ButtonLink, Card, CardTitle, Disclaimer, Eyebrow, StatTile } from "@/components/ui/primitives";
import model from "@/data/model.json";
import results from "@/data/results.json";
import { heroTrace } from "@/lib/demo-trace";

export default function DashboardPage() {
  const trace = heroTrace();
  const o = results.overall;
  return (
    <div className="space-y-12">
      <section className="grid items-center gap-8 lg:grid-cols-[1.1fr_1fr]">
        <div className="space-y-6">
          <Eyebrow>Research demo · PTB-XL · Selective state-space model</Eyebrow>
          <h1 className="text-4xl font-semibold tracking-tight text-balance sm:text-5xl">
            Cardio<span className="text-accent">Mamba</span>
          </h1>
          <p className="text-xl text-ink text-balance sm:text-2xl">
            Mamba-based deep learning for multi-lead ECG classification
          </p>
          <p className="max-w-xl text-ink-2 text-pretty">
            A pure-PyTorch bidirectional Mamba-1 (S6) network for five-class multi-label
            classification of 10-second, 12-lead ECGs from PTB-XL.
          </p>
          <div className="flex flex-wrap gap-3">
            <ButtonLink href="/analyze">Analyze an ECG →</ButtonLink>
            <ButtonLink href="/architecture" variant="ghost">
              How does it work?
            </ButtonLink>
            <ButtonLink href="/results" variant="ghost">
              Held-out results
            </ButtonLink>
          </div>
        </div>
        <HeroTrace
          values={trace.values}
          caption={`Lead II, first ${trace.seconds} s of PTB-XL record ${trace.ecgId} (demo sample, original ${trace.fs} Hz signal).`}
        />
      </section>

      <section aria-label="Key facts" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="ECG leads" value="12" note="I, II, III, aVR, aVL, aVF, V1–V6" />
        <StatTile label="Input sampling" value="100 Hz" note="resampled from 500 Hz · 10 s" />
        <StatTile label="Bi-Mamba blocks" value={String(model.config.n_layers)} note={`${model.parameters.toLocaleString("en-US")} parameters`} />
        <StatTile label="Diagnostic superclasses" value="5" note="NORM · MI · STTC · CD · HYP" />
      </section>

      <Card>
        <CardTitle sub="From raw 12-lead recording to five independent label probabilities. Tensor shapes are the real shapes used by the model.">
          Architecture pipeline
        </CardTitle>
        <PipelineFlow />
      </Card>

      <section className="grid gap-4 md:grid-cols-3">
        <Card as="article">
          <CardTitle>What goes in</CardTitle>
          <p className="text-sm text-ink-2">
            A 10-second, 12-lead ECG. The original 500 Hz recording is resampled to 100 Hz with
            an anti-aliasing polyphase filter and normalized per lead using statistics from the
            training folds only.
          </p>
        </Card>
        <Card as="article">
          <CardTitle>How it is processed</CardTitle>
          <p className="text-sm text-ink-2">
            A convolutional patch stem turns the signal into 250 tokens of 40 ms. Four
            bidirectional Mamba blocks model the token sequence with an input-dependent
            (selective) state-space recurrence, read left-to-right and right-to-left.
          </p>
        </Card>
        <Card as="article">
          <CardTitle>What comes out</CardTitle>
          <p className="text-sm text-ink-2">
            Five independent probabilities (multi-label), one per PTB-XL superclass. Frozen
            per-class thresholds chosen on the validation fold turn them into predicted labels.
            Held-out test macro AUROC:{" "}
            <span className="tabular font-medium text-ink">{o.macro_auroc.toFixed(4)}</span>.
          </p>
        </Card>
      </section>

      <Disclaimer />
    </div>
  );
}
