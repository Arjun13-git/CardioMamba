import type { Metadata } from "next";
import { LineChart } from "@/components/results/LineChart";
import { Card, CardTitle, PageHeading, StatTile, fmt } from "@/components/ui/primitives";
import results from "@/data/results.json";
import training from "@/data/training.json";
import { CLASS_INFO, CLASS_ORDER } from "@/lib/constants";

export const metadata: Metadata = { title: "Results — CardioMamba" };

export default function ResultsPage() {
  const o = results.overall;
  const v = results.validation_reference;
  const epochs = training.epochs.map((e) => e.epoch);
  const best = training.best_epoch;
  const stopMarker = { x: training.epochs_completed, label: `stopped (${training.epochs_completed})` };
  const bestMarker = { x: best, label: `best epoch ${best}` };
  return (
    <div className="space-y-8">
      <PageHeading eyebrow="Frozen held-out evaluation" title="Results">
        Official one-time evaluation of the frozen epoch-{results.checkpoint_epoch} checkpoint on
        PTB-XL fold 10 ({results.n_records.toLocaleString("en-US")} labelled records,{" "}
        {results.n_patients.toLocaleString("en-US")} patients), never used for training, model selection
        or threshold choice.
      </PageHeading>

      <section aria-label="Headline test metrics" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Macro AUROC" value={fmt(o.macro_auroc)} note="threshold-free · primary" />
        <StatTile label="Macro AUPRC" value={fmt(o.macro_auprc)} note="threshold-free · primary" />
        <StatTile label="Macro F1" value={fmt(o.macro_f1)} note="frozen validation thresholds" />
        <StatTile label="Micro F1" value={fmt(o.micro_f1)} note="frozen validation thresholds" />
      </section>

      <Card>
        <CardTitle sub="Fold 10 (test). AUROC/AUPRC from probabilities; precision, recall and F1 at the frozen per-class thresholds.">
          Per-class test metrics
        </CardTitle>
        <div className="-mx-2 overflow-x-auto">
          <table className="tabular w-full min-w-[640px] text-sm">
            <caption className="sr-only">Per-class held-out test metrics</caption>
            <thead className="text-left text-xs text-ink-3">
              <tr className="border-b border-line">
                <th scope="col" className="px-2 py-2 font-medium">Class</th>
                <th scope="col" className="px-2 py-2 text-right font-medium">AUROC</th>
                <th scope="col" className="px-2 py-2 text-right font-medium">AUPRC</th>
                <th scope="col" className="px-2 py-2 text-right font-medium">F1</th>
                <th scope="col" className="px-2 py-2 text-right font-medium">Precision</th>
                <th scope="col" className="px-2 py-2 text-right font-medium">Recall</th>
                <th scope="col" className="px-2 py-2 text-right font-medium">Threshold</th>
                <th scope="col" className="px-2 py-2 text-right font-medium">Support</th>
              </tr>
            </thead>
            <tbody>
              {CLASS_ORDER.map((c) => {
                const m = results.per_class[c];
                return (
                  <tr key={c} className="border-b border-line/60 hover:bg-surface-2/40">
                    <th scope="row" className="px-2 py-2.5 text-left font-normal">
                      <span className="font-mono font-semibold">{c}</span>
                      <span className="ml-2 text-ink-3">{CLASS_INFO[c].name}</span>
                    </th>
                    <td className="px-2 py-2.5 text-right">{fmt(m.auroc)}</td>
                    <td className="px-2 py-2.5 text-right">{fmt(m.auprc)}</td>
                    <td className="px-2 py-2.5 text-right">{fmt(m.f1)}</td>
                    <td className="px-2 py-2.5 text-right text-ink-2">{fmt(m.precision)}</td>
                    <td className="px-2 py-2.5 text-right text-ink-2">{fmt(m.recall)}</td>
                    <td className="px-2 py-2.5 text-right text-ink-2">{m.threshold.toFixed(2)}</td>
                    <td className="px-2 py-2.5 text-right text-ink-2">{m.support}</td>
                  </tr>
                );
              })}
              <tr className="font-medium">
                <th scope="row" className="px-2 py-2.5 text-left">Macro</th>
                <td className="px-2 py-2.5 text-right">{fmt(o.macro_auroc)}</td>
                <td className="px-2 py-2.5 text-right">{fmt(o.macro_auprc)}</td>
                <td className="px-2 py-2.5 text-right">{fmt(o.macro_f1)}</td>
                <td className="px-2 py-2.5 text-right">{fmt(o.macro_precision)}</td>
                <td className="px-2 py-2.5 text-right">{fmt(o.macro_recall)}</td>
                <td className="px-2 py-2.5 text-right text-ink-3">—</td>
                <td className="px-2 py-2.5 text-right text-ink-3">—</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-ink-3">
          Micro precision {fmt(o.micro_precision)} · micro recall {fmt(o.micro_recall)} · micro F1{" "}
          {fmt(o.micro_f1)}. HYP is the rarest class ({results.per_class.HYP.support} of{" "}
          {results.n_records.toLocaleString("en-US")} test records) and the weakest. Source:{" "}
          <span className="font-mono">{results.source}</span>.
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardTitle sub={`Recorded per-epoch validation (fold 9). Best epoch ${best}: ${fmt(training.best_val_macro_auroc)}; early stopping after ${training.patience} epochs without improvement.`}>
            Validation macro AUROC
          </CardTitle>
          <LineChart
            x={epochs}
            series={[{ name: "Validation macro AUROC", color: "var(--series-blue)", values: training.epochs.map((e) => e.val_macro_auroc) }]}
            yLabel="Validation macro AUROC"
            markers={[bestMarker, stopMarker]}
          />
        </Card>
        <Card>
          <CardTitle sub="Training loss keeps falling while validation loss rises after epoch 5 — the overfitting that early stopping guards against.">
            BCE loss
          </CardTitle>
          <LineChart
            x={epochs}
            series={[
              { name: "Training loss", color: "var(--series-orange)", values: training.epochs.map((e) => e.train_loss) },
              { name: "Validation loss", color: "var(--series-blue)", values: training.epochs.map((e) => e.val_loss) },
            ]}
            yLabel="BCE loss"
            markers={[bestMarker]}
          />
        </Card>
      </div>

      <details className="rounded-2xl border border-line bg-surface/60 p-5 text-sm">
        <summary className="cursor-pointer font-medium">Training history as a table</summary>
        <div className="mt-3 overflow-x-auto">
          <table className="tabular w-full min-w-[520px]">
            <thead className="text-left text-xs text-ink-3">
              <tr className="border-b border-line">
                {["Epoch", "Train loss", "Val loss", "Val macro AUROC", "Val macro AUPRC"].map((h) => (
                  <th key={h} scope="col" className="px-2 py-1.5 text-right font-medium first:text-left">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {training.epochs.map((e) => (
                <tr key={e.epoch} className={`border-b border-line/50 ${e.epoch === best ? "text-accent" : ""}`}>
                  <td className="px-2 py-1.5">{e.epoch}{e.epoch === best ? " (best)" : ""}</td>
                  <td className="px-2 py-1.5 text-right">{e.train_loss.toFixed(4)}</td>
                  <td className="px-2 py-1.5 text-right">{e.val_loss.toFixed(4)}</td>
                  <td className="px-2 py-1.5 text-right">{e.val_macro_auroc.toFixed(4)}</td>
                  <td className="px-2 py-1.5 text-right">{e.val_macro_auprc.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardTitle>Methodology</CardTitle>
          <dl className="space-y-0 text-sm">
            {[
              ["Dataset", "PTB-XL 1.0.3 (PhysioNet)"],
              ["Train", "folds 1–8 (17,084 labelled records)"],
              ["Validation", "fold 9 (2,146)"],
              ["Test", `fold 10 (${results.n_records.toLocaleString("en-US")}), evaluated once`],
              ["Input", "12-lead, 10-second ECG"],
              ["Sampling", "500 Hz → 100 Hz (resample_poly), train-fold normalization"],
              ["Loss / optimizer", `BCEWithLogits · AdamW lr ${training.config.learning_rate}, wd ${training.config.weight_decay}, batch ${training.config.batch_size}`],
              ["Model selection", `validation macro AUROC (best epoch ${best} of ${training.epochs_completed})`],
              ["Thresholds", "per-class F1 maximisation on fold 9, frozen"],
              ["Seed", String(training.config.seed)],
            ].map(([k, val]) => (
              <div key={k} className="flex justify-between gap-4 border-b border-line/60 py-2">
                <dt className="text-ink-3">{k}</dt>
                <dd className="text-right">{val}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card>
          <CardTitle sub="Same checkpoint. Validation F1 is optimistic because thresholds were tuned there; the test values are the unbiased estimate.">
            Validation vs held-out test
          </CardTitle>
          <table className="tabular w-full text-sm">
            <thead className="text-left text-xs text-ink-3">
              <tr className="border-b border-line">
                <th scope="col" className="py-2 font-medium">Metric</th>
                <th scope="col" className="py-2 text-right font-medium">Validation (fold 9)</th>
                <th scope="col" className="py-2 text-right font-medium">Test (fold 10)</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Macro AUROC", v.macro_auroc, o.macro_auroc],
                ["Macro AUPRC", v.macro_auprc, o.macro_auprc],
                ["Macro F1", v.macro_f1_calibrated, o.macro_f1],
                ["Micro F1", v.micro_f1_calibrated, o.micro_f1],
              ].map(([k, a, b]) => (
                <tr key={k as string} className="border-b border-line/60">
                  <th scope="row" className="py-2 text-left font-normal">{k}</th>
                  <td className="py-2 text-right text-ink-2">{fmt(a as number)}</td>
                  <td className="py-2 text-right font-medium">{fmt(b as number)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-ink-3">
            Single training run (seed {training.config.seed}); no confidence intervals were
            estimated. Research metrics — not a clinical validation.
          </p>
        </Card>
      </div>
    </div>
  );
}
