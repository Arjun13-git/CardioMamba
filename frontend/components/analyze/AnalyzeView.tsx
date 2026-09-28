"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { PredictionPanel } from "@/components/analyze/PredictionPanel";
import { EcgViewer } from "@/components/ecg/EcgViewer";
import { Card, CardTitle } from "@/components/ui/primitives";
import manifest from "@/public/demo/manifest.json";
import { ApiError, getModelInfo, predict } from "@/lib/api";
import type { ClassName, DemoSample, ModelInfo, PredictionResponse } from "@/lib/types";

const MAX_BYTES = 2 * 1024 * 1024;
const SAMPLES = manifest.samples as DemoSample[];
const STAGES = [
  "Validating the 12-lead recording",
  "Resampling 500 → 100 Hz and normalizing",
  "Patch stem → 250 tokens",
  "Running 4 bidirectional Mamba blocks",
  "Sigmoid probabilities + frozen thresholds",
];

type Status =
  | { kind: "idle" }
  | { kind: "loading"; source: string }
  | { kind: "done"; result: PredictionResponse; source: string; sample?: DemoSample; roundTripMs: number }
  | { kind: "error"; message: string };

type Backend = { state: "checking" } | { state: "online"; info: ModelInfo } | { state: "offline"; message: string };

export function AnalyzeView() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [backend, setBackend] = useState<Backend>({ state: "checking" });
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<DemoSample>(SAMPLES[0]);
  const inputRef = useRef<HTMLInputElement>(null);

  const fetchBackend = useCallback(() => {
    getModelInfo()
      .then((info) => setBackend({ state: "online", info }))
      .catch((e: unknown) =>
        setBackend({ state: "offline", message: e instanceof ApiError ? e.message : String(e) }),
      );
  }, []);

  const checkBackend = useCallback(() => {
    setBackend({ state: "checking" });
    fetchBackend();
  }, [fetchBackend]);

  useEffect(fetchBackend, [fetchBackend]); // initial state is already "checking"

  async function run(file: File, source: string, sample?: DemoSample) {
    setStatus({ kind: "loading", source });
    const t0 = performance.now();
    try {
      const result = await predict(file);
      setStatus({ kind: "done", result, source, sample, roundTripMs: performance.now() - t0 });
      if (backend.state !== "online") checkBackend();
    } catch (e) {
      setStatus({
        kind: "error",
        message: e instanceof ApiError ? e.message : "Unexpected error while contacting the backend.",
      });
    }
  }

  function onFile(file: File | undefined) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setStatus({
        kind: "error",
        message:
          "Unsupported file type. Upload a .csv with a header row of the 12 lead names and " +
          "5000 rows (10 s at 500 Hz) or 1000 rows (10 s at 100 Hz), values in mV.",
      });
      return;
    }
    if (file.size > MAX_BYTES) {
      setStatus({ kind: "error", message: "File is larger than 2 MB." });
      return;
    }
    void run(file, file.name);
  }

  async function loadDemo(sample: DemoSample) {
    setSelected(sample);
    setStatus({ kind: "loading", source: `Demo sample · PTB-XL record ${sample.ecg_id}` });
    try {
      const res = await fetch(sample.file);
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      const file = new File([blob], sample.file.split("/").pop() ?? "demo.csv", { type: "text/csv" });
      await run(file, `Demo sample · PTB-XL record ${sample.ecg_id}`, sample);
    } catch {
      setStatus({ kind: "error", message: "Could not load the bundled demo sample file." });
    }
  }

  const busy = status.kind === "loading";

  return (
    <div className="space-y-6">
      <BackendStatus backend={backend} onRetry={checkBackend} />

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardTitle sub="Bundled PTB-XL records from the held-out test fold — for a reliable live demo.">
            Demo sample
          </CardTitle>
          <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Demo samples">
            {SAMPLES.map((s) => (
              <button
                key={s.id}
                type="button"
                role="radio"
                aria-checked={selected.id === s.id}
                onClick={() => setSelected(s)}
                className={`rounded-lg border px-3 py-1.5 text-left text-sm transition-colors ${
                  selected.id === s.id
                    ? "border-accent/60 bg-accent/10 text-ink"
                    : "border-line text-ink-2 hover:border-line-strong hover:text-ink"
                }`}
              >
                <span className="font-mono">#{s.ecg_id}</span>
                <span className="ml-2 text-xs text-ink-3">ref. {s.reference_labels.join(", ")}</span>
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => loadDemo(selected)}
            disabled={busy}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-[#062420] transition-colors hover:bg-[#5eead4] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Load Demo Sample
          </button>
          <p className="mt-3 text-xs text-ink-3">
            {manifest.provenance}. Selection rule: {manifest.selection_rule}. The original 500 Hz
            signal is sent through the same preprocessing as any upload.
          </p>
        </Card>

        <Card>
          <CardTitle sub="CSV: header row I, II, III, aVR, aVL, aVF, V1–V6 · values in mV · 5000 rows (500 Hz) or 1000 rows (100 Hz) · 10 s · ≤ 2 MB.">
            Upload a compatible ECG recording
          </CardTitle>
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              onFile(e.dataTransfer.files[0]);
            }}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-4 py-8 text-center transition-colors ${
              dragging ? "border-accent bg-accent/5" : "border-line-strong hover:border-accent/50"
            } ${busy ? "pointer-events-none opacity-50" : ""}`}
          >
            <svg viewBox="0 0 24 24" className="h-7 w-7 text-ink-3" aria-hidden="true">
              <path fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
            </svg>
            <span className="text-sm">
              Drop ECG file here or <span className="text-accent underline underline-offset-4">choose file</span>
            </span>
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              onChange={(e) => {
                onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
          <p className="mt-3 text-xs text-ink-3">
            Files are processed in memory by the local backend and not stored.
          </p>
        </Card>
      </div>

      <div aria-live="polite">
        {status.kind === "loading" ? <Processing source={status.source} /> : null}
        {status.kind === "error" ? <ErrorCard message={status.message} /> : null}
        {status.kind === "done" ? (
          <Results
            result={status.result}
            source={status.source}
            sample={status.sample}
            roundTripMs={status.roundTripMs}
          />
        ) : null}
      </div>
    </div>
  );
}

function BackendStatus({ backend, onRetry }: { backend: Backend; onRetry: () => void }) {
  if (backend.state === "checking")
    return <p className="text-sm text-ink-3">Checking inference backend…</p>;
  if (backend.state === "offline")
    return (
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-danger/30 bg-danger/5 px-4 py-3 text-sm">
        <span className="h-2 w-2 rounded-full bg-danger" aria-hidden="true" />
        <span className="text-ink-2">Backend offline — {backend.message}</span>
        <button type="button" onClick={onRetry} className="ml-auto rounded-md border border-line-strong px-2.5 py-1 text-xs hover:bg-surface-2">
          Retry
        </button>
      </div>
    );
  const i = backend.info;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-xl border border-line bg-surface/60 px-4 py-2.5 text-sm text-ink-2">
      <span className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_8px_var(--accent)]" aria-hidden="true" />
        Backend online
      </span>
      <span className="tabular">CardioMamba · epoch-{i.checkpoint_epoch} checkpoint · {i.parameters.toLocaleString("en-US")} parameters</span>
      <span className="tabular text-ink-3">
        {i.device.toUpperCase()} · {i.precision.toUpperCase()} · {i.preprocessing_version}
      </span>
    </div>
  );
}

function Processing({ source }: { source: string }) {
  return (
    <Card className="relative overflow-hidden">
      <div className="absolute inset-x-0 top-0 h-0.5 overflow-hidden bg-[var(--grid)]" aria-hidden="true">
        <div className="h-full w-1/3 bg-accent" style={{ animation: "scan 1.2s linear infinite" }} />
      </div>
      <p className="font-medium">Processing ECG</p>
      <p className="mt-0.5 text-sm text-ink-3">{source}</p>
      <ul className="mt-4 grid gap-2 text-sm text-ink-2 sm:grid-cols-2 lg:grid-cols-5">
        {STAGES.map((s) => (
          <li key={s} className="flex items-center gap-2 rounded-lg border border-line bg-surface-2/40 px-3 py-2">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" aria-hidden="true" />
            {s}
          </li>
        ))}
      </ul>
    </Card>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-2xl border border-danger/40 bg-danger/5 p-5">
      <p className="font-medium text-ink">Unable to process this ECG.</p>
      <p className="mt-1 text-sm text-ink-2">{message}</p>
    </div>
  );
}

function Results({
  result,
  source,
  sample,
  roundTripMs,
}: {
  result: PredictionResponse;
  source: string;
  sample?: DemoSample;
  roundTripMs: number;
}) {
  const inp = result.input;
  const trace = [
    ["Input", `${inp.samples} × ${inp.leads} @ ${inp.sampling_rate_hz} Hz (${inp.format.toUpperCase()})`],
    ["Resampled", `1000 × 12 @ ${inp.model_sampling_rate_hz} Hz${inp.sampling_rate_hz === 100 ? " (already 100 Hz)" : " · resample_poly 1/5"}`],
    ["Normalized", "per lead, frozen training-fold statistics"],
    ["Model", `CardioMamba epoch ${result.model.checkpoint_epoch} · ${result.device.toUpperCase()} · ${result.inference_ms} ms forward pass`],
    ["Output", "sigmoid → 5 probabilities → frozen thresholds"],
  ];
  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-[1.6fr_1fr]">
        <Card>
          <CardTitle sub={source}>12-lead ECG</CardTitle>
          <dl className="mb-4 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
            {[
              ["Sampling rate", `${result.waveform.sampling_rate_hz} Hz`],
              ["Duration", "10 s"],
              ["Leads", String(result.waveform.leads.length)],
              ["Samples", String(result.waveform.data[0].length)],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-line bg-surface-2/40 px-3 py-2">
                <dt className="text-xs text-ink-3">{k}</dt>
                <dd className="tabular font-medium">{v}</dd>
              </div>
            ))}
          </dl>
          <EcgViewer data={result.waveform.data} leads={result.waveform.leads} fs={result.waveform.sampling_rate_hz} />
        </Card>
        <Card>
          <CardTitle sub="Five independent sigmoid outputs, thresholded with the frozen validation thresholds.">
            Model predictions
          </CardTitle>
          <PredictionPanel
            predictions={result.predictions}
            referenceLabels={sample?.reference_labels as ClassName[] | undefined}
          />
        </Card>
      </div>
      <Card>
        <CardTitle sub={`Request ${result.request_id.slice(0, 8)} · round trip ${roundTripMs.toFixed(0)} ms (upload + preprocessing + inference).`}>
          What the backend just did
        </CardTitle>
        <ol className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-5">
          {trace.map(([k, v], i) => (
            <li key={k} className="rounded-lg border border-line bg-surface-2/40 p-3">
              <p className="text-xs text-ink-3">
                {i + 1}. {k}
              </p>
              <p className="mt-1 text-ink-2">{v}</p>
            </li>
          ))}
        </ol>
        <p className="mt-4 text-sm text-ink-3">
          Curious how the Bi-Mamba blocks work?{" "}
          <Link href="/architecture" className="text-accent underline underline-offset-4">
            See the architecture
          </Link>
          .
        </p>
      </Card>
    </div>
  );
}
