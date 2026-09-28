import type { Metadata } from "next";
import { AnalyzeView } from "@/components/analyze/AnalyzeView";
import { Disclaimer, PageHeading } from "@/components/ui/primitives";

export const metadata: Metadata = { title: "Analyze ECG — CardioMamba" };

export default function AnalyzePage() {
  return (
    <div className="space-y-8">
      <PageHeading eyebrow="Live inference" title="Analyze ECG">
        Run the frozen CardioMamba checkpoint on a 12-lead, 10-second ECG. The recording goes
        through the exact production preprocessing used in training before inference.
      </PageHeading>
      <Disclaimer compact />
      <AnalyzeView />
    </div>
  );
}
