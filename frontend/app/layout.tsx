import type { Metadata } from "next";
import "./globals.css";
import { SiteHeader } from "@/components/ui/SiteHeader";
import { DISCLAIMER } from "@/lib/constants";

export const metadata: Metadata = {
  title: "CardioMamba — Mamba-based multi-lead ECG classification",
  description:
    "Research/educational demo of a pure-PyTorch bidirectional Mamba-1/S6 model for 5-class " +
    "multi-label ECG classification on PTB-XL.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2"
        >
          Skip to content
        </a>
        <SiteHeader />
        <main id="main" className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 sm:py-12">
          {children}
        </main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-xs text-ink-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
            <p>{DISCLAIMER}</p>
            <p>
              Data: PTB-XL 1.0.3 (PhysioNet, CC BY 4.0). Model: CardioMamba, epoch-5 checkpoint.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
