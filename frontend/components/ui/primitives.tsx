import Link from "next/link";
import type { ReactNode } from "react";
import { DISCLAIMER } from "@/lib/constants";

export function Card({
  children,
  className = "",
  as: Tag = "section",
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article";
}) {
  return (
    <Tag
      className={`min-w-0 rounded-2xl border border-line bg-surface/80 p-5 shadow-[0_1px_0_rgba(255,255,255,0.03)_inset] sm:p-6 ${className}`}
    >
      {children}
    </Tag>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <p className="text-xs font-medium uppercase tracking-[0.16em] text-accent">{children}</p>
  );
}

export function PageHeading({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="max-w-3xl space-y-3">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h1 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">{title}</h1>
      {children ? <div className="text-ink-2 text-pretty">{children}</div> : null}
    </div>
  );
}

export function CardTitle({ children, sub }: { children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-4">
      <h2 className="text-base font-semibold tracking-tight">{children}</h2>
      {sub ? <p className="mt-1 text-sm text-ink-3">{sub}</p> : null}
    </div>
  );
}

export function StatTile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface-2/60 p-4">
      <p className="text-sm text-ink-2">{label}</p>
      <p className="tabular mt-1 text-3xl font-semibold tracking-tight">{value}</p>
      {note ? <p className="mt-1 text-xs text-ink-3">{note}</p> : null}
    </div>
  );
}

export function ButtonLink({
  href,
  children,
  variant = "primary",
}: {
  href: string;
  children: ReactNode;
  variant?: "primary" | "ghost";
}) {
  const styles =
    variant === "primary"
      ? "bg-accent text-[#062420] hover:bg-[#5eead4]"
      : "border border-line-strong text-ink hover:bg-surface-2";
  return (
    <Link
      href={href}
      className={`inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ${styles}`}
    >
      {children}
    </Link>
  );
}

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <div
      role="note"
      className={`flex gap-3 rounded-xl border border-warn/30 bg-warn/5 text-sm text-ink-2 ${
        compact ? "px-3 py-2" : "p-4"
      }`}
    >
      <svg viewBox="0 0 20 20" className="mt-0.5 h-4 w-4 shrink-0 text-warn" aria-hidden="true">
        <path
          fill="currentColor"
          d="M10 2 1 18h18L10 2Zm0 5.5c.5 0 .9.4.9.9v4.2a.9.9 0 1 1-1.8 0V8.4c0-.5.4-.9.9-.9Zm0 8.9a1.1 1.1 0 1 1 0-2.2 1.1 1.1 0 0 1 0 2.2Z"
        />
      </svg>
      <p>
        <strong className="font-semibold text-ink">Research prototype</strong> —{" "}
        {DISCLAIMER.replace("Research prototype — ", "")}
      </p>
    </div>
  );
}

export function fmt(v: number, digits = 4): string {
  return v.toFixed(digits);
}
