"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV } from "@/lib/constants";

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="8" fill="#0f1620" stroke="#2c3a4b" />
      <path
        d="M4 17h6l2-5 3 10 3-14 2 9h8"
        fill="none"
        stroke="#2dd4bf"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
          <Logo />
          <span>
            Cardio<span className="text-accent">Mamba</span>
          </span>
        </Link>
        <nav aria-label="Main" className="-mx-1 flex flex-1 flex-wrap gap-1">
          {NAV.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                  active
                    ? "bg-surface-2 text-ink ring-1 ring-line-strong"
                    : "text-ink-2 hover:bg-surface hover:text-ink"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
        <span className="hidden rounded-full border border-line px-2.5 py-1 text-xs text-ink-3 md:inline">
          Research prototype
        </span>
      </div>
    </header>
  );
}
