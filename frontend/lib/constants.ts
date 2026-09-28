import modelData from "@/data/model.json";
import type { ClassName } from "./types";

// Frozen class order of the checkpoint (never reorder).
export const CLASS_ORDER = modelData.class_order as ClassName[];

// PTB-XL diagnostic superclass names (dataset terminology, not diagnoses).
export const CLASS_INFO: Record<ClassName, { name: string }> = {
  NORM: { name: "Normal ECG" },
  MI: { name: "Myocardial infarction" },
  STTC: { name: "ST/T change" },
  CD: { name: "Conduction disturbance" },
  HYP: { name: "Hypertrophy" },
};

// Frozen validation (fold 9) thresholds, exported from the run's thresholds.json.
export const THRESHOLDS = modelData.thresholds as Record<ClassName, number>;

export const DISCLAIMER =
  "Research prototype — not a medical diagnostic system. Model outputs are for educational " +
  "and research demonstration only.";

export const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/analyze", label: "Analyze ECG" },
  { href: "/architecture", label: "Architecture" },
  { href: "/results", label: "Results" },
] as const;
