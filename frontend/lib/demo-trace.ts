import { readFileSync } from "node:fs";
import path from "node:path";
import manifest from "@/public/demo/manifest.json";

/** Real lead-II samples from the first demo record (server-side only, read at build time). */
export function heroTrace(seconds = 4, step = 2) {
  const sample = manifest.samples[0];
  const csv = readFileSync(path.join(process.cwd(), "public", sample.file), "utf8");
  const [header, ...rows] = csv.trim().split("\n");
  const col = header.split(",").indexOf("II");
  const n = Math.min(rows.length, seconds * sample.sampling_rate_hz);
  const values: number[] = [];
  for (let i = 0; i < n; i += step) values.push(Number(rows[i].split(",")[col]));
  return { values, ecgId: sample.ecg_id, seconds, fs: sample.sampling_rate_hz };
}
