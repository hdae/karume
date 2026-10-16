/**
 * artifacts — compares the verify artifacts of two environments (plan 0-3 of the 0.15 migration).
 *
 *     node tools/verify-diff/artifacts.ts --before <env dir> --after <env dir> [--family <name>]... [--json]
 *
 * An environment directory holds seats `<date>_<family>/` (ADR 0106): e.g. the frozen Deno baseline
 * `outputs/misc/0.15-baseline-2026-10-16/deno-nvidia-geforce-rtx-3080-ti` and the live
 * `outputs/verify/node-nvidia-geforce-rtx-3080-ti`. For every family present on both sides the
 * latest seat is taken and files are paired by name:
 *
 * - identical bytes → `identical`;
 * - `.png` / `.rgb` → 8-bit channels: max abs, mean abs, share of differing bytes;
 * - `.wav` → f32 samples: max abs, relative RMS of the difference;
 * - `.safetensors` → per tensor (f32 compared as floats, other dtypes as bytes), the worst tensor;
 * - `results.json` → cases whose status differs, and sha256-gated cases whose actual hash differs.
 *
 * Read-only. Exit code 0 even when differences exist (this is a report, not a gate); 1 when a file
 * cannot be read or decoded.
 */

import { readdir, readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { decodePng } from "../../packages/runtime/tests/helpers/png-decode.ts";
import { parseSafetensors, tensorBytes } from "../../packages/runtime/src/format/safetensors.ts";
import { decodeWav } from "../../packages/models/src/audio/wav.ts";

type Seat = { readonly family: string; readonly dir: string };

const seatsOf = async (root: string): Promise<Map<string, Seat>> => {
  const seats = new Map<string, Seat>();
  for (const entry of (await readdir(root, { withFileTypes: true })).filter((e) => e.isDirectory())) {
    const match = /^(\d{4}-\d{2}-\d{2})_(.+)$/.exec(entry.name);
    if (match === null) continue;
    const [, date, family] = match;
    const previous = seats.get(family!);
    // Latest date wins (names sort by date).
    if (previous === undefined || previous.dir < `${root}/${date}_${family}`) {
      seats.set(family!, { family: family!, dir: `${root}/${entry.name}` });
    }
  }
  return seats;
};

const owned = async (path: string): Promise<Uint8Array<ArrayBuffer>> => {
  const buffer = await readFile(path);
  const copy = new Uint8Array(new ArrayBuffer(buffer.byteLength));
  copy.set(buffer);
  return copy;
};

const sameBytes = (a: Uint8Array, b: Uint8Array): boolean => {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i += 1) if (a[i] !== b[i]) return false;
  return true;
};

type Stats = { readonly maxAbs: number; readonly meanAbs: number; readonly differing: number; readonly count: number; readonly relRms?: number };

const compareNumbers = (a: ArrayLike<number>, b: ArrayLike<number>, relative: boolean): Stats => {
  const count = Math.min(a.length, b.length);
  let maxAbs = 0, sumAbs = 0, differing = 0, sumSq = 0, refSq = 0;
  for (let i = 0; i < count; i += 1) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d > 0 || Number.isNaN(d)) differing += 1;
    if (d > maxAbs) maxAbs = d;
    sumAbs += d;
    sumSq += d * d;
    refSq += a[i]! * a[i]!;
  }
  const base: Stats = { maxAbs, meanAbs: count === 0 ? 0 : sumAbs / count, differing, count };
  return relative ? { ...base, relRms: refSq === 0 ? (sumSq === 0 ? 0 : Infinity) : Math.sqrt(sumSq / refSq) } : base;
};

type FileReport = { readonly file: string; readonly verdict: string; readonly detail?: string };

const fmt = (n: number): string => n === 0 ? "0" : n >= 1e-3 && n < 1e4 ? n.toPrecision(4) : n.toExponential(3);

const compareFile = async (name: string, before: string, after: string): Promise<FileReport> => {
  const [a, b] = await Promise.all([owned(`${before}/${name}`), owned(`${after}/${name}`)]);
  if (sameBytes(a, b)) return { file: name, verdict: "identical" };
  const ext = name.slice(name.lastIndexOf(".") + 1);
  if (ext === "png" || ext === "rgb") {
    const [x, y] = ext === "png"
      ? [(await decodePng(a, `${before}/${name}`)).data, (await decodePng(b, `${after}/${name}`)).data]
      : [a, b];
    if (x.length !== y.length) return { file: name, verdict: "size differs", detail: `${x.length} vs ${y.length} bytes` };
    const s = compareNumbers(x, y, false);
    return { file: name, verdict: "differs", detail: `u8 max ${s.maxAbs} mean ${fmt(s.meanAbs)} differing ${fmt(100 * s.differing / s.count)}% of ${s.count}` };
  }
  if (ext === "wav") {
    const x = decodeWav(a).data, y = decodeWav(b).data;
    if (x.length !== y.length) return { file: name, verdict: "length differs", detail: `${x.length} vs ${y.length} samples` };
    const s = compareNumbers(x, y, true);
    return { file: name, verdict: "differs", detail: `f32 max ${fmt(s.maxAbs)} relRMS ${fmt(s.relRms ?? 0)} differing ${fmt(100 * s.differing / s.count)}% of ${s.count}` };
  }
  if (ext === "safetensors") {
    const x = parseSafetensors(a.buffer), y = parseSafetensors(b.buffer);
    const lines: string[] = [];
    let worst = 0;
    for (const [tensor, view] of x.tensors) {
      const other = y.tensors.get(tensor);
      if (other === undefined) { lines.push(`${tensor}: missing after`); continue; }
      const xb = tensorBytes(x, view), yb = tensorBytes(y, other);
      if (sameBytes(xb, yb)) continue;
      if (view.dtype === "F32" && other.dtype === "F32") {
        const s = compareNumbers(
          new Float32Array(xb.buffer, xb.byteOffset, xb.byteLength / 4),
          new Float32Array(yb.buffer, yb.byteOffset, yb.byteLength / 4),
          true,
        );
        worst = Math.max(worst, s.relRms ?? 0);
        lines.push(`${tensor}: f32 max ${fmt(s.maxAbs)} relRMS ${fmt(s.relRms ?? 0)} differing ${fmt(100 * s.differing / s.count)}%`);
      } else {
        const s = compareNumbers(xb, yb, false);
        lines.push(`${tensor}: ${view.dtype} bytes differing ${fmt(100 * s.differing / s.count)}%`);
      }
    }
    for (const tensor of y.tensors.keys()) if (!x.tensors.has(tensor)) lines.push(`${tensor}: missing before`);
    return lines.length === 0
      ? { file: name, verdict: "identical tensors", detail: "header bytes differ only" }
      : { file: name, verdict: "differs", detail: `${lines.length} tensor(s), worst relRMS ${fmt(worst)}: ${lines.slice(0, 4).join(" | ")}${lines.length > 4 ? " | …" : ""}` };
  }
  if (name === "results.json") {
    type Case = { id: string; status: string; expected?: string; actual?: string };
    const cases = (bytes: Uint8Array<ArrayBuffer>): Map<string, Case> =>
      new Map(((JSON.parse(new TextDecoder().decode(bytes)) as { cases: Case[] }).cases).map((c) => [c.id, c]));
    const x = cases(a), y = cases(b);
    const lines: string[] = [];
    for (const [id, c] of x) {
      const o = y.get(id);
      if (o === undefined) { lines.push(`${id}: missing after`); continue; }
      if (c.status !== o.status) lines.push(`${id}: ${c.status} → ${o.status}`);
      else if (c.actual !== undefined && o.actual !== undefined && c.actual !== o.actual) lines.push(`${id}: sha differs`);
    }
    for (const id of y.keys()) if (!x.has(id)) lines.push(`${id}: missing before`);
    return lines.length === 0 ? { file: name, verdict: "same cases" } : { file: name, verdict: "cases differ", detail: lines.join(" | ") };
  }
  return { file: name, verdict: "differs", detail: `${a.byteLength} vs ${b.byteLength} bytes (no decoder for .${ext})` };
};

const main = async (): Promise<void> => {
  const { values } = parseArgs({
    options: { before: { type: "string" }, after: { type: "string" }, family: { type: "string", multiple: true }, json: { type: "boolean" } },
  });
  if (values.before === undefined || values.after === undefined) {
    console.error("usage: node tools/verify-diff/artifacts.ts --before <env dir> --after <env dir> [--family <name>]... [--json]");
    process.exit(2);
  }
  const [before, after] = await Promise.all([seatsOf(values.before), seatsOf(values.after)]);
  const families = [...before.keys()].filter((f) => after.has(f) && (values.family === undefined || values.family.includes(f))).sort();
  const report: Record<string, FileReport[]> = {};
  for (const family of families) {
    const b = before.get(family)!, a = after.get(family)!;
    const [bf, af] = await Promise.all([readdir(b.dir), readdir(a.dir)]);
    const names = new Set([...bf, ...af]);
    const rows: FileReport[] = [];
    for (const name of [...names].sort()) {
      if (!bf.includes(name)) rows.push({ file: name, verdict: "missing before" });
      else if (!af.includes(name)) rows.push({ file: name, verdict: "missing after" });
      else rows.push(await compareFile(name, b.dir, a.dir));
    }
    report[family] = rows;
  }
  if (values.json === true) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  for (const [family, rows] of Object.entries(report)) {
    const identical = rows.filter((r) => r.verdict === "identical" || r.verdict === "same cases").length;
    console.log(`== ${family}: ${identical}/${rows.length} identical`);
    for (const r of rows) if (r.verdict !== "identical" && r.verdict !== "same cases") console.log(`   ${r.verdict.padEnd(16)} ${r.file}${r.detail === undefined ? "" : `  ${r.detail}`}`);
  }
  const onlyBefore = [...before.keys()].filter((f) => !after.has(f));
  if (onlyBefore.length > 0) console.log(`families without an after seat: ${onlyBefore.join(", ")}`);
};

await main();
