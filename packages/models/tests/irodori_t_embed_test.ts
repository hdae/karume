// ホスト生成の `t_embed` 表を、exporter が出した **torch 基準の golden** と突き合わせる。
//
// `t_embed` は IR に載らない（`cos` が op 語彙に無い — ADR 0043 / 0047 決定 4）ので、この式の
// 正しさを見る門は他に無い。DiT の入力そのものなので、取り違えは「全 step で条件が別の時刻を
// 指す」形で latent 全体を変える。
//
// golden は `tools/exporter/irodori_pipeline.py` が書く
// `outputs/series/irodori-v4-small/pipeline/t-embed.safetensors`（`t` `[40]` と
// `t_embed` `[40,512]`）。生成していない環境では**明示 SKIP** する（ADR 0005）。GPU は要らない。

import { assert, assertEquals } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { tSchedule } from "../src/irodori/host/sampler.ts";
import { timestepEmbedding, timestepFrequencies } from "../src/irodori/host/t-embed.ts";

const GOLDEN = new URL(
  "../../../outputs/series/irodori-v4-small/pipeline/t-embed.safetensors",
  import.meta.url,
);

/** SKIP 時にそのまま貼れる生成コマンド（`irodori_pipeline.py` の docstring）。 */
const GENERATE_COMMAND =
  "cd tools/exporter && uv run --with 'transformers==5.14.1' python irodori_pipeline.py";

/**
 * `t_embed` 表の突合に使う許容誤差。
 *
 * 実測（`atol=0` の素の突合・40 step × 512 列 = 20,480 要素）: **maxAbs 3.05e-5**
 * （step 13 / t=0.674319 / 列 k=24）。内訳は 2 段に割れる:
 *
 * - **255/256 の周波数列は 1 ulp 以内**（≤ 5.96e-8 = 値域 [-1,1] の 1 ulp）
 * - **k=24 の 1 列だけが 3.05e-5**。原因は周波数表そのもので、torch の f32 `exp` が返す
 *   `freqs[24]` は JS の `Math.exp` 経由の値の **ちょうど 1 ulp 上**（実測: その 1 ビットを
 *   足すと列全体が 5.96e-8 まで落ちる）。`args = t·freqs[24] ≈ 284` は f32 の ulp が 3.05e-5 の
 *   領域なので、周波数の 1 ulp がそのまま `cos` / `sin` の絶対差として出る。
 *
 * つまり**大きい `args` を持つ列では超越関数の 1 ulp が絶対差 1e-5 級へ拡大する**構造で、
 * ビット一致させるには SLEEF の多項式近似を写すしかない（anima の `timestepsProj` と同じ判断）。
 *
 * atol 3e-4 は実測最悪 3.05e-5 の約 10 倍。実装バグの差はこの 4 桁以上上に出る（故障注入の
 * 実測）: **cos/sin の前後半反転で 1.41e+0**・**周波数の 1000 倍忘れで 2.00e+0**。
 */
const T_EMBED_ATOL = 3e-4;

/**
 * 閉形式 t スケジュールと golden の `t`（上流 `linspace`）の許容差。
 *
 * 実測 **5.96e-8**（= exporter が meta.json に載せる `tScheduleClosedFormMaxAbs` と同値）。
 * `linspace` は `start + i·step` を f32 で積むので、閉形式 `0.999·(1 − i/40)` とは最終 ulp が
 * 割れる点がある。t が入るのは `t_embed` と刻み幅だけで、latent への効き方は golden の
 * 突合閾値の 3 桁下（`irodori_pipeline.py` の `EULER_REFERENCE_ATOL` の doc）。
 */
const T_SCHEDULE_ATOL = 1e-7;

const goldenBytes = await Deno.readFile(GOLDEN).catch(() => undefined);
const AVAILABLE = goldenBytes !== undefined;
if (!AVAILABLE) {
  console.warn(
    `[karume] ${GOLDEN.pathname} が無いため Irodori の t_embed 突合を SKIP する` +
      `（full-loop golden はリポジトリ管理外）。生成: ${GENERATE_COMMAND}`,
  );
}

/** golden の `[名前 → f32 view]`（登録時に 1 回だけ読む）。 */
const readGolden = (bytes: Uint8Array<ArrayBuffer>) => {
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (name: string): { data: Float32Array; shape: readonly number[] } => {
    const tensor = file.tensors.get(name);
    assert(tensor !== undefined, `golden に '${name}' が無い`);
    assertEquals(tensor.dtype, "F32", `golden '${name}' の dtype`);
    return {
      data: new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4),
      shape: tensor.shape,
    };
  };
  return { t: view("t"), table: view("t_embed") };
};

Deno.test({
  name: "Irodori t_embed: ホスト式が torch 基準の golden 表と一致する",
  ignore: !AVAILABLE,
  fn: () => {
    const { t, table } = readGolden(goldenBytes as Uint8Array<ArrayBuffer>);
    const [steps, width] = [table.shape[0], table.shape[1]];
    assertEquals(t.shape, [steps], "t と t_embed の step 数");
    const frequencies = timestepFrequencies(width);
    let worst = 0;
    let where = "";
    for (let step = 0; step < steps; step += 1) {
      const row = timestepEmbedding(t.data[step], frequencies);
      assertEquals(row.length, width, `step ${step} の幅`);
      for (let column = 0; column < width; column += 1) {
        const diff = Math.abs(row[column] - table.data[step * width + column]);
        if (diff > worst) {
          worst = diff;
          where = `step ${step}（t=${t.data[step]}）列 ${column}`;
        }
      }
    }
    assert(
      worst <= T_EMBED_ATOL,
      `t_embed の最大絶対差 ${worst.toExponential(4)} が atol ${T_EMBED_ATOL} を超えた（${where}）`,
    );
  },
});

Deno.test({
  name: "Irodori t スケジュール: 閉形式が上流 linspace と 1 ulp 級で一致する",
  ignore: !AVAILABLE,
  fn: () => {
    const { t } = readGolden(goldenBytes as Uint8Array<ArrayBuffer>);
    const steps = t.shape[0];
    // golden は末尾の 0 を落とした `[:-1]` なので、こちらも同じ範囲だけを見る。
    const schedule = tSchedule(steps, 0.999);
    let worst = 0;
    for (let index = 0; index < steps; index += 1) {
      worst = Math.max(worst, Math.abs(schedule[index] - t.data[index]));
    }
    assert(
      worst <= T_SCHEDULE_ATOL,
      `t スケジュールの最大絶対差 ${worst.toExponential(4)} が atol ${T_SCHEDULE_ATOL} を超えた`,
    );
  },
});
