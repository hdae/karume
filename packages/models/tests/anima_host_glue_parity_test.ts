/**
 * Anima のホスト糊（IR の外側で TS が持つ段）を、Python の参照フィクスチャと**ビット一致**で
 * 突き合わせる。対象は 4 関数: `sigmaSchedule` / `cfgEulerStep`（`sampler.ts`）・
 * `denormalizeLatents` / `padSequence`（`latents.ts`）。GPU は要らない。
 *
 * フィクスチャは `tools/export-recipes/anima/pipeline_ref.py` が書く
 * `outputs/series/anima-pipeline{,-f16,-i8,-turbo-f16}/pipeline.{json,safetensors}`
 * （生成コマンドと役割の索引は recipe README の「Anima host pipeline reference fixture」節）。
 * 4 変種とも、各段の入力と出力が同じファイルに入っている。そのため、DiT の誤差と混ざらない
 * 形でホスト糊だけを見られる（DiT の生出力 `noise_*` を入力として渡し、次の latent を比べる）。
 *
 * MUST: 突合は**ビット一致**（tolerance を持たない）。4 関数はどれも `Math.fround` で 1 演算
 * ずつ f32 に丸める実装で、参照（numpy / torch の f32 逐次計算）と最終ビットまで一致するのが
 * recipe README の実測記録そのもの。ここを緩めると、演算順の取り違え（std の逆数で割る →
 * 掛け算、f32 化の位置の移動）が「GPU の誤差」として後段の tolerance に吸われる。
 *
 * NOTE（turbo 変種で CFG 合成のケースが無い理由）: turbo の参照は guidance_scale=1.0 で採って
 * おり、`pipeline_ref.py` は uncond 側の DiT を**呼ばない**。そのため `noise_uncond_stepNNNN`
 * キー自体が存在せず、`uncond + scale·(cond − uncond)` の枝は突き合わせる相手が無い。turbo
 * 変種が固定するのは CFG=1 の枝（cond をそのまま Euler に渡す）で、CFG 合成の枝は他 3 変種
 * （guidance_scale=4）が固定する。キーの有無は `needsUncond(guidance_scale)` と突き合わせる —
 * 「guidance≠1 なのに uncond が無い」フィクスチャは黙って CFG=1 の枝へ流さず落とす。
 *
 * 資産はリポジトリ管理外（`outputs/` は git 追跡外）なので、無い変種は理由を出して**明示
 * SKIP** する（ADR 0005）。
 */

import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, type Tensor } from "@karume/runtime";
import { cfgEulerStep, sigmaSchedule } from "../src/anima/sampler.ts";
import { animaLatents, denormalizeLatents, padSequence } from "../src/anima/latents.ts";
import { needsUncond } from "../src/generation/dpm-solver-multistep.ts";
import { readFileIfPresent, readTextIfPresent } from "./helpers/read-if-present.ts";

/** フィクスチャ 4 変種（ディレクトリ名 = `pipeline_ref.py` の既定出力 / turbo の `--out`）。 */
const VARIANTS = [
  "anima-pipeline",
  "anima-pipeline-f16",
  "anima-pipeline-i8",
  "anima-pipeline-turbo-f16",
] as const;

const SERIES_ROOT = new URL("../../../outputs/series/", import.meta.url);

/** SKIP 時に貼れる生成コマンド（recipe README と同じ）。 */
const GENERATE_HINT =
  "cd tools/export-recipes && uv run --group anima python -m anima.pipeline_ref" +
  "（--dtype f16 / i8・turbo は README の --guidance-scale 1.0 --lora の形）";

/** `pipeline.json` のうち、ここが読むフィールド。 */
type FixtureMeta = {
  readonly steps: number;
  readonly ref_steps: number;
  readonly shift: number;
  readonly guidance_scale: number;
  readonly min_sequence_length: number;
};

type Fixture = {
  readonly meta: FixtureMeta;
  readonly f32: (name: string) => { data: Float32Array; shape: readonly number[] };
  readonly has: (name: string) => boolean;
  /** dtype を問わない shape（id 列 = I32 の行数を読むため）。 */
  readonly shape: (name: string) => readonly number[];
};

const openFixture = async (variant: string): Promise<Fixture | undefined> => {
  const dir = new URL(`${variant}/`, SERIES_ROOT);
  const [metaText, bytes] = await Promise.all([
    readTextIfPresent(new URL("pipeline.json", dir)),
    readFileIfPresent(new URL("pipeline.safetensors", dir)),
  ]);
  if (metaText === undefined || bytes === undefined) return undefined;
  const meta = JSON.parse(metaText) as FixtureMeta;
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    meta,
    has: (name) => file.tensors.has(name),
    shape: (name) => {
      const tensor = file.tensors.get(name);
      assert(tensor !== undefined, `${variant}: フィクスチャに '${name}' が無い`);
      return tensor.shape;
    },
    f32: (name) => {
      const tensor = file.tensors.get(name);
      assert(tensor !== undefined, `${variant}: フィクスチャに '${name}' が無い`);
      assertEquals(tensor.dtype, "F32", `${variant}: '${name}' の dtype`);
      return {
        data: new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4),
        shape: tensor.shape,
      };
    },
  };
};

/**
 * ビット列の一致を見る（値の `===` ではなく）。`-0` と `+0`・NaN のペイロードも区別する —
 * ゼロ詰めの符号や、非有限値が紛れた場合も「一致」と数えない。
 */
const assertBitIdentical = (actual: Float32Array, expected: Float32Array, label: string): void => {
  assertEquals(actual.length, expected.length, `${label}: 要素数`);
  const got = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const want = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  let mismatches = 0;
  let first = -1;
  for (let index = 0; index < got.length; index += 1) {
    if (got[index] !== want[index]) {
      mismatches += 1;
      if (first < 0) first = index;
    }
  }
  assert(
    mismatches === 0,
    `${label}: ${mismatches}/${got.length} 要素がビット不一致（先頭 index ${first}: ` +
      `${actual[first]} vs 参照 ${expected[first]}）`,
  );
};

const stepTag = (step: number): string => `step${String(step).padStart(4, "0")}`;

for (const variant of VARIANTS) {
  const fixture = await openFixture(variant);
  if (fixture === undefined) {
    console.warn(
      `[karume] outputs/series/${variant}/pipeline.{json,safetensors} が無いため Anima ホスト糊の` +
        `パリティを SKIP する（参照フィクスチャはリポジトリ管理外）。生成: ${GENERATE_HINT}`,
    );
  }
  const ignore = fixture === undefined;

  Deno.test({
    name: `${variant}: sigmaSchedule が参照の sigma 列（終端 0 込み）とビット一致する`,
    ignore,
    fn: () => {
      assert(fixture !== undefined);
      const { steps, shift } = fixture.meta;
      assertBitIdentical(sigmaSchedule(steps, shift), fixture.f32("sigmas").data, "sigmas");
    },
  });

  Deno.test({
    name: `${variant}: cfgEulerStep を ref_steps 回つなぐと参照の latent 列とビット一致する`,
    ignore,
    fn: () => {
      assert(fixture !== undefined);
      const { ref_steps: refSteps, guidance_scale: guidance } = fixture.meta;
      const sigmas = fixture.f32("sigmas").data;
      // uncond キーの有無は guidance から決まる（NOTE 参照）。食い違えばフィクスチャか
      // 参照側の分岐が壊れているので、ケースを飛ばさず落とす。
      assertEquals(
        fixture.has(`noise_uncond_${stepTag(1)}`),
        needsUncond(guidance),
        `guidance_scale ${guidance} と noise_uncond の有無が食い違う`,
      );
      // 各 step は参照側の前 step の latent から始める（TS の出力を持ち回らない）。持ち回ると
      // 1 step 目のずれが 2 step 目以降へ伝播し、どの step で割れたのかが読めなくなる。
      let previous = fixture.f32("latents_init").data;
      for (let step = 1; step <= refSteps; step += 1) {
        const tag = stepTag(step);
        const uncond = needsUncond(guidance) ? fixture.f32(`noise_uncond_${tag}`).data : undefined;
        // Δσ の綴りは denoise ループ（`pipeline.ts` の `denoiseStep`）と同じ。
        const next = cfgEulerStep(
          previous,
          fixture.f32(`noise_cond_${tag}`).data,
          uncond,
          Math.fround(sigmas[step] - sigmas[step - 1]),
          guidance,
        );
        const expected = fixture.f32(`latents_${tag}`).data;
        assertBitIdentical(next, expected, `latents_${tag}`);
        previous = expected;
      }
    },
  });

  Deno.test({
    name: `${variant}: denormalizeLatents が参照の逆正規化 latent とビット一致する`,
    ignore,
    fn: () => {
      assert(fixture !== undefined);
      // 本番は VAE config の写し（animaLatents）を使うので、ここでもそれを渡す。写しが
      // この変種の参照と同じ数であることを先に押さえる（変種ごとに VAE を読み直して採った値）。
      const { mean, std } = animaLatents();
      assertBitIdentical(mean, fixture.f32("latents_mean").data, "latents_mean");
      assertBitIdentical(std, fixture.f32("latents_std").data, "latents_std");
      // 逆正規化の入力は最後に参照を採った step の latent（`pipeline_ref.py` の decode 段）。
      const last = fixture.f32(`latents_${stepTag(fixture.meta.ref_steps)}`);
      const expected = fixture.f32("latents_denorm");
      assertEquals(expected.shape, last.shape, "latents_denorm の shape");
      assertBitIdentical(
        denormalizeLatents(last.data, last.shape, mean, std),
        expected.data,
        "latents_denorm",
      );
    },
  });

  Deno.test({
    name: `${variant}: padSequence が参照の 512 行ゼロ詰め（正・ネガティブ）とビット一致する`,
    ignore,
    fn: () => {
      assert(fixture !== undefined);
      const rows = fixture.meta.min_sequence_length;
      for (const prefix of ["", "neg_"]) {
        // IR の conditioner は T5 の id 列と同じ行数 T を出す。参照は上流の conditioner が
        // `min_sequence_length` 行まで詰めた形なので、その先頭 T 行を IR 側の出力の代わりに
        // 渡し、詰めた結果が参照全体（余白のゼロの符号まで）と一致するかを見る。T を参照の
        // id 列から取るので、余白の開始位置の取り違えもここで割れる。
        const targetLength = fixture.shape(`${prefix}t5_input_ids`)[1];
        const padded = fixture.f32(`${prefix}encoder_hidden_states`);
        const [batch, total, width] = padded.shape;
        assertEquals([batch, total], [1, rows], `${prefix}encoder_hidden_states の shape`);
        assert(targetLength < rows, `${prefix}t5 の長さ ${targetLength} に余白が無い`);
        const hidden: Tensor = {
          dtype: "f32",
          shape: [1, targetLength, width],
          data: padded.data.slice(0, targetLength * width),
        };
        assertBitIdentical(
          padSequence(hidden, rows),
          padded.data,
          `${prefix}encoder_hidden_states`,
        );
      }
    },
  });
}
