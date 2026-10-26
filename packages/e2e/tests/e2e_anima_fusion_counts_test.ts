/**
 * Fusion hit counts of the Anima distribution graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the assets are `models/karume-anima/` (official repo, 5 variants, default =
 * anima-turbo-v1.1 — ADR 0087) and `models/karume-anima-extra/` (extra variants whose shared
 * components are cross-repo references into the official repo). Both are untracked local assets;
 * where they are missing the cases SKIP explicitly with a reason (ADR 0005).
 *
 * NOTE: ここで固定するのは **run 1 回あたり**の値（`lastRunFusions` と同じ寿命）。ADR 0040 の
 * 実測欄が載せている predict 1 回ぶんの合計は、これらをパイプラインの run 回数で畳んだもの:
 * rope 56×8step + 56 = **504** / silu 2×8 + 28 + 29×9tile = **305** /
 * upsample2x 3×9tile = **27** / identityExpand 112 + 48 = **160** / adaln 85×8step = **680**
 * （ADR の「adaln 85」は静的な鎖の本数 = 本ファイルの DiT 1 run ぶん）。
 * VAE の 9 タイルは 1024px（latent 128・タイル 64・最小重なり 8）で 1 軸 3 枚 × 2 軸。
 * NOTE: text encoder の rope は 56（ADR 0040 執筆時点の記録は 55）。窓内 passthrough を
 * 掴めるようになって、sin 表の `sym_prefix_slice` が鎖の隙間に落ちる**初出 1 箇所**が
 * 拾えるようになったぶん。
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import {
  type ContainerManifest,
  containerPart0,
} from "../../runtime/tests/helpers/container-graph.ts";
import { exists, fusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";

const ANIMA_DIR = new URL("../../../models/karume-anima/", import.meta.url);

const ASSETS_AVAILABLE = await exists(new URL("karume.json", ANIMA_DIR));
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ANIMA_DIR.pathname} に karume.json が無いため実資産の` +
      "融合ヒット数を SKIP する（エクスポータのノード発行順の退行は実資産でしか検出できない）",
  );
}

/**
 * 追加変種の配布形（`models/karume-anima-extra/` — anima-wai-v1.0 / anima-copycat-20260610）。
 * text_encoder / vae_decoder を公式リポへの**越境参照**（ADR 0038 §7）で焼くので、公式の
 * ミラーが揃って初めて読める。
 */
const EXTRA_DIR = new URL("../../../models/karume-anima-extra/", import.meta.url);

/**
 * 越境参照（ADR 0038 §7）の repo → ローカルミラー。公式リポは自己完結（ADR 0087）で、
 * 越境で読むのは extra の共有部品だけ。
 */
const MIRRORS: ReadonlyMap<string, URL> = new Map([["hdae/karume-anima", ANIMA_DIR]]);

const EXTRA_AVAILABLE = ASSETS_AVAILABLE && (await exists(new URL("karume.json", EXTRA_DIR)));
if (!EXTRA_AVAILABLE) {
  console.warn(
    `[karume] ${EXTRA_DIR.pathname} に karume.json が無い（または越境先の ${ANIMA_DIR.pathname} が` +
      "無い）ため追加変種の融合ヒット数を SKIP する（release-runbook の越境参照の手順で焼く）",
  );
}

// 資産が無い環境では 1 バイトも読まない（この定数を触るのは ignore を抜けたテストだけ）。
// **綴りを持つのは配布形**（part 本数・path・越境の有無はエクスポータが決める）なので、テスト側に
// 焼かず manifest から引く — 焼くと分割数が動いた瞬間に融合ヒット数の門が NotFound で落ちて、
// 退行検出そのものが止まる。
const ANIMA_MANIFEST: ContainerManifest | undefined = ASSETS_AVAILABLE
  ? JSON.parse(await readTextFile(new URL("karume.json", ANIMA_DIR)))
  : undefined;
const EXTRA_MANIFEST: ContainerManifest | undefined = EXTRA_AVAILABLE
  ? JSON.parse(await readTextFile(new URL("karume.json", EXTRA_DIR)))
  : undefined;

/** 公式リポの既定モデル（turbo）の部品のグラフ。 */
const readAnimaGraph = (component: string, dtype: string): Promise<IrGraph> => {
  const manifest = ANIMA_MANIFEST as ContainerManifest;
  return readDistributionGraph(
    ANIMA_DIR,
    manifest,
    manifest.defaultModel,
    component,
    dtype,
    MIRRORS,
  );
};

/**
 * DiT の入力 shape（S = パッチトークン数。1024px = 64×64 = 4096）。ヒット数は S に依存しない
 * ことを下のテストが 2 点で確かめる。
 */
const ditShapes = (sequence: number): Readonly<Record<string, readonly number[]>> => ({
  tokens: [1, sequence, 68],
  timesteps_proj: [1, 2048],
  encoder_hidden_states: [1, 512, 1024],
  rope_cos: [1, 1, sequence, 128],
  rope_sin: [1, 1, sequence, 128],
});

/**
 * The DiT's counts under `linearCompute: "a8"` (the `f16+dit8-a8*` presets; counted 2026-10-25):
 * the f32 counts plus **linearPrologue 28 on the i8 seat** — the erf gelu folded into the step ① of
 * the i8 linear that reads it (perf-ledger F-4), the FFN down projection of the 28 blocks (gelu
 * [S, 8192] → linear [2048, 8192]). The f16 seat has no i8a8 linear, so it stays at 0 under a8.
 */
const ditA8 = (quant: "i8" | "f16"): FusionCounts => ({
  ...NONE,
  silu: 2,
  rope: 56,
  adaln: 85,
  linearPrologue: quant === "i8" ? 28 : 0,
});

const a8Counts = (graph: IrGraph, shapes: Readonly<Record<string, readonly number[]>>) =>
  fusionCounts(graph, shapes, undefined, false, false, false, "a8");

/**
 * The DiT's counts on the a8 attention seat (the `f16+dit8-a8-attn8*` presets — linearCompute and
 * attentionCompute "a8"; counted 2026-10-25): the a8 counts plus **attentionPermute 56** — each of
 * the 56 attentions (28 self, 28 cross) absorbs the `[0, 2, 1, 3]` permute of v right before it
 * (perf-ledger F-3 A4 / F-8). Its q / k reach it through the per-head rms_norm and the RoPE (self) or
 * the per-head rms_norm alone (cross), so their permutes are not adjacent. On these seats, which store
 * S, the permute of O after it stays a node.
 */
const ditAttn8 = (quant: "i8" | "f16"): FusionCounts => ({ ...ditA8(quant), attentionPermute: 56 });

const attn8Counts = (graph: IrGraph, shapes: Readonly<Record<string, readonly number[]>>) =>
  fusionCounts(graph, shapes, undefined, false, false, false, "a8", "a8");

/**
 * Device facts that hold the online kernel at the DiT's D 128 (subgroups and 48 KiB of workgroup
 * storage — the sg32 form needs 31,232 B).
 */
const ONLINE_DEVICE = { subgroups: true, maxComputeWorkgroupStorageSize: 48 * 1024 } as const;

/**
 * The DiT's counts on a user-chosen online seat (attentionCompute "a8" × attentionScoreStorage
 * "none" on a device that holds the online kernel; counted 2026-10-26): **attentionPermute 112**.
 * Every attention takes the online route (D 128; the key count — S for self-attention, the 512 text
 * tokens for cross-attention — is a multiple of 4 at every S checked here), so each of the 56 also
 * absorbs the permute of O right after it (perf-ledger F-3 A4).
 */
const ditOnline = (quant: "i8" | "f16"): FusionCounts => ({
  ...ditA8(quant),
  attentionPermute: 112,
});

const onlineCounts = (graph: IrGraph, shapes: Readonly<Record<string, readonly number[]>>) =>
  fusionCounts(graph, shapes, undefined, false, false, false, "a8", "a8", "none", ONLINE_DEVICE);

test.skipIf(!ASSETS_AVAILABLE)(
  "the real DiT grabs adaln 85 / rope 56 / silu 2 per run (the same on w8a8 and f16; under a8, linearPrologue 28 on the i8 seat only; attentionPermute 56 on the a8 attention seat that stores S, 112 on the online seat)",
  async () => {
    const expected: FusionCounts = { ...NONE, silu: 2, rope: 56, adaln: 85 };
    for (const quant of ["i8", "f16"] as const) {
      const graph = await readAnimaGraph("transformer", quant);
      // 融合は f32 の計算経路だけを見るので、重みの格納形が変わってもヒット数は動かない。
      assertEquals(fusionCounts(graph, ditShapes(4096)), expected, `${quant}: 1024px（S=4096）`);
      assertEquals(fusionCounts(graph, ditShapes(1024)), expected, `${quant}: 512px（S=1024）`);
      for (const sequence of [4096, 1024]) {
        assertEquals(
          a8Counts(graph, ditShapes(sequence)),
          ditA8(quant),
          `${quant} a8: S=${sequence}`,
        );
        assertEquals(
          attn8Counts(graph, ditShapes(sequence)),
          ditAttn8(quant),
          `${quant} a8 attention: S=${sequence}`,
        );
        assertEquals(
          onlineCounts(graph, ditShapes(sequence)),
          ditOnline(quant),
          `${quant} a8 attention, online: S=${sequence}`,
        );
      }
    }
  },
);

test.skipIf(!ASSETS_AVAILABLE)(
  "実資産の text encoder / conditioner / VAE decoder の融合ヒット数",
  async () => {
    const textEncoder = await readAnimaGraph("text_encoder", "f16");
    assertEquals(
      fusionCounts(textEncoder, { input_ids: [1, 64] }),
      { ...NONE, silu: 28, rope: 56, identityExpand: 112 },
      "text encoder",
    );
    const conditionerShapes = {
      source_hidden_states: [1, 64, 1024],
      target_input_ids: [1, 512],
    };
    const conditioner = await readAnimaGraph("text_conditioner", "f16");
    assertEquals(
      fusionCounts(conditioner, conditionerShapes),
      { ...NONE, rope: 24, identityExpand: 48 },
      "conditioner（BSHD の q/k 各12鎖も融合）",
    );
    // Its 6 erf gelu → linear pairs do not fold even under a8: the conditioner's weights are f16 /
    // bf16 on every preset (no i8a8 linear to absorb them; the quant's session reaches only the DiT).
    for (const quant of ["f16", "bf16"] as const) {
      assertEquals(
        a8Counts(await readAnimaGraph("text_conditioner", quant), conditionerShapes),
        { ...NONE, rope: 24, identityExpand: 48 },
        `conditioner ${quant} a8: no linearPrologue`,
      );
    }
    const vae = await readAnimaGraph("vae_decoder", "f16");
    assertEquals(
      fusionCounts(vae, { latents: [1, 16, 64, 64] }),
      { ...NONE, silu: 29, upsample2x: 3 },
      "VAE decoder（タイル 1 枚ぶん）",
    );
  },
);

/**
 * 追加変種（extra）の融合ヒット数。値は 2026-09-24 に extra（karume/0.13.0・越境先
 * hdae/karume-anima@adb9dcf0）で実測して凍結したもの — 公式と同じアーキテクチャの fine-tune
 * なので結果的に公式の値と一致するが、**この配布形のグラフを実際に計画した値**として持つ
 * （extra は変種ごとに transformer / text_conditioner を自前で焼くので、発行順の退行はここにも
 * 独立に出うる）。
 *
 * text_encoder / vae_decoder は公式リポへの越境参照なので、この門は {@link MIRRORS} の
 * 読み分けを実資産で通す門を兼ねる。
 */
const EXTRA_MODELS = ["anima-copycat-20260610", "anima-wai-v1.0"] as const;
const EXTRA_CROSS_REPO_COMPONENTS = ["text_encoder", "vae_decoder"] as const;

test.skipIf(!EXTRA_AVAILABLE)(
  "実資産の extra は 2 変種とも manifest の変種表と一致し、共有部品を越境参照で持つ",
  () => {
    const manifest = EXTRA_MANIFEST as ContainerManifest;
    // 変種が増えた日に、融合の門を通らない変種が黙って配られないように表ごと固定する。
    assertEquals(Object.keys(manifest.models).sort(), [...EXTRA_MODELS]);
    for (const model of EXTRA_MODELS) {
      // extra が自己完結へ戻った日に、この門が黙って「越境を通らない門」に化けないように。
      const crossed = ["text_encoder", "text_conditioner", "transformer", "vae_decoder"].filter(
        (component) =>
          Object.keys(manifest.models[model].weights[component]).some(
            (dtype) => containerPart0(manifest, model, component, dtype).repo !== undefined,
          ),
      );
      assertEquals(crossed, [...EXTRA_CROSS_REPO_COMPONENTS], `${model}: 越境参照で持つ部品`);
    }
  },
);

for (const model of EXTRA_MODELS) {
  test.skipIf(!EXTRA_AVAILABLE)(
    `実資産の extra ${model} の融合ヒット数（越境先の共有部品を含む）`,
    async () => {
      const manifest = EXTRA_MANIFEST as ContainerManifest;
      const read = (component: string, dtype: string): Promise<IrGraph> =>
        readDistributionGraph(EXTRA_DIR, manifest, model, component, dtype, MIRRORS);
      const dit: FusionCounts = { ...NONE, silu: 2, rope: 56, adaln: 85 };
      for (const quant of ["i8", "f16"] as const) {
        const graph = await read("transformer", quant);
        assertEquals(fusionCounts(graph, ditShapes(4096)), dit, `${quant}: 1024px（S=4096）`);
        assertEquals(fusionCounts(graph, ditShapes(1024)), dit, `${quant}: 512px（S=1024）`);
        assertEquals(a8Counts(graph, ditShapes(4096)), ditA8(quant), `${quant} a8: S=4096`);
        assertEquals(
          attn8Counts(graph, ditShapes(4096)),
          ditAttn8(quant),
          `${quant} a8 attention: S=4096`,
        );
        assertEquals(
          onlineCounts(graph, ditShapes(4096)),
          ditOnline(quant),
          `${quant} a8 attention, online: S=4096`,
        );
      }
      assertEquals(
        fusionCounts(await read("text_encoder", "f16"), { input_ids: [1, 64] }),
        { ...NONE, silu: 28, rope: 56, identityExpand: 112 },
        "text encoder（越境）",
      );
      assertEquals(
        fusionCounts(await read("text_conditioner", "f16"), {
          source_hidden_states: [1, 64, 1024],
          target_input_ids: [1, 512],
        }),
        { ...NONE, rope: 24, identityExpand: 48 },
        "conditioner",
      );
      assertEquals(
        fusionCounts(await read("vae_decoder", "f16"), { latents: [1, 16, 64, 64] }),
        { ...NONE, silu: 29, upsample2x: 3 },
        "VAE decoder（越境・タイル 1 枚ぶん）",
      );
    },
  );
}
