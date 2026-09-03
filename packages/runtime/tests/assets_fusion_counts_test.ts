/**
 * **実配布グラフに対する融合ヒット数の門**（GPU 不要）。
 *
 * 融合はエクスポータのノード発行順が 1 つ変わるだけで黙って外れる。値は正しいまま
 * （掴めなければ素の列に落ちるので）性能だけが戻り、例外も警告も出ない — ADR 0040 §3 が
 * `lastRunFusions` を常設した理由がここで、**その数字に突合相手を与える**のが本ファイル。
 *
 * `planFusions` は純関数なので、実資産から IR を読んで計画するだけで判定できる（1 dispatch も
 * 出さない）。安全のため safetensors の**ヘッダだけ**を読む — 実体は合計 7GB 級で、
 * IR は `__metadata__.karume_ir` に載っている。
 *
 * MUST: 資産は `models/karume-anima/`（公式 5 変種同居・既定 = anima-turbo-v1.1 —
 * ADR 0087）と `outputs/series/embeddinggemma-300m/` / `gemma4-e2b-decode{,-token}` /
 * `minicpm5-1b-decode` と `models/karume-irodori-v4-small/`（いずれも untracked・ローカル
 * 資産）。無い環境は理由を出して**明示 SKIP** する（テストを消して無音で緑にしない —
 * ADR 0005）。
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

import { assertEquals } from "@std/assert";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import { type FusionCounts, planFusions } from "../src/runtime/fusion.ts";
import { bindSymbols, countUses, planGraph } from "../src/runtime/plan.ts";
import { resolveShards } from "./helpers/shard-files.ts";

const ANIMA_DIR = new URL("../../../models/karume-anima/", import.meta.url);
/**
 * グラフを持つのは配布形の**先頭 shard** だけ（ADR 0081）。分割されていない資産では代表 path
 * そのものが返るので、どちらの形でも同じ 1 行で足りる。
 */
const GEMMA_MODEL = resolveShards(
  new URL("../../../outputs/series/embeddinggemma-300m/model.safetensors", import.meta.url),
)[0];
const IRODORI_DIR = new URL(
  "../../../models/karume-irodori-v4-small/v4-small/",
  import.meta.url,
);

/** safetensors のヘッダ JSON だけを読む（実体は読まない）。 */
const readIrGraph = async (source: URL): Promise<IrGraph> => {
  const file = await Deno.open(source, { read: true });
  try {
    const lengthBytes = new Uint8Array(8);
    await file.read(lengthBytes);
    const length = Number(new DataView(lengthBytes.buffer).getBigUint64(0, true));
    const headerBytes = new Uint8Array(length);
    for (let read = 0; read < length;) {
      const chunk = await file.read(headerBytes.subarray(read));
      if (chunk === null) {
        throw new Error(`${source.pathname}: ヘッダ ${length} バイトを読み切れない`);
      }
      read += chunk;
    }
    const header = JSON.parse(new TextDecoder().decode(headerBytes));
    return parseIrGraph(header.__metadata__.karume_ir);
  } finally {
    file.close();
  }
};

/**
 * 融合ヒット数は**ノード列だけ**で決まる（格納 dtype はどの initializer をどう読むかしか
 * 変えない）ので、系列が増えても f32 の 1 本を見れば足りる — 見ているのはエクスポータの
 * ノード発行順で、それは系列を跨いで同じ 1 回の変換から出る。
 */
const readIrodoriGraph = (name: string): Promise<IrGraph> =>
  readIrGraph(resolveShards(new URL(`${name}/model.f32.safetensors`, IRODORI_DIR))[0]);

/**
 * 資産の有無。
 *
 * MUST: NotFound 以外は伝播させる — 全 I/O エラーを「未生成」に丸めると、資産ルートの
 * マウント異常（EIO 等）が SKIP に化けて、実行されていない検証が静かに緑になる。
 */
const exists = async (url: URL): Promise<boolean> => {
  try {
    await Deno.stat(url);
    return true;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const ASSETS_AVAILABLE = await exists(new URL("karume.json", ANIMA_DIR));
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ANIMA_DIR.pathname} に karume.json が無いため実資産の` +
      "融合ヒット数を SKIP する（エクスポータのノード発行順の退行は実資産でしか検出できない）",
  );
}

/**
 * 越境参照（ADR 0038 §7）の repo → ローカルミラー。公式リポは自己完結（ADR 0087）なので
 * 現在は空 — `karume-anima-extra` のミラー（リリース時に越境で焼く）が生えたら、その門と
 * 一緒にエントリを戻す。読み分けの機構は残す。
 */
const MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * manifest のうちこのファイルが引く欄だけ。**綴りを持つのは配布形**（shard 本数・path・越境の
 * 有無はエクスポータが決める）なので、テスト側に焼かずここから引く — 焼くと分割数が動いた
 * 瞬間に融合ヒット数の門が NotFound で落ちて、退行検出そのものが止まる。
 */
type AnimaManifest = {
  readonly defaultModel: string;
  readonly models: Readonly<
    Record<string, {
      readonly weights: Readonly<
        Record<
          string,
          Readonly<Record<string, { readonly shards: readonly ShardRef[] }>>
        >
      >;
    }>
  >;
};
type ShardRef = { readonly path: string; readonly repo?: string };

// 資産が無い環境では 1 バイトも読まない（この定数を触るのは ignore を抜けたテストだけ）。
const ANIMA_MANIFEST: AnimaManifest | undefined = ASSETS_AVAILABLE
  ? JSON.parse(await Deno.readTextFile(new URL("karume.json", ANIMA_DIR)))
  : undefined;

/**
 * コンポーネントの**グラフ shard**（`karume_ir` を持つ先頭 shard — ADR 0070 決定 3）を読む。
 * 後続の重み shard は metadata を持たないので、融合の計画に要るのはこの 1 本だけ。
 */
const readAnimaGraph = (component: string, dtype: string): Promise<IrGraph> => {
  const manifest = ANIMA_MANIFEST as AnimaManifest;
  const [head] = manifest.models[manifest.defaultModel].weights[component][dtype].shards;
  if (head.repo === undefined) return readIrGraph(new URL(head.path, ANIMA_DIR));
  const mirror = MIRRORS.get(head.repo);
  if (mirror === undefined) {
    throw new Error(
      `${component}/${dtype} の越境先 '${head.repo}' に対応するローカルミラーが無い` +
        `（既知: ${[...MIRRORS.keys()].join(" / ")}）`,
    );
  }
  return readIrGraph(new URL(head.path, mirror));
};

const GEMMA_AVAILABLE = await exists(GEMMA_MODEL);
if (!GEMMA_AVAILABLE) {
  console.warn(
    `[karume] ${GEMMA_MODEL.pathname} が無いため EmbeddingGemma の融合ヒット数を SKIP する`,
  );
}

// 見るのは**ファイル**（ディレクトリではない）— 配布形の綴りが動いたときに、空でない
// ディレクトリだけを見て「資産あり」と判断すると、読めない path で FAIL する形になる。
const IRODORI_AVAILABLE = await exists(
  resolveShards(new URL("dit/model.f32.safetensors", IRODORI_DIR))[0],
);
if (!IRODORI_AVAILABLE) {
  console.warn(
    `[karume] ${IRODORI_DIR.pathname} が無いため Irodori の融合ヒット数を SKIP する`,
  );
}

const fusionCounts = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
  stateShapes?: ReadonlyMap<string, readonly number[]>,
): FusionCounts =>
  planFusions(planGraph(graph, bindSymbols(graph, inputShapes), stateShapes).nodes, {
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    // WebGPU core 既定（128MiB）を判定に使う。行ブロック枚数はヒット数に効かないが、
    // **上限の値を機の実測から取らない**ことでこの門が機に依らない固定であり続ける。
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  }).counts;

/**
 * states 形 decode グラフの融合ヒット数（物理 chunk 行数 M を振って計画する）。
 *
 * スロットの容量記号は 640 で解く — 融合はノード列と**値**の shape で決まり、スロット容量の
 * 値には依存しない（e2e の検収値と同じ 640 を使うのは読み合わせやすさだけ）。入力は全て
 * `[1, M]`（token-only 形の `last_row[1]` は数値次元なのでそのまま）。
 */
const decodeFusionCounts = (graph: IrGraph, rows: number): FusionCounts => {
  const inputShapes = Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : rows)),
    ]),
  );
  const stateShapes = new Map(
    Object.entries(graph.states).map(([name, slot]) => [
      name,
      slot.shape.map((dim) => (typeof dim === "number" ? dim : 640)),
    ]),
  );
  return fusionCounts(graph, inputShapes, stateShapes);
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

const NONE: FusionCounts = {
  silu: 0,
  upsample2x: 0,
  rope: 0,
  adaln: 0,
  rowBlockAttention: 0,
  identityExpand: 0,
};

Deno.test({
  name: "実資産の DiT は run 1 回で adaln 85 / rope 56 / silu 2 を掴む（w8a8 と f16 で同一）",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    const expected: FusionCounts = { ...NONE, silu: 2, rope: 56, adaln: 85 };
    for (const quant of ["i8", "f16"] as const) {
      const graph = await readAnimaGraph("transformer", quant);
      // 融合は f32 の計算経路だけを見るので、重みの格納形が変わってもヒット数は動かない。
      assertEquals(fusionCounts(graph, ditShapes(4096)), expected, `${quant}: 1024px（S=4096）`);
      assertEquals(fusionCounts(graph, ditShapes(1024)), expected, `${quant}: 512px（S=1024）`);
    }
  },
});

Deno.test({
  name: "実資産の text encoder / conditioner / VAE decoder の融合ヒット数",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    const textEncoder = await readAnimaGraph("text_encoder", "f16");
    assertEquals(
      fusionCounts(textEncoder, { input_ids: [1, 64] }),
      { ...NONE, silu: 28, rope: 56, identityExpand: 112 },
      "text encoder",
    );
    const conditioner = await readAnimaGraph("text_conditioner", "f16");
    assertEquals(
      fusionCounts(conditioner, {
        source_hidden_states: [1, 64, 1024],
        target_input_ids: [1, 512],
      }),
      { ...NONE, identityExpand: 48 },
      "conditioner",
    );
    const vae = await readAnimaGraph("vae_decoder", "f16");
    assertEquals(
      fusionCounts(vae, { latents: [1, 16, 64, 64] }),
      { ...NONE, silu: 29, upsample2x: 3 },
      "VAE decoder（タイル 1 枚ぶん）",
    );
  },
});

/**
 * EmbeddingGemma-300m（24 層 × q / k の 2 本 = 48 鎖）。Anima と違い head 幅は 256 で、
 * cos / sin 表は θ 系統（local 1e4 / global 1e6）ごとに `sym_prefix_slice` で T へ縮められる。
 * その初出 2 箇所は鎖の隙間に落ちるので、窓内 passthrough を跨げないと 48 のうち 2 本が
 * 黙って外れる — **ヒット数 48 はそこまで含めた門**。
 *
 * identityExpand は **0**。SDPA を保存した資産（ADR 0023 改訂 — mask 付き attention）では
 * 分解由来の恒等 expand が IR ごと消える（決定 6 と同じ機序。分解資産の頃は 96 だった）。
 */
Deno.test({
  name: "実資産の EmbeddingGemma は rope 48（head 幅 256・窓内 passthrough 込み）を掴む",
  ignore: !GEMMA_AVAILABLE,
  fn: async () => {
    const graph = await readIrGraph(GEMMA_MODEL);
    const expected: FusionCounts = { ...NONE, rope: 48 };
    // ヒット数は T に依存しない（Tmax = 512 の内側で 2 点）。
    for (const sequence of [12, 318]) {
      assertEquals(
        fusionCounts(graph, { input_ids: [1, sequence], pool_mask: [1, sequence] }),
        expected,
        `T=${sequence}`,
      );
    }
  },
});

const GEMMA4_DECODE_MODEL = resolveShards(
  new URL("../../../outputs/series/gemma4-e2b-decode/model.safetensors", import.meta.url),
)[0];
const GEMMA4_TOKEN_MODEL = resolveShards(
  new URL("../../../outputs/series/gemma4-e2b-decode-token/model.safetensors", import.meta.url),
)[0];
const MINICPM5_DECODE_MODEL = resolveShards(
  new URL("../../../outputs/series/minicpm5-1b-decode/model.safetensors", import.meta.url),
)[0];

const GEMMA4_DECODE_AVAILABLE = await exists(GEMMA4_DECODE_MODEL) &&
  await exists(GEMMA4_TOKEN_MODEL);
if (!GEMMA4_DECODE_AVAILABLE) {
  console.warn(
    `[karume] ${GEMMA4_DECODE_MODEL.pathname} と ${GEMMA4_TOKEN_MODEL.pathname} が揃っていない` +
      "ため Gemma 4 decode の融合ヒット数を SKIP する",
  );
}

const MINICPM5_DECODE_AVAILABLE = await exists(MINICPM5_DECODE_MODEL);
if (!MINICPM5_DECODE_AVAILABLE) {
  console.warn(
    `[karume] ${MINICPM5_DECODE_MODEL.pathname} が無いため MiniCPM5 decode の融合ヒット数を` +
      " SKIP する",
  );
}

/**
 * Gemma 4 E2B decode（states 形・35 層）。**rope はヒット 15 本・prefill 形（M=32）は 0 本**で、
 * これは全 50 鎖（q 側 35 + k 側 15 所有層）が掴めている状態ではない — op 名列は 50 箇所とも
 * matcher の窓（`mul,slice,slice,neg,cat,mul,add`）に並ぶが、計画では大半が外れる。同じ
 * 表引き RoPE の MiniCPM5 が M 非依存で全鎖適合する（下のテスト）ので、gemma4 固有の発行形が
 * 原因と見られる — 機序は未特定（実測の記録 =
 * docs/research/2026-08-30-gemma4-decode-wallclock.md §4。GPU 時間への寄与は 1ms 級なので
 * 追跡は性能実需待ち）。
 *
 * この門が守るのは他と同じ 2 方向: **掴めている 15 本が黙って外れない**こと（exporter の
 * 発行順退行 — dispatch が値の正しいまま +200 本級に増える）と、**数字が動いたら受理集合か
 * 発行形が変わった**と気づけること（増える側は改善 — その時この期待値を更新する）。
 */
Deno.test({
  name: "実資産の Gemma 4 E2B decode は M=1 で rope 15 を掴む（token-only 形も同一・M=32 は 0）",
  ignore: !GEMMA4_DECODE_AVAILABLE,
  fn: async () => {
    for (
      const [name, source] of [
        ["logits opt-in", GEMMA4_DECODE_MODEL],
        ["token-only", GEMMA4_TOKEN_MODEL],
      ] as const
    ) {
      const graph = await readIrGraph(source);
      assertEquals(decodeFusionCounts(graph, 1), { ...NONE, rope: 15 }, `${name} decode（M=1）`);
      assertEquals(decodeFusionCounts(graph, 32), NONE, `${name} prefill 形（M=32）`);
    }
  },
});

/**
 * MiniCPM5-1B decode（24 層 × q/k = 48 鎖・KV 共有なし）。gemma4 と違い**全鎖が M 非依存で
 * 適合する** — 同じ表引き RoPE でもここまで割れる、という gemma4 側の対照実証でもある
 * （片方だけ動いたら、どちらの発行形が変わったのかがこの対で割れる）。silu 24 は 24 層の
 * gate MLP。
 */
Deno.test({
  name: "実資産の MiniCPM5 decode は rope 48 / silu 24 を掴む（M 非依存）",
  ignore: !MINICPM5_DECODE_AVAILABLE,
  fn: async () => {
    const graph = await readIrGraph(MINICPM5_DECODE_MODEL);
    const expected: FusionCounts = { ...NONE, rope: 48, silu: 24 };
    for (const rows of [1, 32]) {
      assertEquals(decodeFusionCounts(graph, rows), expected, `M=${rows}`);
    }
  },
});

/** DiT の入力 shape（S = latent フレーム数。mask は `S+1519` の派生次元）。 */
const irodoriDitShapes = (sequence: number): Readonly<Record<string, readonly number[]>> => ({
  x_t: [1, sequence, 32],
  t_embed: [1, 512],
  mask: [1, 1, 1, sequence + 1519],
  text_state: [1, 256, 512],
  speaker_state: [1, 751, 768],
  caption_state: [1, 512, 512],
});

/**
 * Irodori DiT（12 ブロック）。**rope も adaln も 0** で、これは退行ではなく matcher の受理集合が
 * この綴りを含まないことの記録:
 *
 * - rope 0: DiT の RoPE は**偶奇形**（`[1,S,H·D/2,2]` へ view して最終軸を slice / neg / cat し
 *   reshape で戻す）。`runtime/fusion.ts` の ROPE_RULE は `[1,H,S,D]` の半割り形だけを受理し、
 *   偶奇形は「式が似ている」で広げない MUST を掲げている（掴めなければ素の列で値は正しい）。
 * - adaln 0: ADALN_RULE の先頭 op は `layer_norm` だが、DiT の正規化は `rms_norm` 87 本で
 *   `layer_norm` は 1 本も無い。
 * - silu 17: 前段の条件 MLP 5 本 + 12 ブロック × 1 本。残る sigmoid 12 本は
 *   `mul(v, sigmoid(u))` のゲート（自分自身に掛からないので SiLU ではない）。
 * - **rowBlockAttention 12**: 12 ブロックの分解 attention（`bmm → reshape → add(mask) →
 *   safe_softmax → expand → reshape → expand → reshape → bmm`）を全て掴む。この綴りを出すのは
 *   実資産では DiT だけで、他の 7 パートは `attention` op（ADR 0023）を持つ。
 * - identityExpand 24: 12 ブロック × 4 本のうち、窓の内側の 2 本（P 側と V 側）は融合ステップに
 *   飲まれるので別名化として数えられなくなる。**48 → 24 は退行ではなくこの移動**で、
 *   `rowBlockAttention` が 0 に戻れば 48 に戻る（片方だけ動いたら受理集合の事故）。
 *
 * したがってこの門が守るのは「掴めている 3 種が外れないこと」と「**受理集合を広げたとき
 * ここが動く**こと」の両方。0 が非 0 に変わったら、まず ROPE_RULE / ADALN_RULE の受理集合が
 * 意図せず広がっていないかを見る（広げた瞬間 fallback の正しさ保証が消える）。
 */
Deno.test({
  name:
    "実資産の Irodori DiT は run 1 回で rowBlockAttention 12 / silu 17 / identityExpand 24 を掴む（rope / adaln は綴りが違って 0）",
  ignore: !IRODORI_AVAILABLE,
  fn: async () => {
    const graph = await readIrodoriGraph("dit");
    const expected: FusionCounts = {
      ...NONE,
      silu: 17,
      rowBlockAttention: 12,
      identityExpand: 24,
    };
    // ヒット数は S に依存しない（`ditSymMax` 750 の内側で 2 点）。**行ブロック枚数は
    // S で変わる**（128MiB 上限では S=125 が 1 枚・S=750 が 2 枚）が、ヒット数は動かない。
    for (const sequence of [125, 750]) {
      assertEquals(fusionCounts(graph, irodoriDitShapes(sequence)), expected, `S=${sequence}`);
    }
  },
});

Deno.test({
  name: "実資産の Irodori 条件経路と codec の融合ヒット数",
  ignore: !IRODORI_AVAILABLE,
  fn: async () => {
    // backbone（ModernBERT-ja 25 層）だけは半割り形の RoPE なので、25 層 × q/k = 50 を掴む。
    // Irodori 自前のブロック（dit / speaker）が 0 なのと対になる観測点で、**片方だけ外れたら
    // どちらの綴りが変わったのか**がここで割れる。
    assertEquals(
      fusionCounts(await readIrodoriGraph("backbone"), { input_ids: [1, 256] }),
      { ...NONE, rope: 50 },
      "backbone",
    );
    assertEquals(
      fusionCounts(await readIrodoriGraph("text_proj"), { hidden: [1, 256, 768] }),
      { ...NONE, silu: 1 },
      "text_proj",
    );
    assertEquals(
      fusionCounts(await readIrodoriGraph("caption_proj"), { hidden: [1, 512, 768] }),
      { ...NONE, silu: 1 },
      "caption_proj",
    );
    // speaker は 8 ブロック（silu 8 + ゲート 8）。RoPE は DiT と同じ偶奇形なので 0。
    assertEquals(
      fusionCounts(await readIrodoriGraph("speaker"), { latent: [1, 750, 128] }),
      { ...NONE, silu: 8 },
      "speaker",
    );
    assertEquals(
      fusionCounts(await readIrodoriGraph("duration"), {
        text_state: [1, 256, 512],
        speaker_vec: [1, 768],
        has_speaker: [1, 1],
        caption_vec: [1, 512],
        has_caption: [1, 1],
      }),
      { ...NONE, silu: 5 },
      "duration",
    );
    // codec は Snake 活性（`sin` 29 本 — sigmoid ではない）と `conv_transpose1d` 4 本で、
    // silu / upsample2x のどちらの形も持たない。全 0 は「掴む形が無い」の記録で、非 0 へ
    // 変わったら受理集合が広がったということ。
    for (
      const [name, shapes] of [
        ["codec_decoder", { latent: [1, 750, 32] }],
        ["codec_encoder", { wav: [1, 750, 1920] }],
      ] as const
    ) {
      assertEquals(fusionCounts(await readIrodoriGraph(name), shapes), NONE, name);
    }
  },
});
