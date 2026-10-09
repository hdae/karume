/**
 * Wan2.2 TI2V-5B の VAE **encoder の chunk 0**（I2V の条件画像 1 枚）の参照照合（実 GPU — ADR 0121 段 9a の検収）。
 *
 * 系列 `outputs/series/wan2.2-ti2v-5b-f16-dyn/` の encoder の 3 グラフ（f16 席 — recipe
 * `tools/export-recipes/wan/export_vae_encoder.py`）を、golden の RGB8 から TS の前処理（`[-1, 1]` → patchify）で作った
 * 入力で回し、出口の mu を TS の正規化（`normalizeWanLatents`）に通して、上流の非タイル encode（diffusers・CPU f32・
 * 重みは f16 へ丸めた値）の mu の正規化と比べる。
 *
 * - pre `[12,1,8h,8w]` → `[640,1,h,w]`（記号 h, w）/ attn `[640,S]` → `[640,S]`（記号 S = h·w）/ post `[640,1,h,w]` →
 *   mu `[48,1,h,w]`。h, w は潜在の高さ・幅。IR の次元式は記号 1 つの一次式なので、h·w を 1 軸に作る mid の
 *   attention の前後でグラフを分けてある（ADR 0121 決定 11）。
 * - golden の encoder の入力（上流の `patchify`）と TS の前処理の出力がビット一致することも見る（GPU に渡すのは
 *   TS の前処理の出力 — 製品の経路と同じ入力）。
 *
 * ## 受け渡し（常駐テンソル・ホストを経由しない）
 *
 * 3 本の Session は同じ GpuContext に張り、1 本の batch の中で pre → attn → post を `enqueue` する。pre / attn の
 * 出口は `copyOutputs` で常駐テンソルへ写し、次のグラフの入力にその常駐をそのまま束ねる。`[640,1,h,w]` と
 * `[640,S]` は同じバイト列の読み替えなので写し直しは要らない（常駐は大きさしか持たない — 検査は宣言 shape ぶんの
 * バイト数の一致だけ）。常駐の入力は記号の束縛源にならないので、attn には `S`、post には `h` / `w` を `bindings`
 * で渡す。mu も常駐へ写し、`finishAndRead` で 1 度だけ読む（フェンスは batch の決着 1 本）。
 *
 * ## ケースと帯
 *
 * - golden はテスト画像 3 枚 × 3 寸法（1280×704・704×1280〈横長の画像を明示した縦長〉・256×160〈受理集合の外の
 *   縮小の経路〉）。帯の決定用は boxing-cats / cat-dog-baking、受入れは ferret（recipe の `BAND_IMAGES` /
 *   `ACCEPT_IMAGES` — golden のメタ `role` と突き合わせる）。
 * - **9 本とも既定のレーンで回す**（2026-10-09 の実測で全 9 本の照合は約 3 s — 1280×704 の 2 本だけより約 1 s
 *   多いだけ）。同じ Session を 1280×704 → 704×1280（S は同じで h と w が入れ替わる）→ 256×160 の順に回し、
 *   束縛ごとに golden と突き合わせるので、**資産が寸法に依らないこと**（同じ容器・同じ Session・束縛だけが違う）と、
 *   束縛が変わったときに計画が前の束縛の形を持ち越さないこと（pre の reshape〈記号の割り算の space-to-depth〉・
 *   slice / cat・sum の縮約を含む全ノード・f16 の重み）を既定のレーンで見る。縦長の I2V で I2V に固有な encoder の
 *   寸法は、ここが覆う。
 * - 帯は決定用の全 6 本（2 枚 × 3 寸法）の正規化後の maxAbs の最悪 × 5 を有効数字 2 桁へ切り上げた絶対値
 *   （`rtol = 0` — decoder の chunk 列の e2e と同じ導き方）。受入れは帯の内であることを門にする。`undefined` =
 *   未導出: 各ケースは maxAbs を記録して赤で止まり、最後の step が候補を出す。
 *
 * ## 故障注入（受入れの 1280×704 で帯の外に出ることを門にする）
 *
 * - patchify の高さと幅の副添字の取り違え（ホスト — 4 チャネルの組の 1 番と 2 番を入れ替えた入力）。
 * - std と 1/std の取り違え（ホスト — 統計の std を逆数にして同じ正規化を通す = mu · std）。
 * - post の束縛の h と w の入れ替え（受け渡し — 常駐のバイト数は同じなので大きさの検査を素通りし、post は
 *   `[640,1,h,w]` のバイト列を `[640,1,w,h]` と読む）。
 * - mu の代わりに logvar の 48 ch（post の出口の slice を後半へ — 宣言の差し替え）。
 * - AvgDown3D の偶奇の取り違え（時間倍率 2 の 2 ブロックで、ショートカットの平均を奇数番でなく偶数番のチャネルに
 *   足す — pre の宣言の slice 2 本と cat の入力の順の差し替え）。
 *
 * NOTE: std と 1/std の取り違えは GPU を回さない（照合の step で読んだ mu をホストで正規化し直すだけ）。言えるのは
 * 「帯が正規化の誤りより狭い」ことだけで、GPU の門の検出力の証明にはならない（式そのものはホストテスト
 * `wan_ti2v_i2v_preprocess_test.ts` が縛る）。GPU の受け渡しの経路を狙うのは束縛の入れ替え。
 *
 * 後の 2 つはグラフの中のノードなので、グラフの入出力からは注入できない（ショートカットは重みを持たず、mu の
 * 切り出しはグラフの出口そのもの）。資産は書き換えず、開いた容器の**宣言だけ**をメモリ上で差し替えた容器から
 * Session を張る（重みの block は元の容器から読む — {@link withRewrittenNodes}）。注入の Session は使い終わったら
 * その場で畳む（pre の 2 本目は重みと 1280×704 の計画の backing で約 0.9 GiB を持つ）。
 *
 * ## mid の attention（D = 640・S = h·w）
 *
 * 各ケースの直後に、attn の直近 run の計画が融合 attention の 3 段（`attention_qk` / `attention_stats` /
 * `attention_pv`）を 1 本以上積むことを門にする（decoder の D = 1024 と同じ門 — 値の正しさは帯が見るが、融合の
 * 経路を通ったかは帯からは分からない）。行統計の変種は S で変わる（キーの `rc` — S = 3,520 で 14・S = 160 で 1）
 * ので、寸法ごとに見る。
 *
 * ## 記録（門ではない — ADR 0121 段 9a の追記へ写す）
 *
 * Session 3 本の構築時間・ケースごとの encode の壁時間（寸法ごとの 1 本目は計画の導出とパイプラインの生成を
 * 含む）・Session の VRAM の内訳（重み・計画の backing）・常駐の大きさ・mu の差・attention の pipeline キー。
 *
 * NOTE: 寸法を変えたときに保持される backing の本数は、ADR 0095 決定 1 の予算つき LRU（`planBackingBudgetBytes` —
 * 既定 256 MiB）で決まる。入力が常駐かホストのテンソルかには依らない。2026-10-09 の実測: pre は 1280×704 の
 * 1 本（801.5 MiB）だけで予算を超えるので、256×160 で置き換わる（1 本）。attn（81.7 + 2.4 MiB）と post（17.2 MiB
 * × 2 + 0.8 MiB）は予算の内なので、寸法の数だけ残る（2 本 / 3 本）。Session あたりの保持は
 * `max(予算, 最大の 1 本)` を超えない。
 *
 * 資産（3 容器と golden 9 本）が 1 つも無い環境と GPU 無し環境は明示 SKIP。一部だけある環境は FAIL（GPU 不要の
 * 1 本目のテスト）。
 */

import { assert, assertEquals } from "@std/assert";
import {
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type ResidentTensor,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import { WAN22_TI2V_GENERATION } from "../src/wan/descriptor.ts";
import { patchifyWanImage, wanI2vPixels } from "../src/wan/i2v-preprocess.ts";
import { normalizeWanLatents } from "../src/wan/latents.ts";
import { disposeSteps } from "../src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { filePresent, firstBitMismatch, readBuffer, viewOf } from "./helpers/wan-ti2v-dit.ts";
import {
  WAN_I2V_GOLDEN_CASES,
  WAN_I2V_GOLDEN_GENERATE,
  WAN_I2V_GOLDEN_KEYS,
  WAN_I2V_GOLDEN_METADATA,
  WAN_I2V_IMAGE_SHA256,
  type WanI2vGoldenCase,
  wanI2vGoldenName,
  wanI2vGoldenUrl,
} from "./helpers/wan-i2v-image.ts";
import {
  WAN_TI2V_VAE_MODEL_FILE as MODEL_FILE,
  WAN_TI2V_VAE_ROOT as ROOT,
  WAN_TI2V_VAE_SERIES as SERIES,
} from "./helpers/wan-ti2v-vae.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import type { BoundContainer } from "../../runtime/src/format/container/bind.ts";
import type { IrNode } from "../../runtime/src/format/ir.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/**
 * 正規化後の潜在の帯（絶対値）。`undefined` = 未導出（モジュール doc「ケースと帯」）。
 *
 * 導出（2026-10-09・RTX 3080 Ti・Deno・全 9 本）: 決定用の最悪 boxing-cats 704×1280 の 1.550e-5 × 5 = 7.749e-5 →
 * 有効数字 2 桁へ切り上げ 7.8e-5。同じ日に 9 本を既定のレーンへ移した形で 2 回走らせ直し、全ケースの maxAbs が
 * 下の表と同値だった（帯を導いた走行の回数は記録が上書きされて残っていないので、確かめられる回数だけを書く）。
 *
 * | 寸法 | boxing-cats（決定用） | cat-dog-baking（決定用） | ferret（受入れ） | 参照の最大絶対値 |
 * |---|---|---|---|---|
 * | 1280×704 | 1.150e-5 | 1.264e-5 | 5.394e-6 | 2.95〜3.34 |
 * | 704×1280 | 1.550e-5 | 1.073e-5 | 6.557e-6 | 2.89〜3.68 |
 * | 256×160 | 5.186e-6 | 4.768e-6 | 4.813e-6 | 3.01〜4.04 |
 *
 * 故障注入（ferret 1280×704）の maxAbs: patchify の副添字の取り違え 2.44・std と 1/std の取り違え 2.54・post の
 * 束縛の h と w の入れ替え 1.18・logvar の 48 ch 67.3・AvgDown3D の偶奇の取り違え 2.77（どれも帯の 1.5 万倍以上）。
 *
 * MUST: 受入れ（ferret）の結果を見てこの値も決定用のケースも変えない。受入れが帯を外れたら、帯を広げずに原因を
 * 調べる。
 */
const BAND: number | undefined = 7.8e-5;
/** 判定と記録の帯（未導出の回は無限の帯として記録し、判定は赤にする）。 */
const TOLERANCE: Tolerance = { atol: BAND ?? Number.POSITIVE_INFINITY, rtol: 0 };

/** 部品名（= 部品ディレクトリ名。グラフ名は系列のグラフ名の表から引く）。 */
const PRE = "vae_encoder_pre";
const ATTENTION = "vae_encoder_attn";
const POST = "vae_encoder_post";
const PARTS = [PRE, ATTENTION, POST] as const;

/** グラフ入力の綴り（recipe の各モジュールの forward の引数名）。 */
const INPUT = { pre: "image", attn: "tokens", post: "hidden" } as const;

/** mid block のチャネル数（`base_dim 160 × dim_mult[-1] 4`）。 */
const HIDDEN_CHANNELS = 640;
/** 潜在のチャネル数（上流 config の `z_dim` — mu の本数）。 */
const LATENT_CHANNELS = 48;
/** patchify の倍率と、潜在 1 に対する画像の辺（encoder の down 3 段 8 × patchify 2）。 */
const PATCH = WAN22_TI2V_GENERATION.vaePatchSize;
const SPATIAL_COMPRESSION = 8 * PATCH;
/** 要素あたりのバイト数（意味論 dtype は全て 4 バイト — ADR 0009）。 */
const BYTES_PER_ELEMENT = 4;

/** 帯の決定用の画像（recipe の `BAND_IMAGES`・golden のメタ `role` と突き合わせる）。 */
const BAND_IMAGES: readonly string[] = ["boxing-cats", "cat-dog-baking"];

/** 故障注入の入力（受入れの 1280×704）。 */
const FAULT_SIZE = { width: 1280, height: 704 } as const;
const FAULT_IMAGE = "ferret";

/** 記録する融合 attention の 3 段（ADR 0023 — ①QK・②行統計・③PV のパイプラインキーの接頭辞）。 */
const ATTENTION_KERNELS = ["attention_qk", "attention_stats", "attention_pv"] as const;

type Role = "band" | "accept";
type EncoderCase = WanI2vGoldenCase & {
  readonly name: string;
  readonly role: Role;
};

/**
 * encoder の golden（fit = crop の 9 本 — golden の表の並び = 寸法ごとに画像 3 枚。回す順もこの並び = 束縛は
 * 1280×704 → 704×1280 → 256×160）。
 */
const CASES: readonly EncoderCase[] = WAN_I2V_GOLDEN_CASES
  .filter(({ fit }) => fit === "crop")
  .map((testCase) => ({
    ...testCase,
    name: wanI2vGoldenName(testCase),
    role: BAND_IMAGES.includes(testCase.image) ? "band" : "accept",
  }));
const FAULT_CASE = CASES.find(({ image, width, height }) =>
  image === FAULT_IMAGE && width === FAULT_SIZE.width && height === FAULT_SIZE.height
);

const modelUrl = (part: string): URL => new URL(`${part}/${MODEL_FILE}`, ROOT);
const ASSETS = [
  ...PARTS.map((part) => ({
    path: modelUrl(part).pathname,
    present: modelPresent(modelUrl(part)),
  })),
  ...CASES.map((testCase) => {
    const url = wanI2vGoldenUrl(testCase);
    return { path: url.pathname, present: filePresent(url) };
  }),
];
const ANY_ASSETS = ASSETS.some(({ present }) => present);
const ALL_ASSETS = ASSETS.every(({ present }) => present);
if (!ANY_ASSETS) {
  console.warn(
    `[karume] ${ROOT.pathname} に Wan2.2 TI2V の VAE encoder の 3 グラフと golden が無いため、encoder の照合を ` +
      `SKIP する。生成: ${WAN_I2V_GOLDEN_GENERATE}`,
  );
}

const results = openResults("wan-ti2v-vae-encoder");

// ---- golden ---------------------------------------------------------------------------------

type EncoderGolden = {
  readonly rgb8: Uint8Array<ArrayBuffer>;
  readonly encoderInput: Float32Array<ArrayBuffer>;
  readonly mu: Float32Array<ArrayBuffer>;
  readonly latent: Float32Array<ArrayBuffer>;
};

const metadataOf = (file: SafetensorsFile, key: string, where: string): string => {
  const value = file.metadata.get(key);
  if (value === undefined) throw new Error(`${where}: __metadata__.${key} が無い`);
  return value;
};

/** テンソルを dtype と形の検査つきで写す（U8 が前に並ぶと F32 の開始が 4 バイト境界に載る保証が無い）。 */
const tensorBytes = (
  file: SafetensorsFile,
  key: string,
  dtype: "U8" | "F32",
  shape: readonly number[],
  where: string,
): ArrayBuffer => {
  const view = viewOf(file, key, where);
  assertEquals(view.dtype, dtype, `${where}: ${key} の dtype`);
  assertEquals([...view.shape], [...shape], `${where}: ${key} の形`);
  return file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
};

const readGolden = async (testCase: EncoderCase): Promise<EncoderGolden> => {
  const where = testCase.name;
  const file = parseSafetensors(await readBuffer(wanI2vGoldenUrl(testCase)));
  const meta = WAN_I2V_GOLDEN_METADATA;
  const { width, height } = testCase;
  assertEquals(metadataOf(file, meta.image, where), testCase.image, `${where}: 画像`);
  assertEquals(
    metadataOf(file, meta.imageSha256, where),
    WAN_I2V_IMAGE_SHA256[testCase.image],
    `${where}: 画像の sha256 が表と違う`,
  );
  assertEquals(metadataOf(file, meta.fit, where), "crop", `${where}: fit`);
  assertEquals(
    [metadataOf(file, meta.width, where), metadataOf(file, meta.height, where)],
    [String(width), String(height)],
    `${where}: 出力寸法`,
  );
  assertEquals(
    metadataOf(file, meta.role, where),
    testCase.role,
    `${where}: 役割（band / accept）`,
  );
  const latentShape = [
    LATENT_CHANNELS,
    1,
    height / SPATIAL_COMPRESSION,
    width / SPATIAL_COMPRESSION,
  ];
  const keys = WAN_I2V_GOLDEN_KEYS;
  return {
    rgb8: new Uint8Array(tensorBytes(file, keys.rgb8, "U8", [height, width, 3], where)),
    encoderInput: new Float32Array(
      tensorBytes(
        file,
        keys.encoderInput,
        "F32",
        [3 * PATCH * PATCH, 1, height / PATCH, width / PATCH],
        where,
      ),
    ),
    mu: new Float32Array(tensorBytes(file, keys.mu, "F32", latentShape, where)),
    latent: new Float32Array(tensorBytes(file, keys.latent, "F32", latentShape, where)),
  };
};

/** TS の前処理（golden の RGB8 → `[-1, 1]` → patchify）。 */
const encoderInputOf = (testCase: EncoderCase, rgb8: Uint8Array<ArrayBuffer>) =>
  patchifyWanImage(
    wanI2vPixels({ data: rgb8, width: testCase.width, height: testCase.height }),
    [3, testCase.height, testCase.width],
    PATCH,
  );

// ---- 容器と宣言の差し替え -------------------------------------------------------------------

type EncoderParts<T> = { readonly pre: T; readonly attn: T; readonly post: T };

const openParts = async (): Promise<EncoderParts<BoundContainer>> => {
  const [pre, attn, post] = await Promise.all(
    PARTS.map((part) => openSeriesContainer(modelUrl(part))),
  );
  return { pre, attn, post };
};

const prepareParts = (opened: EncoderParts<BoundContainer>): EncoderParts<PreparedModel> => ({
  pre: prepareContainer(opened.pre, seriesGraph(SERIES, PRE)),
  attn: prepareContainer(opened.attn, seriesGraph(SERIES, ATTENTION)),
  post: prepareContainer(opened.post, seriesGraph(SERIES, POST)),
});

/** グラフの出口の名前（宣言から読む — 綴りは export の都合で決まる）。 */
const outputOf = (model: PreparedModel, part: string): string => {
  const [output, ...rest] = model.graph.outputs;
  if (output === undefined || rest.length > 0) {
    throw new Error(`${part}: 出力が 1 本でない（${model.graph.outputs.join(" / ")}）`);
  }
  return output;
};

/**
 * 宣言のノードだけを差し替えた容器（故障注入用）。重みの block と供給計画は元の容器のまま — 資産は書き換えない。
 */
const withRewrittenNodes = (
  opened: BoundContainer,
  graphName: string,
  rewrite: (nodes: readonly IrNode[]) => readonly IrNode[],
): BoundContainer => {
  const bound = opened.graphs[graphName];
  if (bound === undefined) throw new Error(`容器にグラフ '${graphName}' が無い`);
  return {
    graphs: {
      [graphName]: {
        declaration: { ...bound.declaration, nodes: rewrite(bound.declaration.nodes) },
        supplies: bound.supplies,
      },
    },
    readBlock: (id) => opened.readBlock(id),
  };
};

/** 値名 → それを出すノードの添字。 */
const producers = (nodes: readonly IrNode[]): ReadonlyMap<string, number> =>
  new Map(nodes.flatMap((node, index) => node.outs.map((name) => [name, index] as const)));

const sliceRange = (node: IrNode | undefined): string =>
  node === undefined || node.op !== "slice"
    ? `${node?.op ?? "無し"}`
    : `slice dim ${node.attrs["dim"]} [${node.attrs["start"]}, ${node.attrs["end"]})`;

/**
 * 故障: post の出口の切り出しを quant_conv の出力の後半（logvar 側の 48 ch）へ。出口の slice（dim 0・[0, 48)）が
 * 宣言に無ければ投げる（注入が空振りしない）。
 */
const takeLogvarHalf = (output: string) => (nodes: readonly IrNode[]): readonly IrNode[] => {
  const index = producers(nodes).get(output);
  const node = index === undefined ? undefined : nodes[index];
  if (
    index === undefined || node?.op !== "slice" || node.attrs["dim"] !== 0 ||
    node.attrs["start"] !== 0 || node.attrs["end"] !== LATENT_CHANNELS
  ) {
    throw new Error(`post の出口 '${output}' が mu の切り出しでない（${sliceRange(node)}）`);
  }
  const logvar = {
    ...node,
    attrs: { ...node.attrs, start: LATENT_CHANNELS, end: 2 * LATENT_CHANNELS },
  };
  return nodes.map((current, at) => at === index ? logvar : current);
};

/**
 * 故障: AvgDown3D の時間倍率 2 のブロック（2 本 — down_blocks.1 / .2）で、ショートカットの空間 2×2 の平均を
 * 奇数番（`2c+1`）でなく偶数番（`2c`）のチャネルに足す。
 *
 * recipe の閉じた形は `pairs = residual.reshape(C, 2, h, w)` → `cat([pairs[:, 0:1], pairs[:, 1:2] + mean], 1)`。
 * 2 本の slice の範囲を入れ替え（0:1 ↔ 1:2）、cat の入力の順を入れ替えると `[pairs[:, 0:1] + mean, pairs[:, 1:2]]`
 * になる。形が想定と違えば投げる（注入が空振りしない）。
 */
const swapShortcutParity = (nodes: readonly IrNode[]): readonly IrNode[] => {
  const producedBy = producers(nodes);
  const nodeOf = (name: string): { readonly index: number; readonly node: IrNode } | undefined => {
    const index = producedBy.get(name);
    return index === undefined ? undefined : { index, node: nodes[index] };
  };
  const isSlice = (node: IrNode, start: number, end: number): boolean =>
    node.op === "slice" && node.attrs["dim"] === 1 && node.attrs["start"] === start &&
    node.attrs["end"] === end;
  const replaced = new Map<number, IrNode>();
  const cats = nodes.flatMap((node, index) => node.op === "cat" ? [index] : []);
  assertEquals(cats.length, 2, "pre の cat が時間倍率 2 のショートカットの 2 本でない");
  for (const catIndex of cats) {
    const cat = nodes[catIndex];
    const [zeroName, sumName] = cat.ins;
    const zero = nodeOf(zeroName ?? "");
    const sum = nodeOf(sumName ?? "");
    if (
      cat.attrs["dim"] !== 1 || cat.ins.length !== 2 || zero === undefined || sum === undefined ||
      !isSlice(zero.node, 0, 1) || sum.node.op !== "add"
    ) {
      throw new Error(`cat '${cat.outs[0]}' がショートカットの閉じた形でない`);
    }
    const data = sum.node.ins.map(nodeOf).find((entry) =>
      entry !== undefined && isSlice(entry.node, 1, 2) && entry.node.ins[0] === zero.node.ins[0]
    );
    if (data === undefined) throw new Error(`cat '${cat.outs[0]}' の奇数番の slice が無い`);
    replaced.set(zero.index, { ...zero.node, attrs: { ...zero.node.attrs, start: 1, end: 2 } });
    replaced.set(data.index, { ...data.node, attrs: { ...data.node.attrs, start: 0, end: 1 } });
    replaced.set(catIndex, { ...cat, ins: [sumName, zeroName] });
  }
  return nodes.map((node, index) => replaced.get(index) ?? node);
};

/** patchify の高さと幅の副添字の取り違え（4 チャネルの組 `c·4 + dx·2 + dy` の 1 番と 2 番を入れ替える）。 */
const swapSubIndices = (input: Float32Array, plane: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(input);
  for (let group = 0; group < input.length / (4 * plane); group += 1) {
    const one = (group * 4 + 1) * plane;
    const two = (group * 4 + 2) * plane;
    out.set(input.subarray(one, one + plane), two);
    out.set(input.subarray(two, two + plane), one);
  }
  return out;
};

// ---- encode（pre → attn → post を 1 本の batch で） ------------------------------------------

/** 寸法ごとの常駐（pre の出口 / attn の出口 / mu）。大きさで引き当てる（1280×704 と 704×1280 は同じ大きさ）。 */
class EncoderResidents {
  readonly #gpu: GpuContext;
  readonly #byArea = new Map<number, EncoderParts<ResidentTensor>>();

  constructor(gpu: GpuContext) {
    this.#gpu = gpu;
  }

  async forArea(area: number): Promise<EncoderParts<ResidentTensor>> {
    const known = this.#byArea.get(area);
    if (known !== undefined) return known;
    const hidden = HIDDEN_CHANNELS * area * BYTES_PER_ELEMENT;
    const created = {
      pre: await this.#gpu.createResident(hidden, `encoder pre ${area}`),
      attn: await this.#gpu.createResident(hidden, `encoder attn ${area}`),
      post: await this.#gpu.createResident(
        LATENT_CHANNELS * area * BYTES_PER_ELEMENT,
        `encoder mu ${area}`,
      ),
    };
    this.#byArea.set(area, created);
    return created;
  }

  get bytes(): number {
    return [...this.#byArea.values()].reduce(
      (total, { pre, attn, post }) => total + pre.byteLength + attn.byteLength + post.byteLength,
      0,
    );
  }

  /** MUST: Session を畳んだ後に呼ぶ（焼き込みの参照が残る間は破棄を拒まれる）。 */
  dispose(): void {
    for (const { pre, attn, post } of this.#byArea.values()) {
      pre.dispose();
      attn.dispose();
      post.dispose();
    }
    this.#byArea.clear();
  }
}

/**
 * 3 グラフを 1 本の batch で回して mu を読む（モジュール doc「受け渡し」）。
 *
 * MUST: 同じ device の別の batch・run を並行に発行しない（batch は device の区間ロックを持つ）。
 */
const encodeOnGpu = async (
  gpu: GpuContext,
  sessions: EncoderParts<Session>,
  outputs: EncoderParts<string>,
  residents: EncoderResidents,
  input: Float32Array<ArrayBuffer>,
  size: { readonly width: number; readonly height: number },
  /** post の束縛の差し替え（故障注入だけが渡す — 既定は size から導いた h, w）。 */
  postBindings?: { readonly h: number; readonly w: number },
): Promise<Float32Array<ArrayBuffer>> => {
  const h = size.height / SPATIAL_COMPRESSION;
  const w = size.width / SPATIAL_COMPRESSION;
  const resident = await residents.forArea(h * w);
  const batch = await gpu.beginBatch();
  try {
    await sessions.pre.enqueue(
      { [INPUT.pre]: { dtype: "f32", shape: [3 * PATCH * PATCH, 1, 8 * h, 8 * w], data: input } },
      { batch, copyOutputs: { [outputs.pre]: resident.pre } },
    );
    await sessions.attn.enqueue(
      { [INPUT.attn]: resident.pre },
      { batch, bindings: { S: h * w }, copyOutputs: { [outputs.attn]: resident.attn } },
    );
    await sessions.post.enqueue(
      { [INPUT.post]: resident.attn },
      { batch, bindings: postBindings ?? { h, w }, copyOutputs: { [outputs.post]: resident.post } },
    );
  } catch (cause) {
    // MUST: 区間を閉じてロックを返す（閉じないと同じ device の次の batch / run が永久に待つ）。
    await batch.finish().catch(() => undefined);
    throw cause;
  }
  const read = await batch.finishAndRead({ mu: resident.post });
  return new Float32Array(read["mu"]);
};

// ---- 記録 -------------------------------------------------------------------------------------

/** 有効数字 2 桁への切り上げ（decoder の chunk 列の e2e の帯の候補と同じ規則）。 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(value / unit) * unit).toPrecision(2));
};

/**
 * 融合 attention の 3 段の dispatch 本数と、`attention` で始まるパイプラインキーの一覧（直近 run の計画から）。
 *
 * MUST: census の無い run を 0 本として数えない（融合の門が黙って空振りする）— 無ければ投げる。
 */
const attentionPipelines = (diagnostics: SessionDiagnostics) => {
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) throw new Error("attn: 直近 run の pipeline の census が無い");
  const dispatches = (kernel: (typeof ATTENTION_KERNELS)[number]): number =>
    pipelines.filter(({ key }) => key.split(":")[0] === kernel)
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  return {
    dispatches: Object.fromEntries(ATTENTION_KERNELS.map((kernel) => [kernel, dispatches(kernel)])),
    keys: pipelines.filter(({ key }) => key.startsWith("attention")).map(({ key }) => key),
  };
};

/**
 * Session ごとの VRAM の内訳（重み・計画の backing — enqueue の経路の中間は backing に載る）と常駐の大きさ。
 */
const vramNote = (sessions: EncoderParts<Session>, residents: EncoderResidents): string => {
  const mib = (bytes: number): string => `${(bytes / 2 ** 20).toFixed(1)} MiB`;
  const parts = Object.entries(sessions).map(([part, session]) => {
    const stats = session.diagnostics();
    return `${part} 重み ${mib(stats.weights.allocatedBytes)}・backing ${
      mib(stats.planBacking.residentBytes + stats.planBacking.inputBytes)
    }（${stats.planBacking.retainedCount} 本）`;
  });
  return [...parts, `常駐 ${mib(residents.bytes)}`].join(" / ");
};

const maxAbsOf = (values: Float32Array): number =>
  values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

// ---- テスト -----------------------------------------------------------------------------------

Deno.test({
  name:
    "Wan2.2 TI2V VAE encoder: 3 グラフと golden 9 本が揃っている（一部だけなら FAIL・無ければ SKIP）",
  ignore: !ANY_ASSETS,
  fn: () => {
    const missing = ASSETS.filter(({ present }) => !present).map(({ path }) => path);
    assertEquals(missing, [], `encoder の資産が欠けている（採り直す: ${WAN_I2V_GOLDEN_GENERATE}）`);
  },
});

Deno.test({
  name:
    "Wan2.2 TI2V VAE encoder: pre → attn → post（常駐の受け渡し）の mu の正規化が上流の非タイル encode と" +
    "帯の中で一致し、故障注入は帯の外（実 GPU）",
  ignore: !ALL_ASSETS || !GPU_AVAILABLE,
  fn: async (t) => {
    await assertRunningAdapter();
    assert(FAULT_CASE !== undefined, "故障注入のケースが表に無い");
    const opened = await openParts();
    const models = prepareParts(opened);
    const outputs = {
      pre: outputOf(models.pre, PRE),
      attn: outputOf(models.attn, ATTENTION),
      post: outputOf(models.post, POST),
    };
    /** ケースごとの maxAbs（帯の候補の材料 — 判定の前に積むので帯の外の回も残る）。 */
    const observed = new Map<string, number>();
    /** 受入れ（故障注入の入力）の正しい mu（std と 1/std の取り違えの注入に使う）。 */
    let faultMu: Float32Array<ArrayBuffer> | undefined;

    const gpu = await acquireTestGpu();
    const residents = new EncoderResidents(gpu);
    /** 製品の Session 3 本（後始末で畳む — 故障注入の Session は {@link encodeWithRewritten} がその場で畳む）。 */
    const created: Session[] = [];
    const createSession = async (model: PreparedModel): Promise<Session> => {
      const session = await model.createContainerSession(gpu);
      created.push(session);
      return session;
    };
    let failure: { readonly error: unknown } | undefined;
    try {
      const building = performance.now();
      const live: EncoderParts<Session> = {
        pre: await createSession(models.pre),
        attn: await createSession(models.attn),
        post: await createSession(models.post),
      };
      console.log(
        `[wan-ti2v-vae-encoder] Session 3 本の構築 ${(performance.now() - building).toFixed(0)} ms`,
      );

      await t.step("取り決め: 入出力の名前と形・記号（pre [h,w]・attn [S]・post [h,w]）", () => {
        const io = (model: PreparedModel, part: string) => ({
          symbols: [...model.graph.symbols],
          inputs: model.graph.inputs.map(({ name, shape }) => [name, [...shape]]),
          output: [...(model.graph.values[outputOf(model, part)]?.shape ?? [])],
        });
        assertEquals(io(models.pre, PRE), {
          symbols: ["h", "w"],
          inputs: [[INPUT.pre, [3 * PATCH * PATCH, 1, "8h", "8w"]]],
          output: [HIDDEN_CHANNELS, 1, "h", "w"],
        });
        assertEquals(io(models.attn, ATTENTION), {
          symbols: ["S"],
          inputs: [[INPUT.attn, [HIDDEN_CHANNELS, "S"]]],
          output: [HIDDEN_CHANNELS, "S"],
        });
        assertEquals(io(models.post, POST), {
          symbols: ["h", "w"],
          inputs: [[INPUT.post, [HIDDEN_CHANNELS, 1, "h", "w"]]],
          output: [LATENT_CHANNELS, 1, "h", "w"],
        });
      });

      /** 寸法ごとの 1 本目か（壁時間に計画の導出とパイプラインの生成が入る）。 */
      const seenSizes = new Set<string>();
      for (const testCase of CASES) {
        await t.step(
          `${testCase.name}（${testCase.role}）: TS の前処理が golden の encoder 入力とビット一致し、` +
            "正規化した mu が帯の中",
          async () => {
            await runRecordedCase(results, { id: testCase.name }, async ({ measurements }) => {
              const golden = await readGolden(testCase);
              const input = encoderInputOf(testCase, golden.rgb8);
              const mismatch = firstBitMismatch(input, golden.encoderInput);
              assertEquals(
                mismatch,
                -1,
                `${testCase.name}: TS の前処理が golden の encoder 入力と割れる`,
              );

              const sizeKey = `${testCase.width}x${testCase.height}`;
              const cold = !seenSizes.has(sizeKey);
              seenSizes.add(sizeKey);
              const started = performance.now();
              const mu = await encodeOnGpu(gpu, live, outputs, residents, input, testCase);
              const encodeMs = performance.now() - started;
              const latent = normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
              const report = allclose(latent, golden.latent, TOLERANCE);
              const muReport = allclose(mu, golden.mu, { atol: Number.POSITIVE_INFINITY, rtol: 0 });
              const attention = attentionPipelines(live.attn.diagnostics());
              measurements.push({
                output: "latent",
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: TOLERANCE,
                stage: "karume",
              });
              const line = `${testCase.name}: latent maxAbs=${report.maxAbsError} refMax=${
                maxAbsOf(golden.latent)
              } / mu maxAbs=${muReport.maxAbsError} refMax=${
                maxAbsOf(golden.mu)
              } nonFinite=${report.nonFiniteCount} encodeMs=${encodeMs.toFixed(0)}${
                cold ? "（寸法の 1 本目）" : ""
              } / ${vramNote(live, residents)} / attention ${
                JSON.stringify(attention.dispatches)
              } ${attention.keys.join(" ")}`;
              console.log(`[wan-ti2v-vae-encoder] ${line}`);
              assertEquals(report.nonFiniteCount, 0, `${testCase.name}: 非有限`);
              // mid の attention（D = 640・S = h·w）が融合 attention の 3 段を通る（モジュール doc）。行統計の
              // 変種は S で変わるので寸法ごとに見る。
              for (const kernel of ATTENTION_KERNELS) {
                assert(
                  attention.dispatches[kernel] > 0,
                  `${testCase.name}: ${kernel} の dispatch が 0 本（${
                    JSON.stringify(attention.dispatches)
                  }）`,
                );
              }
              observed.set(testCase.name, report.maxAbsError);
              if (testCase === FAULT_CASE) faultMu = mu;
              if (BAND === undefined) {
                throw new Error(
                  `${testCase.name}: 帯が未導出（maxAbs ${report.maxAbsError}）— 決定用の最悪 × 5 を帯の定数へ` +
                    "書く（帯の候補は最後の step が出す）",
                );
              }
              assert(
                report.pass,
                `${testCase.name}: maxAbs ${report.maxAbsError} が帯 ${BAND} の外`,
              );
              return { status: "pass", note: line };
            });
          },
        );
      }

      /** 故障の入力で encode して、受入れの golden に対して帯の外であることを見る。 */
      const faultStep = async (
        id: string,
        label: string,
        latentOf: (golden: EncoderGolden, faultCase: EncoderCase) => Promise<Float32Array>,
      ): Promise<void> => {
        await t.step(`故障注入: ${label} → ${FAULT_CASE.name} が帯の外`, async () => {
          await runRecordedCase(results, { id: `fault/${id}` }, async ({ measurements }) => {
            const golden = await readGolden(FAULT_CASE);
            const latent = await latentOf(golden, FAULT_CASE);
            const report = allclose(latent, golden.latent, TOLERANCE);
            measurements.push({
              output: `latent(fault ${id})`,
              maxAbs: report.maxAbsError,
              maxRel: report.maxRelError,
              tolerance: TOLERANCE,
              stage: "karume",
            });
            const line =
              `fault ${id}: maxAbs=${report.maxAbsError} nonFinite=${report.nonFiniteCount}${
                BAND === undefined ? "" : ` band×${(report.maxAbsError / BAND).toFixed(0)}`
              }`;
            console.log(`[wan-ti2v-vae-encoder] ${line}`);
            if (BAND === undefined) {
              throw new Error(`${label}: 帯が未導出（故障注入の maxAbs ${report.maxAbsError}）`);
            }
            assert(!report.pass, `${label}: 帯 ${BAND} の中に収まった（帯が広すぎる兆候）`);
            return { status: "pass", note: line };
          });
        });
      };

      await faultStep(
        "patchify-swap",
        "patchify の高さと幅の副添字の取り違え",
        async (golden, faultCase) => {
          const plane = (faultCase.height / PATCH) * (faultCase.width / PATCH);
          const input = swapSubIndices(encoderInputOf(faultCase, golden.rgb8), plane);
          const mu = await encodeOnGpu(gpu, live, outputs, residents, input, faultCase);
          return normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
        },
      );

      await faultStep(
        "std-inverse",
        "std と 1/std の取り違え（mu · std — ホストの正規化だけ・GPU は回さない）",
        () => {
          if (faultMu === undefined) {
            throw new Error(`${FAULT_CASE.name} の mu が無い（照合の step が落ちた）`);
          }
          const { mean, std } = WAN22_TI2V_GENERATION.latents;
          return Promise.resolve(
            normalizeWanLatents(faultMu, { mean, std: std.map((value) => 1 / value) }),
          );
        },
      );

      await faultStep(
        "post-bindings-swap",
        "post の束縛の h と w の入れ替え（常駐のバイト数は同じ — 受け渡しの検査を素通りする）",
        async (golden, faultCase) => {
          const h = faultCase.height / SPATIAL_COMPRESSION;
          const w = faultCase.width / SPATIAL_COMPRESSION;
          assert(h !== w, `${faultCase.name}: h = w では入れ替えが空振りする`);
          const mu = await encodeOnGpu(
            gpu,
            live,
            outputs,
            residents,
            encoderInputOf(faultCase, golden.rgb8),
            faultCase,
            { h: w, w: h },
          );
          return normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
        },
      );

      /**
       * 宣言を差し替えた 1 部品の Session で encode する（他の 2 部品は製品の Session）。
       *
       * 注入の Session は使い終わったらその場で畳み、解放を待つ（pre の 2 本目は重みと 1280×704 の計画の backing で
       * 約 0.9 GiB — 残りの step と後始末まで持ち越すと、VRAM の小さい機で注入の step だけが OOM になりうる）。畳む
       * 失敗で本体の失敗を上書きしない。
       */
      const encodeWithRewritten = async (
        part: "pre" | "post",
        rewrite: (nodes: readonly IrNode[]) => readonly IrNode[],
        golden: EncoderGolden,
        faultCase: EncoderCase,
      ): Promise<Float32Array> => {
        const name = part === "pre" ? PRE : POST;
        const graphName = seriesGraph(SERIES, name);
        const faulty = await prepareContainer(
          withRewrittenNodes(opened[part], graphName, rewrite),
          graphName,
        ).createContainerSession(gpu);
        let injected: { readonly error: unknown } | undefined;
        try {
          const mu = await encodeOnGpu(
            gpu,
            { ...live, [part]: faulty },
            outputs,
            residents,
            encoderInputOf(faultCase, golden.rgb8),
            faultCase,
          );
          return normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
        } catch (error) {
          injected = { error };
          throw error;
        } finally {
          await disposeSteps([
            () => {
              if (injected !== undefined) throw injected.error;
            },
            () => faulty.dispose(),
            () => settleReleases(gpu),
          ]);
        }
      };

      await faultStep(
        "logvar",
        "mu の代わりに logvar の 48 ch（post の出口の slice を後半へ）",
        (golden, faultCase) =>
          encodeWithRewritten("post", takeLogvarHalf(outputs.post), golden, faultCase),
      );

      await faultStep(
        "avgdown-parity",
        "AvgDown3D の偶奇の取り違え（平均を偶数番のチャネルに足す — 時間倍率 2 の 2 ブロック）",
        (golden, faultCase) => encodeWithRewritten("pre", swapShortcutParity, golden, faultCase),
      );

      if (BAND === undefined) {
        await t.step("帯の候補（未導出）", async () => {
          const bandCases = CASES.filter(({ role }) => role === "band");
          const missing = bandCases.filter(({ name }) => !observed.has(name)).map(({ name }) =>
            name
          );
          const worst = Math.max(...bandCases.map(({ name }) => observed.get(name) ?? 0));
          const candidate = missing.length > 0
            ? `決定用 ${missing.join(" / ")} が maxAbs を出す前に落ちた — 候補にしない`
            : worst > 0
            ? `決定用の最悪 ${worst.toExponential(3)} × 5 = ${(worst * 5).toExponential(3)}` +
              `（有効数字 2 桁へ切り上げ ${roundUpTwoDigits(worst * 5)}）`
            : "決定用の maxAbs が 0 — × 5 で帯を導けない（照合の経路を先に調べる）";
          const accepted = CASES.filter(({ role }) => role === "accept")
            .map(({ name }) => `${name} ${observed.get(name)?.toExponential(3) ?? "—"}`);
          const message = `帯が未導出 — ${candidate}。受入れの maxAbs（帯の決定に使わない）: ${
            accepted.join(" / ")
          }`;
          await runRecordedCase(
            results,
            { id: "band-candidate", failureNote: () => message },
            () => {
              console.log(`[wan-ti2v-vae-encoder] ${message}`);
              throw new Error(message);
            },
          );
        });
      }
    } catch (error) {
      failure = { error };
      throw error;
    } finally {
      // MUST: Session → 常駐 → device の順で畳む（常駐は Session の焼き込みから参照されるので Session を先に畳む。
      // 1 段が落ちても残りの段を必ず通し、畳む失敗で本体の失敗を上書きしない）。
      await disposeSteps([
        () => {
          if (failure !== undefined) throw failure.error;
        },
        ...created.map((session) => () => session.dispose()),
        () => residents.dispose(),
        () => settleReleases(gpu),
        () => gpu.destroy(),
      ]);
    }
  },
});
