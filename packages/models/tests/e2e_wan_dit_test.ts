/**
 * Wan2.1 の DiT（S 形グラフ）の**移植の門**（実 GPU・ADR 0118 段 2 — 決定 3 / 8）。
 *
 * export した IR（f16 席・`outputs/series/wan2.1-t2v-1.3b-f16-dyn/transformer/`）を karume runtime で
 * 回し、ホストの unpatchify（`src/wan/dit-tokens.ts`）を通した潜在を、上流の**素の** diffusers
 * `WanTransformer3DModel`（CPU f32・同じ f16 丸めの重み）の 1 forward と突き合わせる。配布形の manifest は
 * まだ無いので、容器（`krm`）を直接開く。
 *
 * 生成は `cd tools/export-recipes && uv run --group wan --inexact python -m wan.export_dit`（golden の
 * 中身とケースの表は同ファイルの docstring）。golden は 2 本ずつ:
 *
 * - `io.<case>` — グラフの入力（`input.*`）と、パッチ後の torch CPU の出力（`output.0`）
 * - `reference.<case>` — 上流の素の forward の入力の潜在・timestep・出力（`[1,16,F,H,W]`）・各ブロックの出力
 *
 * パッチ後の eager は全ケースで上流と**ビット一致**（export 台本の `[eager]` 行 — patch 埋め込みの Linear 化も
 * この機の torch CPU では一致した）なので、ここで見る差はそのまま「GPU で回した IR」と「上流」の差。
 *
 * ## 帯（決定 8・追記 2026-10-02 の「帯の決定と受入れを分ける」）
 *
 * 指標はケースごとの**比** = 最大絶対差 ÷ 参照の最大絶対値（決定 8 の目安の形）。{@link DIT_RATIO_BAND} は
 * `band` 6 ケースの最悪の比の約 5 倍で決め、`accept` 3 ケース（`band` と seed が違う未見の入力）で受け入れる。
 * `growth` 2 ケース（S = 768）は S に対する伸びの記録で、帯の門には入れない（S = 14,040 の帯は段 3 が実寸で
 * 独立に導く — 決定 8）。故障注入 4 件（RoPE の h / w の取り違え・unpatchify の並びの取り違え・timestep の
 * cos / sin の前後反転・timestep の 1 ずれ）が受入れケースで帯の外へ出ることも門にする。
 *
 * MUST: `accept` の結果を見て `band` のケースを変えない（帯の決定と受入れの独立が崩れる — ケースの正本は
 * `wan/export_dit.py` の `CASES`）。受入れが帯を外れたら、帯を広げずに原因を調べる。
 *
 * 層数に対する伸びは計測用のグラフ（層別の出口 31 本 — `…-f16-dyn-probe/`・`export_dit --layers`）で各
 * ブロックの出力を上流の forward hook の値と突き合わせて記録する（門ではない）。
 *
 * 資産が無い環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。資産が**一部だけ**ある
 * 環境は SKIP ではなく FAIL にする（下の完全性テスト）。
 */

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  type FusionCounts,
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type SafetensorsFile,
  type Tensor,
} from "@karume/runtime";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  wanTokenGrid,
  wanTokenWidth,
} from "../src/wan/dit-tokens.ts";
import { parseWanRopeBase, type WanRopeBase, wanRopeTables } from "../src/wan/dit-rope.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/**
 * GPU（B570・f16 席）の DiT 1 forward と上流（CPU f32）の差の許容（比 = 最大絶対差 ÷ 参照の最大絶対値・
 * unpatchify 後の潜在）。
 *
 * 実測（2026-10-02・`deno-intel-graphics-bmg-g21`・`atol = rtol = 0` の素の突合）:
 *
 * | 役割   | ケース                | 格子 F'·H'·W' | 最大絶対差 | 参照の最大絶対値 | 比（io）     | 比（TS の proj） |
 * | ------ | --------------------- | ------------- | ---------- | ---------------- | ------------ | ---------------- |
 * | band   | `band-s00192-t0999`   | 3·8·8         | 2.694e-5   | 4.024            | 6.70e-6      | 9.30e-6          |
 * | band   | `band-s00192-t0750`   | 2·8·12        | 5.460e-5   | 5.468            | 9.98e-6      | 1.20e-5          |
 * | band   | `band-s00192-t0500`   | 3·8·8         | 6.363e-5   | 3.569            | 1.78e-5      | 1.67e-5          |
 * | band   | `band-s00192-t0250`   | 1·8·24        | 2.694e-5   | 5.321            | 5.06e-6      | 4.70e-6          |
 * | band   | `band-s00192-t0600`   | 2·8·12        | 5.406e-5   | 5.594            | 9.66e-6      | 1.03e-5          |
 * | band   | `band-s00192-t0113`   | 3·8·8         | 8.273e-5   | 4.321            | **1.91e-5**  | 1.62e-5          |
 * | accept | `accept-s00192-t0600` | 2·8·12        | 3.247e-4   | 5.242            | 6.19e-5      | 4.81e-5          |
 * | accept | `accept-s00128-t0030` | 4·4·8         | 3.101e-5   | 4.784            | 6.48e-6      | 5.97e-6          |
 * | accept | `accept-s00192-t0400` | 2·12·8        | 6.151e-5   | 6.234            | 9.87e-6      | 9.87e-6          |
 * | growth | `growth-s00768-t0999` | 3·16·16       | 5.892e-5   | 4.426            | 1.33e-5      | 1.30e-5          |
 * | growth | `growth-s00768-t0500` | 3·16·16       | 4.804e-5   | 4.874            | 9.86e-6      | 8.98e-6          |
 *
 * 「比（io）」は golden の torch の `timesteps_proj` を入れた値（最大絶対差の列もこちら）、「比（TS の proj）」は
 * ホストの `timestepsProj` を入れた値。
 *
 * 受入れケースの故障注入の比（RoPE の h / w・unpatchify の並び・cos / sin 反転・**timestep の 1 ずれ**）:
 * `accept-s00192-t0600` 4.03e-1 / 1.28 / 2.88e-1 / **9.18e-3**・`accept-s00128-t0030` 1.06e-1 / 1.43 / 3.95e-1 /
 * **6.91e-3**・`accept-s00192-t0400` 2.48e-1 / 1.32 / 2.79e-1 / **1.93e-3**。
 *
 * 帯 9.6e-5 は band の最悪の比 1.915e-5（`band-s00192-t0113` の io）の約 5.0 倍。受入れの最悪 6.19e-5 は帯の
 * 0.64 倍。微妙な故障（timestep の 1 ずれ）の最小 1.93e-3 は帯の約 20 倍で、O(1) の故障 3 件は 3〜4 桁上に出る。
 *
 * 絶対差でなく比で見る理由: 旧帯（絶対 atol 3.2e-4 = 当時の決定用 4 ケースの最悪 6.36e-5 × 5）を、検証の未見
 * ケース（今の `accept-s00192-t0600`）が 3.247e-4 で超えた。超えたのは 12,288 要素中 1 要素で、比は 6.2e-5。
 * 最大絶対差は裾の 1 要素で決まり参照の振れ幅（3.6〜6.2）ごと動くので、振れ幅で割った比の方が入力をまたいで
 * 揃う。しかも当時は決定用のケースを受入れの結果を見て足していたので、決定と受入れの独立も崩れていた。
 *
 * 決定用を 1 ケースにしない理由（実測）: t = 999 の 1 ケースで決めた帯を、t = 500 の未見ケースが 1.6 倍
 * 超えた。移植の誤りではなく入力の感度で、根拠は 2 つ:
 * ① パッチ後の eager は上流とビット一致（GPU と eager の差 = GPU と上流の差 — export 台本の `[eager]` 行）。
 * ② CPU の eager でも、中ほどの timestep は `timesteps_proj` の 1e-5 級の揺れを出力で約 4 倍に増幅する
 *    （t = 999 は 0.2 倍）。GPU の丸めの差も同じ増幅を受ける。
 * そこで決定用は timestep を参照の設定の列（999 → 60）の全域に散らし、seed と格子も変えた 6 ケースにした。
 *
 * 層の向き（計測用グラフ・各ブロックの出力の 最大絶対差 / 参照の最大絶対値・11 ケース × 30 ブロック）: 6e-6〜
 * 1.6e-4 の範囲で上下し、層数に対して単調には伸びない（ケースごとの最大は `band-s00192-t0500` の 18 層目
 * 1.54e-4・`growth-s00768-t0999` の 25 層目 1.25e-4 など。多くは後ろの層で 1e-5 級へ戻る）。S = 192 → 768
 * でも最終出力の比は伸びていない。
 */
const DIT_RATIO_BAND = 9.6e-5;

/** Wan2.1 T2V 1.3B の patch（`(1,2,2)`・潜在 16 チャネル — transformer の config）。 */
const WAN_GEOMETRY: WanPatchGeometry = {
  channels: 16,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

const SERIES_NAME = "wan2.1-t2v-1.3b-f16-dyn";
const PROBE_SERIES_NAME = "wan2.1-t2v-1.3b-f16-dyn-probe";
const COMPONENT = "transformer";
const SERIES_DIR = new URL(
  `../../../outputs/series/${SERIES_NAME}/${COMPONENT}/`,
  import.meta.url,
);
const PROBE_DIR = new URL(
  `../../../outputs/series/${PROBE_SERIES_NAME}/${COMPONENT}/`,
  import.meta.url,
);
/** 容器の中のグラフ名（表は helpers/series-graphs.ts の 1 本 — 門番と同じ正本から引く）。 */
const GRAPH = seriesGraph(SERIES_NAME, COMPONENT);
const PROBE_GRAPH = seriesGraph(PROBE_SERIES_NAME, COMPONENT);
const MODEL_FILE = "model.krm";
/** 資産の名前（`wan/export_dit.py` の `ROPE_BASE_ASSET`）。 */
const ROPE_BASE_ASSET = "rope_base";

const GENERATE = "cd tools/export-recipes && uv run --group wan --inexact python -m wan.export_dit";

type CaseRole = "band" | "accept" | "growth";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する**（生成を一部だけ流した環境でテストが
 * 黙って消える形にしない）。正本は `wan/export_dit.py` の `CASES`。
 */
const CASES: readonly { readonly name: string; readonly role: CaseRole }[] = [
  { name: "band-s00192-t0999", role: "band" },
  { name: "band-s00192-t0750", role: "band" },
  { name: "band-s00192-t0500", role: "band" },
  { name: "band-s00192-t0250", role: "band" },
  { name: "band-s00192-t0600", role: "band" },
  { name: "band-s00192-t0113", role: "band" },
  { name: "accept-s00192-t0600", role: "accept" },
  { name: "accept-s00128-t0030", role: "accept" },
  { name: "accept-s00192-t0400", role: "accept" },
  { name: "growth-s00768-t0999", role: "growth" },
  { name: "growth-s00768-t0500", role: "growth" },
];

/**
 * この IR で計画時に掛かる融合（実測 — `lastRunFusions`）。MUST: 値が動いたら赤にする（融合は
 * エクスポータのノード順 1 つで黙って外れ、値は正しいまま性能だけが変わる — 唯一の観測点）。
 *
 * - `adaln` 0: Wan の変調は `layer_norm → add(scale, one) → mul → add(shift)` で計算 4 本の並びはルールと
 *   同じだが、ルールは layer_norm の直後に reshape 2〜3 本（変調ベクトルの unsqueeze）を要求する。Wan は
 *   変調ベクトルを layer_norm の**前**に表から切り出すので、間に reshape が 0 本で掴まれない（ADR 0118
 *   決定 3 の推測どおり。ルールの拡張は perf の別起票）。
 * - `rope` 0: 融合ルールは half-split 形だけを掴む。Wan は interleave 形（実数化パッチ — 非融合の
 *   primitive 列で走る）。
 * - `silu` 2: 時刻埋め込みの MLP の SiLU 2 本（`time_embedder` の中と `act_fn`）。
 */
const EXPECTED_FUSIONS: Partial<FusionCounts> = { adaln: 0, rope: 0, silu: 2 };

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

/** ファイルの有無（NotFound 以外は伝播させる — 権限エラーを「資産が無い」に読み替えない）。 */
const filePresent = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const caseFiles = (name: string): readonly URL[] => [
  new URL(`io.${name}.safetensors`, SERIES_DIR),
  new URL(`reference.${name}.safetensors`, SERIES_DIR),
];

const expectedFiles = CASES.flatMap(({ name }) => caseFiles(name));
const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, SERIES_DIR));
const presentFiles = expectedFiles.filter(filePresent);
const ASSETS_AVAILABLE = MODEL_PRESENT && presentFiles.length === expectedFiles.length;
const ANY_PRESENT = MODEL_PRESENT || presentFiles.length > 0;
const PROBE_AVAILABLE = ASSETS_AVAILABLE && modelPresent(new URL(MODEL_FILE, PROBE_DIR));

if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_DIR.pathname} に DiT の容器と golden が揃っていないため Wan の DiT の e2e を ` +
      `SKIP する（重み 2.8GB につきリポジトリ管理外）。生成: ${GENERATE}`,
  );
} else if (!PROBE_AVAILABLE) {
  console.warn(
    `[karume] ${PROBE_DIR.pathname} に計測用の層別グラフが無いため層ごとの記録を SKIP する。` +
      `生成: ${GENERATE} --layers`,
  );
}

const viewOf = (file: SafetensorsFile, key: string, where: string) => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`${where}: '${key}' が無い`);
  return view;
};

const floatsOf = (
  file: SafetensorsFile,
  key: string,
  where: string,
): Float32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "F32") throw new Error(`${where}: '${key}' が ${view.dtype}`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** 1 ケースぶんの golden（グラフ入力・上流の潜在形と timestep と出力）。 */
type Golden = {
  readonly io: SafetensorsFile;
  readonly reference: SafetensorsFile;
  readonly latentShape: readonly number[];
  readonly timestep: number;
};

const loadGolden = async (name: string): Promise<Golden> => {
  const [ioUrl, referenceUrl] = caseFiles(name);
  const io = parseSafetensors(await readBuffer(ioUrl));
  const reference = parseSafetensors(await readBuffer(referenceUrl));
  const latents = viewOf(reference, "latents", name);
  const timestep = viewOf(reference, "timestep", name);
  if (timestep.dtype !== "I32" || latents.shape.length !== 5 || latents.shape[0] !== 1) {
    throw new Error(`${name}: reference の latents / timestep の形が想定外`);
  }
  return {
    io,
    reference,
    // ホストの潜在はバッチ軸を持たない `[C,F,H,W]`（`src/wan/dit-tokens.ts`）。
    latentShape: latents.shape.slice(1),
    timestep: new Int32Array(reference.buffer, timestep.byteOffset, 1)[0],
  };
};

/** グラフ入力（io の `input.*` — 宣言の名前と shape をそのまま使う）。 */
const graphInputs = (
  golden: Golden,
  inputs: readonly { readonly name: string }[],
  where: string,
): Record<string, Tensor> =>
  Object.fromEntries(
    inputs.map(({ name }) => {
      const view = viewOf(golden.io, `input.${name}`, where);
      return [name, {
        dtype: "f32",
        shape: view.shape,
        data: floatsOf(golden.io, `input.${name}`, where),
      }];
    }),
  );

/** 最初にビットが割れる要素の添字（一致なら -1）。長さが違えば 0。 */
const firstBitMismatch = (actual: Float32Array, expected: Float32Array): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
};

type Difference = {
  readonly maxAbs: number;
  readonly maxRel: number;
  readonly referenceMaxAbs: number;
  readonly nonFinite: number;
};

const difference = (actual: Float32Array, expected: Float32Array): Difference => {
  assertEquals(actual.length, expected.length, "要素数");
  let maxAbs = 0;
  let maxRel = 0;
  let referenceMaxAbs = 0;
  let nonFinite = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const got = actual[index];
    if (!Number.isFinite(got)) nonFinite += 1;
    const error = Math.abs(got - expected[index]);
    const magnitude = Math.abs(expected[index]);
    maxAbs = Math.max(maxAbs, error);
    if (magnitude > 0) maxRel = Math.max(maxRel, error / magnitude);
    referenceMaxAbs = Math.max(referenceMaxAbs, magnitude);
  }
  return { maxAbs, maxRel, referenceMaxAbs, nonFinite };
};

/** 帯の指標（最大絶対差 ÷ 参照の最大絶対値 — {@link DIT_RATIO_BAND}）。 */
const ratioOf = (diff: Difference): number => diff.maxAbs / diff.referenceMaxAbs;

/**
 * results.json の `tolerance` に載せる、このケースで比の帯と同値な絶対の帯（帯 × 参照の最大絶対値）。
 * 記録の形（`Measurement`）は絶対の帯しか持たないので、判定と同じ境界を絶対値へ写して残す。
 */
const recordedTolerance = (diff: Difference) => ({
  atol: DIT_RATIO_BAND * diff.referenceMaxAbs,
  rtol: 0,
});

const formatDifference = (label: string, diff: Difference): string =>
  `${label}: maxAbs ${diff.maxAbs.toExponential(3)} / 参照の最大絶対値 ` +
  `${diff.referenceMaxAbs.toFixed(3)}（比 ${ratioOf(diff).toExponential(2)}）`;

/** 出口の最終次元を `(pt,ph,pw,c)` → `(c,pt,ph,pw)` と読み替える（unpatchify の並びの故障注入）。 */
const transposeTokenAxes = (tokens: Float32Array, geometry: WanPatchGeometry): Float32Array => {
  const channels = geometry.channels;
  const width = wanTokenWidth(geometry);
  const volume = width / channels;
  const out = new Float32Array(tokens.length);
  for (let token = 0; token < tokens.length / width; token += 1) {
    for (let c = 0; c < channels; c += 1) {
      for (let v = 0; v < volume; v += 1) {
        out[token * width + v * channels + c] = tokens[token * width + c * volume + v];
      }
    }
  }
  return out;
};

/** 容器の資産 `rope_base`（役割 `rope-base`）を読んで素表にする。 */
const readRopeBase = async (opened: OpenedContainer): Promise<WanRopeBase> => {
  const reader = opened.asset(ROPE_BASE_ASSET);
  assertEquals(reader.role, "rope-base", `資産 '${ROPE_BASE_ASSET}' の役割`);
  const bytes = await reader.read(0, reader.length);
  return parseWanRopeBase(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
};

const results = openResults("wan-dit");

Deno.test({
  name: "Wan DiT 資産: 容器と全ケースの golden が揃っている",
  // 完全に空の環境だけ「生成していない」として SKIP。何か 1 つでもあれば欠けは FAIL。
  ignore: !ANY_PRESENT,
  fn: () => {
    assert(MODEL_PRESENT, `${SERIES_DIR.pathname}${MODEL_FILE} が無い`);
    assertEquals(
      expectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `golden の欠け（生成: ${GENERATE}）`,
    );
  },
});

Deno.test({
  name: "Wan DiT ホスト: patchify と資産 rope_base からの RoPE 表が golden の入力とビット一致する",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    const base = await readRopeBase(await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR)));
    for (const { name } of CASES) {
      const golden = await loadGolden(name);
      const tokens = patchifyLatents(
        floatsOf(golden.reference, "latents", name),
        golden.latentShape,
        WAN_GEOMETRY,
      );
      assertEquals(
        firstBitMismatch(tokens, floatsOf(golden.io, "input.tokens", name)),
        -1,
        `${name}: patchify`,
      );
      const tables = wanRopeTables(base, wanTokenGrid(golden.latentShape, WAN_GEOMETRY));
      assertEquals(
        firstBitMismatch(tables.cos, floatsOf(golden.io, "input.rope_cos", name)),
        -1,
        `${name}: rope_cos`,
      );
      assertEquals(
        firstBitMismatch(tables.sin, floatsOf(golden.io, "input.rope_sin", name)),
        -1,
        `${name}: rope_sin`,
      );
    }
  },
});

Deno.test({
  name: "Wan DiT 照合（実 GPU / diffusers CPU f32）: S 形の 1 forward が帯の内・故障注入は帯の外",
  ignore: !ASSETS_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR));
    const prepared = prepareContainer(opened, GRAPH);
    assertEquals(prepared.graph.outputs.length, 1, "製品のグラフの出力は 1 本");
    const base = await readRopeBase(opened);
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
      const session = await prepared.createContainerSession(gpu);
      try {
        const run = async (inputs: Record<string, Tensor>): Promise<Float32Array> => {
          const outputs = await session.run(inputs);
          const tensor = outputs[prepared.graph.outputs[0]];
          if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
          return tensor.data;
        };
        for (const { name, role } of CASES) {
          await t.step(`${name}（${role}）`, async () => {
            await runRecordedCase(results, { id: name }, async ({ measurements }) => {
              const golden = await loadGolden(name);
              const inputs = graphInputs(golden, prepared.graph.inputs, name);
              const expected = floatsOf(golden.reference, "output", name);
              const tokens = await run(inputs);
              const latents = unpatchifyTokens(tokens, golden.latentShape, WAN_GEOMETRY);
              const diff = difference(latents, expected);
              measurements.push({
                output: "latents",
                maxAbs: diff.maxAbs,
                maxRel: diff.maxRel,
                tolerance: recordedTolerance(diff),
                stage: "karume",
              });
              const notes = [formatDifference("io の入力", diff)];
              assertEquals(diff.nonFinite, 0, `${name}: 非有限`);
              assertEquals(
                { ...session.diagnostics().lastRunFusions, ...EXPECTED_FUSIONS },
                session.diagnostics().lastRunFusions,
                `${name}: 融合の件数（adaln / rope / silu）`,
              );

              // ホストの timesteps_proj（TS）で回した値（製品の経路の入力）も同じ比の帯で見る。入力そのものの
              // 差（torch と TS の `exp` の 1 ULP — 最悪 3.05e-5）は `wan_dit_host_test.ts` の絶対 atol 1.5e-4 が
              // 押さえる（値域が [-1, 1] の sin / cos なので絶対で足りる — 帯の指標を比に変えても据え置く）。
              const width = viewOf(golden.io, "input.timesteps_proj", name).shape[1];
              const hostTimestep = unpatchifyTokens(
                await run({
                  ...inputs,
                  timesteps_proj: {
                    dtype: "f32",
                    shape: [1, width],
                    data: timestepsProj(golden.timestep, width),
                  },
                }),
                golden.latentShape,
                WAN_GEOMETRY,
              );
              const hostDiff = difference(hostTimestep, expected);
              measurements.push({
                output: "latents@host-timesteps-proj",
                maxAbs: hostDiff.maxAbs,
                maxRel: hostDiff.maxRel,
                tolerance: recordedTolerance(hostDiff),
                stage: "karume",
              });
              notes.push(formatDifference("TS の timesteps_proj", hostDiff));
              const faults = role === "accept"
                ? await faultInjections(golden, inputs, tokens, expected, run, base)
                : [];
              notes.push(...faults.map(({ note }) => note));
              // 判定の前に出す（帯の外へ出た回も全ケースの比が手元に残る）。
              console.log(`[wan-dit] ${name}: ${notes.join(" / ")}`);

              if (role !== "growth") {
                assert(
                  ratioOf(diff) <= DIT_RATIO_BAND,
                  `${name}: ${notes[0]} が帯 ${DIT_RATIO_BAND} の外`,
                );
                assert(
                  ratioOf(hostDiff) <= DIT_RATIO_BAND,
                  `${name}: ${notes[1]} が帯 ${DIT_RATIO_BAND} の外`,
                );
              }
              for (const { label, ratio } of faults) {
                assert(
                  ratio > DIT_RATIO_BAND,
                  `${name}: 故障注入 ${label} が帯 ${DIT_RATIO_BAND} の内に収まった`,
                );
              }
              return { status: "pass", note: notes.join(" / ") };
            });
          });
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

/** 故障注入 1 件の結果（判定は呼び手が全ケースの比を出してから行う）。 */
type FaultResult = { readonly label: string; readonly ratio: number; readonly note: string };

/**
 * 受入れケースの故障注入 4 件（追記 2026-10-02 — 帯が広すぎないことの裏取り）。どれも帯の外へ出ることを
 * 門にする（赤にならない注入があれば帯の決め方を見直す）。
 *
 * - RoPE の h / w の取り違え（非正方の格子で表を組み違える — 正方では値が一致して見えない）
 * - unpatchify の並びの取り違え（出口を入口の並び `(c,pt,ph,pw)` で読む）
 * - timestep の cos / sin の前後反転
 * - **timestep の 1 ずれ**（`timesteps_proj` を t + 1 で組む — TS の `timestepsProj`）。上の 3 件は O(1) の差で
 *   帯がどこにあっても赤になるが、これは比 2e-3〜9e-3 の微妙な故障で、帯の上限側の根拠になる。
 */
const faultInjections = async (
  golden: Golden,
  inputs: Record<string, Tensor>,
  tokens: Float32Array,
  expected: Float32Array,
  run: (inputs: Record<string, Tensor>) => Promise<Float32Array>,
  base: WanRopeBase,
): Promise<FaultResult[]> => {
  const grid = wanTokenGrid(golden.latentShape, WAN_GEOMETRY);
  const unpatchify = (data: Float32Array) =>
    unpatchifyTokens(data, golden.latentShape, WAN_GEOMETRY);
  const results: FaultResult[] = [];
  const measure = (label: string, actual: Float32Array) => {
    const diff = difference(actual, expected);
    const ratio = ratioOf(diff);
    results.push({
      label,
      ratio,
      note: `故障注入 ${label}: maxAbs ${diff.maxAbs.toExponential(3)}（比 ${
        ratio.toExponential(2)
      }）`,
    });
  };

  if (grid.rows !== grid.cols) {
    const swapped = wanRopeTables(base, { ...grid, rows: grid.cols, cols: grid.rows });
    const shape = inputs.rope_cos.shape;
    measure(
      "RoPE の h / w 取り違え",
      unpatchify(
        await run({
          ...inputs,
          rope_cos: { dtype: "f32", shape, data: swapped.cos },
          rope_sin: { dtype: "f32", shape, data: swapped.sin },
        }),
      ),
    );
  }
  measure("unpatchify の並び", unpatchify(transposeTokenAxes(tokens, WAN_GEOMETRY)));
  const proj = inputs.timesteps_proj;
  if (proj.dtype !== "f32") throw new Error(`timesteps_proj が ${proj.dtype}`);
  const half = proj.data.length / 2;
  const flipped = Float32Array.from(
    { length: proj.data.length },
    (_, index) => proj.data[index < half ? half + index : index - half],
  );
  measure(
    "timestep の cos / sin 反転",
    unpatchify(
      await run({ ...inputs, timesteps_proj: { dtype: "f32", shape: proj.shape, data: flipped } }),
    ),
  );
  measure(
    "timestep の 1 ずれ",
    unpatchify(
      await run({
        ...inputs,
        timesteps_proj: {
          dtype: "f32",
          shape: proj.shape,
          data: timestepsProj(golden.timestep + 1, proj.shape[1]),
        },
      }),
    ),
  );
  return results;
};

Deno.test({
  name: "Wan DiT 層別（実 GPU・計測用グラフ）: 各ブロックの出力の差を記録する（門ではない）",
  ignore: !PROBE_AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    const prepared = prepareContainer(
      await openSeriesContainer(new URL(MODEL_FILE, PROBE_DIR)),
      PROBE_GRAPH,
    );
    const outputs = prepared.graph.outputs;
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
      const session = await prepared.createContainerSession(gpu);
      try {
        for (const { name } of CASES) {
          await runRecordedCase(results, { id: `${name}/layers` }, async ({ measurements }) => {
            const golden = await loadGolden(name);
            const produced = await session.run(graphInputs(golden, prepared.graph.inputs, name));
            const blocks = outputs.length - 1;
            const rows: string[] = [];
            for (let index = 0; index < blocks; index += 1) {
              const tensor = produced[outputs[index]];
              if (tensor.dtype !== "f32") throw new Error(`出力 ${index} が ${tensor.dtype}`);
              const key = `block.${String(index).padStart(2, "0")}`;
              const diff = difference(tensor.data, floatsOf(golden.reference, key, name));
              assertEquals(diff.nonFinite, 0, `${name}: ${key} が非有限`);
              measurements.push({
                output: key,
                maxAbs: diff.maxAbs,
                maxRel: diff.maxRel,
                tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
                stage: "karume",
              });
              rows.push(
                `${index}:${(diff.maxAbs / diff.referenceMaxAbs).toExponential(2)}`,
              );
            }
            console.log(
              `[wan-dit] ${name} 層ごとの比（最大絶対差 / 参照の最大絶対値）: ${rows.join(" ")}`,
            );
            return { status: "pass", note: rows.join(" ") };
          });
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});
