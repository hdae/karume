/**
 * Wan2.2 TI2V-5B の DiT（I2V 対応の S 形グラフ・i8 系列）の**移植の門**（実 GPU・ADR 0121 段 2 — 決定 2 / 4 / 5 / 8）。
 *
 * 参照席 `f16+dit8`（重み i8・計算 f32・`session` 空）で系列の容器（`outputs/series/wan2.2-ti2v-5b-i8-dyn/transformer/`）を
 * 回し、ホストの unpatchify（`src/wan/dit-tokens.ts` — 潜在 48 チャネルでも 2.1 の関数のまま）を通した潜在を、同じ i8 の
 * fake-quant 重みで採った上流の CPU 参照（`reference.<case>` の `output.f64`〈活性も f64〉と `output`〈f32〉— 時刻は決定 4 の
 * 形: T2V は 1 次元 timestep の経路・I2V は時刻の MLP を値ごとに M = 1 で回す参照ラッパ）と突き合わせる。前例は Wan2.1 の
 * DiT の e2e（`e2e_wan_dit_test.ts`）で、指標・帯の導出・故障注入・1 submit の門・確保の記録の形はそれに揃える。
 *
 * golden の生成は `tools/export-recipes/wan/ti2v_export_dit.py`（S = 192 は `write`・実寸は `write-full` — 中身とケースの表は
 * 同ファイルの docstring・TS 側の表は `helpers/wan-ti2v-dit.ts`）。パッチ後の eager は全ケースで上流とビット一致する
 * （書き手の eager 同値の門 — 落ちたケースは golden にならない）ので、ここで見る差はそのまま「GPU で回した IR」と「上流」の差。
 *
 * ## 指標と帯（決定 5 — ADR 0118 決定 8・ADR 0120 決定 3 の r 門）
 *
 *   r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷ max|f64|)
 *
 * = GPU の誤差が、同じ入力で CPU f32 の参照自身が出す誤差の何倍か（{@link normalizedRatio}）。帯は組ごとに、その組の決定用
 * 6 本（T2V 3 本 + I2V 3 本で固定）の最悪 r × 5 で独立に導く（組どうし・2.1 の帯は持ち込まない）:
 *
 * - **S = 192 の組**（{@link SMALL_BAND} — 決定用 6 本・受入れ T2V 2 + I2V 2）
 * - **実寸の組**（{@link FULL_BAND} — 2 つの形 = 潜在 `[48,21,30,52]`〈S = 8,190〉と `[48,9,44,80]`〈S = 7,920〉。決定用は
 *   形ごとに 3 本で 2 つの形を合わせて 6 本・受入れは形ごとに T2V 2 + I2V 2 の 8 本）。2 つの形は S が 3% しか違わない
 *   （ADR 0121 追記「裁定 1 の確定」— 開発機に入るトークン数の予算の 2 通りの配分）ので帯は 1 本にし、形ごとの最悪も記録する
 *   （{@link recordWorst}）。
 *
 * `undefined` = 未導出: 各ケースは r を出して赤で止まり、組の最後の step が帯の候補（決定用の最悪 r × 5）を出す
 * （{@link bandCandidate} — 通る値を仮置きして検出力の無い門を作らない）。最初の実走が導いた値を、実測の表と一緒に定数へ書く。
 * MUST: 受入れの結果を見て帯も決定用のケースも指標も変えない。受入れが帯を外れたら、帯を広げずに原因を調べる。
 *
 * ## 故障注入（ADR 0121 検収の段 2 の 4 件 — 受入れのケースで・どれも帯の外が門）
 *
 * - **timestep +1**（生成側の `timesteps_proj` を t + 1 で組む）: T2V と I2V。微妙な注入で、帯の {@link SUBTLE_FAULT_MARGIN}
 *   倍以上離れることも門にする（帯が広すぎる兆候の検出 — 2.1 の床と同じ 2）
 * - **`to_v` の per-channel scale × 2**（{@link SCALE_FAULT_WEIGHT} — 容器の読み口で scale の写しを 2 倍・別の Session）:
 *   T2V と I2V
 * - **条件マスクの 1 トークンずれ**（真の区間を先頭の P トークンから 1..P へ）: I2V だけ
 * - **時刻入力 2 本の取り違え**（`timesteps_proj` と `timesteps_proj_condition` を入れ替える）: I2V だけ
 *
 * T2V では時刻入力 2 本が同じ値でマスクが全て偽なので、後ろの 2 件は原理的に値に出ない（ADR 0121 検収の段 2）。
 *
 * 受入れは加えて、製品の経路の入力（ホストの `timesteps_proj` 2 本とテストの中のホストの条件マスク —
 * `helpers/wan-ti2v-dit.ts` の `conditionMask`）でも帯の内であることを見る。
 *
 * ## 実寸の計測と記録（ADR 0121 検収の段 2）
 *
 * - **計測モード**（`gpuTiming`）の照合: S ごとに Session を張り直し（2 つの S の backing を同時に載せない）、行ブロックの
 *   枚数（device の束縛上限から導いた期待 — {@link expectedRowBlocks}・B570 で S = 8,190 / 7,920 とも f32 3 枚）と、1 submit
 *   ごとの GPU 時間の最大 ≤ {@link SUBMIT_GPU_LIMIT_MS}（最初の run の裏付け前のチャンクを含む）を門にする。
 * - **通常モード**（計測なし）: S ごとに所要（壁時計）・diag（run の直後に生きている確保 = 重み + その run のアリーナ +
 *   slot backing — 2.1 の「診断の確保」と同じ量）・fdinfo の VRAM の山を記録し、出力が計測モードの同じケースと Uint32 で
 *   一致することを門にする。diag は容量の閾値 {@link DIAG_THRESHOLD_BYTES}（決定 8）と並べ、S = 8,190 では容量の表の外挿
 *   （2.1 型のグラフ — {@link EXTRAPOLATED_DIAG_GIB}）との差を I2V 対応の増分の見込み（決定 3 の +0.19 / 悪い側 +0.56 GiB）と
 *   並べて記録する。閾値を超えても赤にはしない（超えたときの手 = 受理するフレーム数の上限を下げるのは、段 6 の前に決める
 *   裁定 — ここで落としても直せない）。超えた回は警告を出し、記録に「超えた」と書く。
 * - **実用席 `f16+dit8-a8-attn8-s16` の diag**（記録だけ — 実用席の門は段 7）: 同じ 2 つの形で、同じ閾値と外挿と並べる。
 * - **S = 12,090**（832×480・121 フレーム — 決定 8 が受理を広げるかを段 2 の実測で決める形）: 両席の diag と完走の可否を
 *   記録する。golden を持たない（入力は乱数の潜在と実プロンプトの埋め込み）ので値の門は無く、非有限 0 と device lost が
 *   無いことだけを見る。確保が最大の形なので opt-in（`KARUME_WAN_TI2V_LONG_PROBE=1`）にして、別の回で回す。
 *
 * 資産が無い環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。golden の完全性（一部だけある環境は
 * FAIL）は GPU 不要のホストテスト（`wan_ti2v_dit_host_test.ts`）が組ごとに見る。
 */

import { assert, assertEquals } from "@std/assert";
import type { SessionSpec } from "@karume/hub";
import {
  acquireGpu,
  type BoundContainer,
  type FusionCounts,
  type GpuContext,
  type OpenedContainer,
  prepareContainer,
  type PreparedModel,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanTokenGrid,
  wanTokenGrid,
} from "../src/wan/dit-tokens.ts";
import { type WanRopeBase, wanRopeTables } from "../src/wan/dit-rope.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { effectiveSessionOptions } from "./helpers/census-table.ts";
import { formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { timestampUnitNs } from "./helpers/timestamp-unit.ts";
import {
  caseFiles,
  conditionMask,
  filePresent,
  firstBitMismatch,
  floatsOf,
  isAccept,
  loadTi2vGolden,
  readRopeBase,
  type Ti2vCase,
  type Ti2vForm,
  type Ti2vFullTokens,
  type Ti2vGolden,
  viewOf,
  WAN22_GEOMETRY,
  WAN_TI2V_CASES,
  WAN_TI2V_COMPONENT,
  WAN_TI2V_FULL_CASES,
  WAN_TI2V_GENERATE,
  WAN_TI2V_GENERATE_FULL,
  WAN_TI2V_MODEL_FILE,
  WAN_TI2V_SERIES_DIR,
  WAN_TI2V_SERIES_NAME,
} from "./helpers/wan-ti2v-dit.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import {
  assertAdapterMatchesEnvironment,
  ENVIRONMENT,
} from "../../runtime/tests/helpers/environment.ts";
import {
  type Measurement,
  openResults,
  runRecordedCase,
} from "../../runtime/tests/helpers/results.ts";

/**
 * S = 192 の組（決定用 6 本・受入れ 4 本）の r の帯（参照席 `f16+dit8`）。`undefined` = 未導出（モジュール doc「指標と帯」）。
 *
 * MUST: 受入れの結果を見てこの値も決定用のケースも変えない。
 */
const SMALL_BAND: number | undefined = undefined;

/**
 * 実寸の組（S = 8,190 / 7,920 の 2 つの形・決定用 6 本・受入れ 8 本）の r の帯（参照席 `f16+dit8`）。`undefined` = 未導出。
 *
 * MUST: 受入れの結果を見てこの値も決定用のケースも変えない。
 */
const FULL_BAND: number | undefined = undefined;

/**
 * 微妙な故障（timestep +1）が帯から離れているべき倍率（ADR 0121 検収の段 2「帯の床（2.1 は 2）以上」— 2.1 の
 * `SUBTLE_FAULT_MARGIN` と同じ値・同じ理由）。
 */
const SUBTLE_FAULT_MARGIN = 2;

/**
 * 1 submit の GPU 実行の幅の上限（ADR 0118 決定 6 と追記 2026-10-02 — Linux xe の `job_timeout_ms` 5,000 ms の 1/5。門は
 * 1 本ずつの最大で、最初の run の裏付け前のチャンクも含む）。
 */
const SUBMIT_GPU_LIMIT_MS = 1000;

/** Wan2.2 TI2V-5B の attention の head 数（transformer の config `num_attention_heads`）。 */
const WAN22_HEADS = 24;
/** スコア行列 S の 1 要素の格納幅（参照席 = 既定の f32 格納）。 */
const SCORE_BYTES = 4;

/**
 * 実寸の self-attention の行ブロック枚数の期待値（ADR 0118 決定 6 の式 — device の束縛上限だけから runtime と独立に導く）:
 *
 *   1 枚の行数 = ⌊上限 ÷ (heads × S × 4 B)⌋・枚数 = ⌈S ÷ 1 枚の行数⌉
 *
 * B570 では S = 8,190 / 7,920 が 3 枚・S = 12,090 が 7 枚（差分調査 §3.2 の表 — 下の GPU 不要のテストが再現する）。
 */
const expectedRowBlocks = (tokens: number, bindingLimit: number): number =>
  Math.ceil(tokens / Math.floor(bindingLimit / (WAN22_HEADS * tokens * SCORE_BYTES)));

/** B570 の `maxStorageBufferBindingSize`（ADR 0118 決定 6 の表の前提）。 */
const B570_STORAGE_BINDING_LIMIT = 2_147_483_644;

/**
 * この IR で計画時に掛かる融合。MUST: 値が動いたら赤にする（融合はエクスポータのノード順 1 つで黙って外れる — 唯一の観測点）。
 *
 * - `adaln` 0 / `rope` 0: 2.1 と同じ理由（変調ベクトルを layer_norm の前に切り出す・RoPE は interleave 形）
 * - `silu` 4: 時刻の MLP の SiLU 2 本（`time_embedder` の中と `act_fn`）を M = 1 で 2 回回す（決定 3）ので 2.1 の 2 本の倍。
 *   IR で `sigmoid` の直後に、その出力を受ける `mul` が来る並び（ルール `silu` の形）を GPU なしで数えた値（4 本）で、
 *   最初の GPU の実走で確かめる
 */
const EXPECTED_FUSIONS: Partial<FusionCounts> = { adaln: 0, rope: 0, silu: 4 };

/**
 * scale × 2 の故障注入の linear（中ほどのブロックの self-attention の V — q / k は直後の RMS norm が倍率を打ち消して値に
 * 出ない。2.1 の `I8_SCALE_FAULT_WEIGHT` と同じ選び方・5B は 30 層）。
 */
const SCALE_FAULT_WEIGHT = "blocks.15.attn1.to_v.weight";

/** 参照席（ADR 0121 決定 2 — `session` 空）と実用席。 */
const REFERENCE_QUANT = "f16+dit8";
const PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";
type Seat = typeof REFERENCE_QUANT | typeof PRACTICAL_QUANT;

/**
 * 実用席の束（ADR 0121 決定 2 の表 — `linearCompute: "a8"`・`attentionCompute: "a8"`・`attentionScoreStorage: "f16"`）。
 *
 * NOTE: 束の正本は配布形の manifest の quant 席（ADR 0110 決定 1）だが、2.2 の配布形は段 8 まで無い。それまでは ADR の宣言を
 * ここに書き、家族の受理表（`WAN_SESSION_POLICY` — 決定 2 の「受理表の 3 キーは ADR 0120 と同じ」）に通す。段 8 で manifest
 * から引く形に替える。
 */
const PRACTICAL_SESSION: SessionSpec = {
  linearCompute: "a8",
  attentionCompute: "a8",
  attentionScoreStorage: "f16",
};

const practicalSessionOptions = (): SessionOptions =>
  effectiveSessionOptions("wan", PRACTICAL_SESSION, {}, `wan-ti2v ti2v-5b/${PRACTICAL_QUANT}`);

const GIB = 2 ** 30;

/**
 * 容量の閾値（ADR 0121 決定 8 — S = 8,190 の diag がどちらかの席で超えたら、受理するフレーム数の上限を下げる手を段 6 の前に
 * 確定する。追記「裁定 1 の確定」で 1280×704 系の 33 フレームにも同じ規則を当てる）。
 */
const DIAG_THRESHOLD_BYTES = 8 * GIB;

/** 容量の表の外挿を持つ S（832×480・81 フレーム）。 */
const EXTRAPOLATED_TOKENS = 8190;

/**
 * 容量の表の外挿（ADR 0121 容量の表の注 — 2.1 の k を 2 点から線形に外挿した S = 8,190 の diag・**2.1 型のグラフ**〈時刻入力
 * 1 本〉の値で I2V 対応の増分を含まない）。実測との差を、I2V 対応の増分の見込み（{@link I2V_INCREMENT_GIB}）と並べる。
 */
const EXTRAPOLATED_DIAG_GIB: Readonly<Record<Seat, number>> = {
  [REFERENCE_QUANT]: 7.57,
  [PRACTICAL_QUANT]: 7.71,
};

/**
 * I2V 対応の増分の見込み（決定 3 — `where` を消費の直前に置けたときの S = 8,190 の +0.19 GiB と、上流の順のまま trace された
 * ときの悪い側 +0.56 GiB。段 1 の IR の検査で `where` は全て直後の 1 ノードが消費 = 見込みは前者）。
 */
const I2V_INCREMENT_GIB = { placed: 0.19, hoisted: 0.56 } as const;

/** S = 12,090 の opt-in（モジュール doc「実寸の計測と記録」）。 */
const LONG_PROBE_ENV = "KARUME_WAN_TI2V_LONG_PROBE";
const LONG_PROBE = Deno.env.get(LONG_PROBE_ENV) === "1";
/** 832×480・121 フレーム（潜在 `[48,31,30,52]` → S = 31·15·26 = 12,090）。 */
const LONG_LATENT_SHAPE: readonly number[] = [48, 31, 30, 52];
const LONG_TOKENS = 12090;
/** S = 12,090 の入力の文脈（実プロンプトの埋め込み — S = 192 の golden の io から借りる・合成の乱数は使わない）。 */
const LONG_CONTEXT_CASE = "band-t2v-s00192-t0999";
/** S = 12,090 の生成側の timestep（生成の最初のステップ）と乱数の潜在の seed。 */
const LONG_TIMESTEP = 999;
const LONG_SEED = 20262121;

const MODEL_URL = new URL(WAN_TI2V_MODEL_FILE, WAN_TI2V_SERIES_DIR);
const MODEL_PRESENT = modelPresent(MODEL_URL);
const SMALL_AVAILABLE = MODEL_PRESENT &&
  WAN_TI2V_CASES.every(({ name }) => caseFiles(name).every(filePresent));
const FULL_AVAILABLE = MODEL_PRESENT &&
  WAN_TI2V_FULL_CASES.every(({ name }) => caseFiles(name).every(filePresent));

if (!SMALL_AVAILABLE) {
  console.warn(
    `[karume] ${WAN_TI2V_SERIES_DIR.pathname} に Wan2.2 TI2V の DiT の容器と S = 192 の golden が揃っていないため、` +
      `S = 192 の r 門を SKIP する（重み 4.7GB につきリポジトリ管理外）。生成: ${WAN_TI2V_GENERATE}`,
  );
}
if (!FULL_AVAILABLE) {
  console.warn(
    `[karume] ${WAN_TI2V_SERIES_DIR.pathname} に Wan2.2 TI2V の実寸の golden が揃っていないため、実寸の r 門と` +
      `確保の記録を SKIP する。生成（CPU・数時間・途中から再開できる）: ${WAN_TI2V_GENERATE_FULL}`,
  );
}
if (!LONG_PROBE) {
  console.warn(
    `[karume] S = 12,090（832×480・121 フレーム）の diag の記録は opt-in のため SKIP する（${LONG_PROBE_ENV}=1 で回す）`,
  );
}

/**
 * 容器の中のグラフ名（表は helpers/series-graphs.ts の 1 本）。**使うときに引く** — 表に行が無いときにモジュール評価で落とすと、
 * 資産の無い機の SKIP まで巻き込む。
 */
const graphName = (): string => seriesGraph(WAN_TI2V_SERIES_NAME, WAN_TI2V_COMPONENT);

/** 実寸の S の順（確保の大きい S = 8,190 が先 — B570 の `destroy()` の解放遅れで後ろの大きい確保が OOM になった前例）。 */
const FULL_TOKENS_ORDER: readonly Ti2vFullTokens[] = [8190, 7920];

/** S ごとのケース（{@link FULL_TOKENS_ORDER} の順）。`normalModeCase` は通常モードで回す 1 本（S ごとの先頭）。 */
const FULL_GROUPS = FULL_TOKENS_ORDER.map((tokens) => {
  const cases = WAN_TI2V_FULL_CASES.filter((spec) => spec.tokens === tokens);
  return { tokens, cases, normalModeCase: cases[0].name };
});

const f32Tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

const boolTensor = (shape: readonly number[], data: Uint32Array<ArrayBuffer>): Tensor => ({
  dtype: "bool",
  shape,
  data,
});

/** グラフ入力（io の `input.*` — 宣言の名前・dtype・shape をそのまま使う。条件マスクは bool = u32 の 0 / 1）。 */
const graphInputs = (
  golden: Ti2vGolden,
  inputs: readonly { readonly name: string; readonly dtype: string }[],
  where: string,
): Record<string, Tensor> =>
  Object.fromEntries(
    inputs.map(({ name, dtype }) => {
      const key = `input.${name}`;
      const view = viewOf(golden.io, key, where);
      if (dtype === "bool") {
        if (view.dtype !== "U32") {
          throw new Error(`${where}: '${key}' が ${view.dtype}（U32 のはず）`);
        }
        return [
          name,
          boolTensor(
            view.shape,
            new Uint32Array(golden.io.buffer, view.byteOffset, view.byteLength / 4),
          ),
        ];
      }
      if (dtype !== "f32") throw new Error(`${where}: 入力 '${name}' の dtype ${dtype} は想定外`);
      return [name, f32Tensor(view.shape, floatsOf(golden.io, key, where))];
    }),
  );

/** 入力の 1 本（宣言にあるはずの名前 — 無ければ落とす）。 */
const inputOf = (inputs: Readonly<Record<string, Tensor>>, name: string): Tensor => {
  const tensor = inputs[name];
  if (tensor === undefined) throw new Error(`グラフ入力 '${name}' が無い`);
  return tensor;
};

/** 時刻入力の幅（`timesteps_proj` の静的次元 — 256 を書かない）。 */
const timestepWidth = (inputs: Readonly<Record<string, Tensor>>): number =>
  inputOf(inputs, "timesteps_proj").shape[1];

/**
 * 製品の経路の入力（ホストの `timesteps_proj` 2 本とテストの中のホストの条件マスク）に差し替えた入力。golden の io の入力との
 * 差は時刻の sinusoid の 1 ULP 級だけ（条件マスクはビット一致 — ホストテスト）。
 */
const hostInputs = (
  golden: Ti2vGolden,
  inputs: Readonly<Record<string, Tensor>>,
  grid: WanTokenGrid,
  form: Ti2vForm,
): Record<string, Tensor> => {
  const width = timestepWidth(inputs);
  return {
    ...inputs,
    timesteps_proj: f32Tensor([1, width], timestepsProj(golden.timestep, width)),
    timesteps_proj_condition: f32Tensor([1, width], timestepsProj(golden.conditionTimestep, width)),
    condition_mask: boolTensor(inputOf(inputs, "condition_mask").shape, conditionMask(grid, form)),
  };
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

/** 比（最大絶対差 ÷ 参照の最大絶対値）。 */
const ratioOf = (diff: Difference): number => diff.maxAbs / diff.referenceMaxAbs;

/** f64 の参照のキー（`wan/export_dit.py` の `REFERENCE_F64_KEY` — 活性も f64 で回した上流の出力を f32 へ丸めた値）。 */
const REFERENCE_F64_KEY = "output.f64";

/**
 * 正規化の分母の元 = CPU f32 の参照（`output`）と f64 の参照（`output.f64`）の差。比が 0（f32 の参照が f64 とビット一致）
 * だと割れないので fail loudly にする。
 */
const referenceErrorOf = (golden: Ti2vGolden, name: string): Difference => {
  const diff = difference(
    floatsOf(golden.reference, "output", name),
    floatsOf(golden.reference, REFERENCE_F64_KEY, name),
  );
  if (!(ratioOf(diff) > 0)) {
    throw new Error(
      `${name}: CPU f32 の参照の f64 に対する比が ${ratioOf(diff)}（正規化できない）`,
    );
  }
  return diff;
};

/** 正規化した比 r = GPU の f64 に対する比 ÷ CPU f32 の参照の f64 に対する比（`referenceRatio`）。 */
const normalizedRatio = (diff: Difference, referenceRatio: number): number =>
  ratioOf(diff) / referenceRatio;

/**
 * results.json の `tolerance` に載せる、このケースで r の帯と同値な絶対の帯（帯 × 分母 × 参照の最大絶対値）。帯が未導出の回は
 * 無限の帯として記録する（判定は赤にする）。
 */
const toleranceOf = (diff: Difference, band: number | undefined, referenceRatio: number) =>
  band === undefined
    ? { atol: Number.POSITIVE_INFINITY, rtol: 0 }
    : { atol: band * referenceRatio * diff.referenceMaxAbs, rtol: 0 };

const formatDifference = (label: string, diff: Difference): string =>
  `${label}: maxAbs ${diff.maxAbs.toExponential(3)} / 参照の最大絶対値 ` +
  `${diff.referenceMaxAbs.toFixed(3)}（比 ${ratioOf(diff).toExponential(2)}）`;

const formatNormalized = (label: string, diff: Difference, referenceRatio: number): string =>
  `${formatDifference(label, diff)}・r ${normalizedRatio(diff, referenceRatio).toPrecision(3)}`;

/** p99.99 ベースの r の分位（記録だけ — 2.1 と同じ）。 */
const RECORDED_QUANTILE = 0.9999;

/** |actual − expected| の q 分位（最近順位法 — 昇順に並べた ceil(q·n) 番目）。 */
const absErrorQuantile = (actual: Float32Array, expected: Float32Array, q: number): number => {
  assertEquals(actual.length, expected.length, "要素数");
  const errors = new Float64Array(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    errors[index] = Math.abs(actual[index] - expected[index]);
  }
  errors.sort();
  return errors[Math.max(0, Math.ceil(q * errors.length) - 1)];
};

/** p99.99 ベースの r の記録の 1 行（門は max ベースの r — 2.1 の `quantileRatio` と同じ形）。 */
const quantileRatio = (
  label: string,
  actual: Float32Array,
  expected: Float32Array,
  referenceQuantile: number,
): string => {
  const numerator = absErrorQuantile(actual, expected, RECORDED_QUANTILE);
  return `${label}: p99.99 ベースの r ${
    (numerator / referenceQuantile).toPrecision(3)
  }（記録だけ・分子 ${numerator.toExponential(3)} / 分母 ${referenceQuantile.toExponential(3)}）`;
};

const gib = (bytes: number): string => `${(bytes / GIB).toFixed(2)} GiB`;

/** 1 run の観測（所要と生きている確保 — 2.1 の `FullRun` と同じ量）。 */
type Run = {
  readonly label: string;
  readonly wallMs: number;
  /** dispatch の pass の GPU 時間の合計（timestamp の生の単位）。計測が無効な device では undefined。 */
  readonly gpuTicks: number | undefined;
  /**
   * run の直後に生きている確保（diag）= 重み + その run のアリーナ（中間・入力・readback staging）+ 保持中の slot backing
   * （中間と入力）。1 本目は中間がアリーナに、2 本目以降は slot backing に載る。
   */
  readonly liveBytes: number;
  readonly weightBytes: number;
  /** そのうちの slot backing（中間と入力 — 2 本目以降）。 */
  readonly backingBytes: number;
  /** そのうちの readback staging（グラフ出力のバイト数の和）。 */
  readonly stagingBytes: number;
};

const observeRun = (
  label: string,
  diagnostics: SessionDiagnostics,
  wallMs: number,
  stagingBytes: number,
): Run => {
  const backingBytes = diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes;
  return {
    label,
    wallMs,
    gpuTicks: diagnostics.lastRunTiming?.totalNs,
    liveBytes: diagnostics.weights.allocatedBytes + (diagnostics.lastRun?.allocatedBytes ?? 0) +
      backingBytes,
    weightBytes: diagnostics.weights.allocatedBytes,
    backingBytes,
    stagingBytes,
  };
};

/** `unitNs` は timestamp の 1 単位の ns（省くと GPU 時間を出さない — 計測なしの run）。 */
const formatRun = (run: Run, unitNs?: number): string =>
  `${run.label}: 壁 ${(run.wallMs / 1000).toFixed(2)} s` +
  (run.gpuTicks === undefined || unitNs === undefined
    ? ""
    : `・GPU ${(run.gpuTicks * unitNs / 1e9).toFixed(2)} s`) +
  `・確保 ${gib(run.liveBytes)}（うち重み ${gib(run.weightBytes)}・slot backing ${
    gib(run.backingBytes)
  }・readback staging ${gib(run.stagingBytes)}）`;

/** グラフ出力のバイト数の和（readback staging の大きさ）。 */
const outputBytes = (outputs: Readonly<Record<string, Tensor>>): number =>
  Object.values(outputs).reduce((total, tensor) => total + tensor.data.byteLength, 0);

/**
 * self-attention の行ブロック枚数（直近 run の `lastRunPipelines` の ①QK の dispatch 本数から — 2.1 の `rowBlocksOf` と同じ
 * 数え方: 行窓のキー `:rwa` は層ごとにブロック枚数ぶん、行窓の無いキーは cross-attention で層ごとに 1 本）。
 */
const rowBlocksOf = (diagnostics: SessionDiagnostics): number => {
  const qk = (diagnostics.lastRunPipelines ?? []).filter(({ key }) =>
    key.startsWith("attention_qk:")
  );
  const count = (windowed: boolean): number =>
    qk.filter(({ key }) => key.includes(":rwa") === windowed)
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  const layers = count(false);
  assert(layers > 0, "①QK の行窓の無い dispatch（cross-attention）が 1 本も無い");
  return count(true) / layers;
};

/** 1 submit の GPU 時間の観測を ms に直した記録（門は {@link SUBMIT_GPU_LIMIT_MS}）。 */
const submitGpuNote = (
  diagnostics: SessionDiagnostics,
  unitNs: number,
): { readonly maxMs: number; readonly unbackedSubmits: number; readonly note: string } => {
  const budget = diagnostics.submit.chunkBudget;
  const observed = budget.submitGpuTime;
  if (observed === undefined) throw new Error("計測モードなのに submit の GPU 時間が無い");
  const ms = (ns: number): string => `${(ns * unitNs / 1e6).toFixed(1)} ms`;
  const note = [
    `1 submit の GPU 時間の最大 ${ms(observed.maxNs)}（${observed.submits} 本）`,
    `裏付け前 ${observed.unbackedSubmits} 本の最大 ${
      observed.maxUnbackedNs === undefined ? "—" : ms(observed.maxUnbackedNs)
    }`,
    `単発 dispatch の最大 ${ms(observed.maxDispatchNs)}（${observed.maxDispatchKey ?? "—"}）`,
    `窓平均の最大 ${budget.maxWindowMeanMs?.toFixed(1) ?? "—"} ms・推定の最大 ${
      budget.maxEstimatedMs?.toFixed(1) ?? "—"
    } ms・予算超過 ${budget.overBudgetChunks} 本`,
    `submit ${diagnostics.submit.submitCount} 本・dispatch ${diagnostics.submit.dispatchCount} 本`,
  ].join("・");
  return { maxMs: observed.maxNs * unitNs / 1e6, unbackedSubmits: observed.unbackedSubmits, note };
};

/**
 * 容量の記録の 1 行（diag を閾値・外挿・I2V 対応の増分の見込みと並べる — モジュール doc「実寸の計測と記録」）。diag は run の
 * 最大（1 本目 = アリーナ・2 本目 = slot backing のどちらが大きいかは形で変わる）。閾値を超えたら警告を出す（赤にはしない）。
 */
const capacityNote = (seat: Seat, tokens: number, runs: readonly Run[]): string => {
  const diag = Math.max(...runs.map(({ liveBytes }) => liveBytes));
  const over = diag > DIAG_THRESHOLD_BYTES;
  const parts = [
    `diag ${gib(diag)}`,
    over
      ? `容量の閾値 ${
        gib(DIAG_THRESHOLD_BYTES)
      } を超えた — 決定 8 の手（受理するフレーム数の上限を下げる）を段 6 の前に確定する`
      : `容量の閾値 ${gib(DIAG_THRESHOLD_BYTES)} の内（余裕 ${gib(DIAG_THRESHOLD_BYTES - diag)}）`,
  ];
  if (tokens === EXTRAPOLATED_TOKENS) {
    const extrapolated = EXTRAPOLATED_DIAG_GIB[seat];
    parts.push(
      `容量の表の外挿 ${extrapolated.toFixed(2)} GiB（2.1 型のグラフ）との差 ${
        (diag / GIB - extrapolated).toFixed(2)
      } GiB — I2V 対応の増分の見込み +${I2V_INCREMENT_GIB.placed}（where を消費の直前に置けたとき）/ 悪い側 +${I2V_INCREMENT_GIB.hoisted} GiB`,
    );
  }
  if (over) {
    console.warn(
      `[wan-ti2v-dit] ${seat} S = ${tokens} の diag ${gib(diag)} が容量の閾値 ${
        gib(DIAG_THRESHOLD_BYTES)
      } を超えた` +
        "（ADR 0121 決定 8 — 受理するフレーム数の上限を下げる手を段 6 の前に確定する）",
    );
  }
  return parts.join("・");
};

/**
 * linear 1 本（`weight`）の per-channel scale を 2 倍にした供給面（2.1 の `scaledScaleContainer` と同じ — 容器のファイルは
 * 書き換えず、取得した block の**写し**を 2 倍にして返す。scale の payload は block の先頭からの f32 の列）。
 */
const scaledScaleContainer = (
  opened: OpenedContainer,
  graph: string,
  weight: string,
): BoundContainer => {
  const bound = Object.hasOwn(opened.graphs, graph) ? opened.graphs[graph] : undefined;
  const scale = bound?.supplies.get(weight)?.scale;
  if (scale === undefined) {
    throw new Error(
      `グラフ '${graph}' の '${weight}' に companion scale が無い（i8 の linear でない）`,
    );
  }
  return {
    graphs: opened.graphs,
    readBlock: async (id) => {
      const bytes = await opened.readBlock(id);
      if (id !== scale.id) return bytes;
      const doubled = bytes.slice();
      const values = new Float32Array(doubled.buffer, 0, scale.payloadBytes / 4);
      for (let index = 0; index < values.length; index += 1) values[index] *= 2;
      return doubled;
    },
  };
};

const results = openResults("wan-ti2v-dit");

/**
 * 計測モードの照合が出した出力（通常モードで回すケースだけ — ケース名 → unpatchify 前のトークン）。通常モードのテストが
 * これと Uint32 で突き合わせる（チャンクの切れ目は値を動かさない — 2.1 の `measuredTokens` と同じ）。
 *
 * MUST: 計測モードの照合を通常モードより前に登録する（Deno は同じファイルのテストを登録順に回す）。計測モードを飛ばした回
 * （`--filter` など）は比べる相手が無いので、比較を黙って省かず落とす。
 */
const measuredTokens = new Map<string, Float32Array>();

/** 故障注入 1 件の結果（判定は呼び手が全件の比を出してから行う）。 */
type FaultResult = { readonly label: string; readonly ratio: number; readonly note: string };

/** 故障注入のラベル（ADR 0121 検収の段 2 の 4 件 — モジュール doc「故障注入」）。 */
const TIMESTEP_OFF_BY_ONE = "timestep +1";
const MASK_SHIFT = "条件マスクの 1 トークンずれ";
const TIME_INPUTS_SWAP = "時刻入力 2 本の取り違え";
const SCALE_FAULT = `${SCALE_FAULT_WEIGHT} の scale × 2`;

/** 同じ Session で回す故障注入（T2V は timestep +1 だけ・I2V は加えて条件マスクのずれと時刻入力の取り違え）。 */
const sessionFaultLabels = (form: Ti2vForm): readonly string[] =>
  form === "i2v" ? [TIMESTEP_OFF_BY_ONE, MASK_SHIFT, TIME_INPUTS_SWAP] : [TIMESTEP_OFF_BY_ONE];

/**
 * 受入れケースの故障注入のうち、同じ Session で入力を差し替えて回すもの（scale × 2 は別の Session — {@link scaleFault}）。
 */
const faultInjections = async (
  golden: Ti2vGolden,
  inputs: Readonly<Record<string, Tensor>>,
  grid: WanTokenGrid,
  form: Ti2vForm,
  expected: Float32Array,
  run: (inputs: Record<string, Tensor>, label: string) => Promise<Float32Array>,
): Promise<FaultResult[]> => {
  const faults: FaultResult[] = [];
  const measure = async (label: string, faulty: Record<string, Tensor>): Promise<void> => {
    const tokens = await run(faulty, `故障注入 ${label}`);
    const diff = difference(unpatchifyTokens(tokens, golden.latentShape, WAN22_GEOMETRY), expected);
    const ratio = ratioOf(diff);
    faults.push({
      label,
      ratio,
      note: `故障注入 ${label}: maxAbs ${diff.maxAbs.toExponential(3)}（比 ${
        ratio.toExponential(2)
      }）`,
    });
  };
  const width = timestepWidth(inputs);
  await measure(TIMESTEP_OFF_BY_ONE, {
    ...inputs,
    timesteps_proj: f32Tensor([1, width], timestepsProj(golden.timestep + 1, width)),
  });
  if (form === "i2v") {
    // 先頭の潜在フレームだけが条件 — F' = 1 だと全トークンが条件で、ずらす先が無い。
    assert(grid.frames >= 2, `I2V の受入れケースの潜在フレームが ${grid.frames}（2 以上のはず）`);
    const shifted = new Uint32Array(grid.count);
    shifted.fill(1, 1, grid.rows * grid.cols + 1);
    await measure(MASK_SHIFT, {
      ...inputs,
      condition_mask: boolTensor(inputOf(inputs, "condition_mask").shape, shifted),
    });
    await measure(TIME_INPUTS_SWAP, {
      ...inputs,
      timesteps_proj: inputOf(inputs, "timesteps_proj_condition"),
      timesteps_proj_condition: inputOf(inputs, "timesteps_proj"),
    });
  }
  assertEquals(
    faults.map(({ label }) => label),
    [...sessionFaultLabels(form)],
    "故障注入の顔ぶれ（黙って減らさない）",
  );
  return faults;
};

/** 1 ケースの r（帯の候補と形ごとの最悪の材料 — 判定の前に積むので帯の外の回も残る）。 */
type CaseRatio = {
  readonly name: string;
  readonly form: Ti2vForm;
  readonly accept: boolean;
  readonly latentShape: readonly number[];
  readonly ratio: number;
};

/** {@link ratioGate} の入力（組 1 つ — 実寸は S 1 つ）。 */
type GateSpec = {
  readonly t: Deno.TestContext;
  readonly gpu: GpuContext;
  readonly prepared: PreparedModel;
  readonly cases: readonly Ti2vCase[];
  /** r の帯。`undefined` = 未導出（r を記録して赤で止める）。 */
  readonly band: number | undefined;
  /** 実寸だけ: golden の S・行ブロックの枚数（束縛上限から導く期待）・計測モードの 1 submit の GPU 時間の門。 */
  readonly full?: {
    readonly tokens: number;
    readonly bindingLimit: number;
    readonly deviceLost: () => string | undefined;
  };
  /** 通常モードの比較相手（{@link measuredTokens}）を残すケース。 */
  readonly normalModeCase?: string;
};

/**
 * r の門（参照席 — モジュール doc「指標と帯」）を組 1 つについて回す（Session を張って全ケースを回し、畳んで解放を待つ）。
 * 1 ケース: io の入力で 1 forward → r を帯で判定。受入れは加えて製品の経路の入力と、同じ Session の故障注入
 * （{@link faultInjections}）。実寸（`full`）は golden の S・行ブロックの枚数・計測モードの 1 submit の GPU 時間も門にする。
 */
const ratioGate = async (spec: GateSpec): Promise<readonly CaseRatio[]> => {
  const { t, gpu, prepared, band, full } = spec;
  const bandLabel = full === undefined ? "S = 192 の帯" : "実寸の帯";
  const ratios: CaseRatio[] = [];
  const built = performance.now();
  const session = await prepared.createContainerSession(gpu);
  const buildMs = performance.now() - built;
  const runs: Run[] = [];
  try {
    const run = async (inputs: Record<string, Tensor>, label: string): Promise<Float32Array> => {
      const started = performance.now();
      const outputs = await session.run(inputs);
      const wallMs = performance.now() - started;
      const tensor = outputs[prepared.graph.outputs[0]];
      if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
      runs.push(observeRun(label, session.diagnostics(), wallMs, outputBytes(outputs)));
      // 次の run の確保（2 本目は slot backing の構築）の前に、この run の中間の解放を待つ。
      await settleReleases(gpu);
      return tensor.data;
    };
    for (const testCase of spec.cases) {
      const { name, role, form } = testCase;
      const accept = isAccept(testCase);
      const id = `${REFERENCE_QUANT}/${name}`;
      await t.step(`${id}（${role}）`, async () => {
        // 帯の外で落ちた回も、比と故障注入の記録を results.json に残す。
        let caseNote = "";
        await runRecordedCase(
          results,
          { id, failureNote: () => caseNote },
          async ({ measurements }) => {
            const golden = await loadTi2vGolden(name);
            const grid = wanTokenGrid(golden.latentShape, WAN22_GEOMETRY);
            if (full !== undefined) assertEquals(grid.count, full.tokens, `${id}: golden の S`);
            const inputs = graphInputs(golden, prepared.graph.inputs, name);
            const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
            const reference = floatsOf(golden.reference, "output", name);
            const referenceError = referenceErrorOf(golden, name);
            const referenceRatio = ratioOf(referenceError);
            const referenceQuantile = absErrorQuantile(reference, expected, RECORDED_QUANTILE);
            const unpatchify = (tokens: Float32Array) =>
              unpatchifyTokens(tokens, golden.latentShape, WAN22_GEOMETRY);
            const tokens = await run(inputs, `${name} io`);
            // 判定の前に残す（帯の外へ出た回も、通常モードのビット一致の門は走る）。
            if (name === spec.normalModeCase) measuredTokens.set(name, tokens);
            const latents = unpatchify(tokens);
            const diff = difference(latents, expected);
            const diffF32 = difference(latents, reference);
            const r = normalizedRatio(diff, referenceRatio);
            ratios.push({ name, form, accept, latentShape: golden.latentShape, ratio: r });
            measurements.push({
              output: "latents",
              maxAbs: diff.maxAbs,
              maxRel: diff.maxRel,
              tolerance: toleranceOf(diff, band, referenceRatio),
              stage: "karume",
            }, {
              output: "latents@f32-reference",
              maxAbs: diffF32.maxAbs,
              maxRel: diffF32.maxRel,
              tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
              stage: "karume",
            }, {
              output: "f32-reference@f64-reference",
              maxAbs: referenceError.maxAbs,
              maxRel: referenceError.maxRel,
              tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
              stage: "karume",
            });
            const diagnostics = session.diagnostics();
            const rowBlocks = full === undefined ? undefined : rowBlocksOf(diagnostics);
            const notes = [
              formatNormalized("io の入力 / f64 参照", diff, referenceRatio),
              formatDifference("CPU f32 参照 / f64 参照（正規化の分母）", referenceError),
              formatDifference("io の入力 / CPU f32 参照（記録）", diffF32),
              quantileRatio("io の入力 / f64 参照", latents, expected, referenceQuantile),
              ...(rowBlocks === undefined ? [] : [`行ブロック ${rowBlocks} 枚`]),
            ];
            assertEquals(diff.nonFinite, 0, `${id}: 非有限`);
            if (full !== undefined) {
              assertEquals(
                rowBlocks,
                expectedRowBlocks(full.tokens, full.bindingLimit),
                `${id}: self-attention の行ブロック枚数（束縛上限 ${full.bindingLimit} B）`,
              );
            }
            assertEquals(
              { ...diagnostics.lastRunFusions, ...EXPECTED_FUSIONS },
              diagnostics.lastRunFusions,
              `${id}: 融合の件数（adaln / rope / silu）`,
            );

            let hostDiff: Difference | undefined;
            let faults: FaultResult[] = [];
            if (accept) {
              const hostLatents = unpatchify(
                await run(hostInputs(golden, inputs, grid, form), `${name} 製品の経路の入力`),
              );
              hostDiff = difference(hostLatents, expected);
              measurements.push({
                output: "latents@host-inputs",
                maxAbs: hostDiff.maxAbs,
                maxRel: hostDiff.maxRel,
                tolerance: toleranceOf(hostDiff, band, referenceRatio),
                stage: "karume",
              });
              notes.push(
                formatNormalized("製品の経路の入力 / f64 参照", hostDiff, referenceRatio),
                quantileRatio(
                  "製品の経路の入力 / f64 参照",
                  hostLatents,
                  expected,
                  referenceQuantile,
                ),
              );
              faults = await faultInjections(golden, inputs, grid, form, expected, run);
              notes.push(
                ...faults.map(({ note, ratio }) =>
                  `${note}・r ${(ratio / referenceRatio).toPrecision(3)}（${
                    band === undefined
                      ? "帯は未導出"
                      : `帯の ${(ratio / referenceRatio / band).toFixed(1)} 倍`
                  }）`
                ),
              );
            }
            // 判定の前に出す（帯の外へ出た回も比が手元に残る）。
            caseNote = notes.join(" / ");
            console.log(`[wan-ti2v-dit] ${id}: ${caseNote}`);

            if (band === undefined) {
              throw new Error(
                `${id}: 帯が未導出（r ${r.toPrecision(3)}）— 決定用の最悪 r × 5 を帯の定数へ書く` +
                  "（帯の候補は同じ組の最後の step が出す）",
              );
            }
            // 故障注入の判定を先に置く（受入れが帯の外でも、帯が故障を拾えるかは判定される）。
            for (const { label, ratio } of faults) {
              const normalized = ratio / referenceRatio;
              assert(
                normalized > band,
                `${id}: 故障注入 ${label} の r ${
                  normalized.toPrecision(3)
                } が${bandLabel} ${band} の内に収まった`,
              );
            }
            if (accept) {
              const subtle = faults.find(({ label }) => label === TIMESTEP_OFF_BY_ONE);
              if (subtle === undefined) {
                throw new Error(`${id}: 故障注入 ${TIMESTEP_OFF_BY_ONE} が無い`);
              }
              const margin = subtle.ratio / referenceRatio / band;
              assert(
                margin >= SUBTLE_FAULT_MARGIN,
                `${id}: 故障注入 ${TIMESTEP_OFF_BY_ONE} が${bandLabel}の ${
                  margin.toFixed(2)
                } 倍（${SUBTLE_FAULT_MARGIN} 倍以上のはず — 帯が広すぎる兆候）`,
              );
            }
            assert(r <= band, `${id}: ${notes[0]} が${bandLabel} ${band} の外`);
            if (hostDiff !== undefined) {
              const hostRatio = normalizedRatio(hostDiff, referenceRatio);
              assert(
                hostRatio <= band,
                `${id}: 製品の経路の入力の r ${
                  hostRatio.toPrecision(3)
                } が${bandLabel} ${band} の外`,
              );
            }
            return { status: "pass", note: caseNote };
          },
        );
      });
    }
    if (full !== undefined) {
      await t.step(
        `${REFERENCE_QUANT}/S = ${full.tokens} の 1 submit の GPU 時間・所要・確保`,
        async () => {
          let note = "";
          await runRecordedCase(
            results,
            { id: `${REFERENCE_QUANT}/full-s${full.tokens}/submit`, failureNote: () => note },
            () => {
              // 換算の表はこのステップでだけ引く（表に無い環境で落ちるのは時間門だけ）。
              const unitNs = timestampUnitNs(ENVIRONMENT.key);
              const submit = submitGpuNote(session.diagnostics(), unitNs);
              note = [
                submit.note,
                `構築 ${(buildMs / 1000).toFixed(1)} s`,
                ...runs.map((observed) => formatRun(observed, unitNs)),
              ].join(" / ");
              console.log(
                `[wan-ti2v-dit] ${REFERENCE_QUANT} 実寸 S = ${full.tokens} の計測モード: ${note}`,
              );
              assert(
                submit.unbackedSubmits > 0,
                "最初の run の裏付け前のチャンク（initialChunkSize で据え置いた submit）を測れていない",
              );
              assert(
                submit.maxMs <= SUBMIT_GPU_LIMIT_MS,
                `1 submit の GPU 時間の最大 ${
                  submit.maxMs.toFixed(1)
                } ms が ${SUBMIT_GPU_LIMIT_MS} ms を超えた`,
              );
              assertEquals(full.deviceLost(), undefined, "device lost");
              return Promise.resolve({ status: "pass", note });
            },
          );
        },
      );
    }
  } finally {
    await session.dispose();
    // 次の Session の確保の前に、この Session の確保の解放を待つ（{@link settleReleases}）。
    await settleReleases(gpu);
  }
  return ratios;
};

/**
 * scale × 2 の故障注入（{@link scaledScaleContainer}）で受入れケースを回し、r が帯の外へ出ることを門にする。Session は
 * {@link ratioGate} の Session を畳んだ後に別に張る（2 本を同時に載せない — 実寸で 2 本は B570 の予算を食う）。
 */
const scaleFault = async (
  t: Deno.TestContext,
  gpu: GpuContext,
  opened: OpenedContainer,
  graph: string,
  cases: readonly Ti2vCase[],
  band: number | undefined,
): Promise<void> => {
  const prepared = prepareContainer(scaledScaleContainer(opened, graph, SCALE_FAULT_WEIGHT), graph);
  const session = await prepared.createContainerSession(gpu);
  try {
    for (const { name } of cases.filter(isAccept)) {
      const id = `${REFERENCE_QUANT}/${name}/scale-fault`;
      await t.step(`${id}（故障注入 ${SCALE_FAULT} → 帯の外）`, async () => {
        let note = "";
        await runRecordedCase(
          results,
          { id, failureNote: () => note },
          async ({ measurements }) => {
            const golden = await loadTi2vGolden(name);
            const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
            const referenceRatio = ratioOf(referenceErrorOf(golden, name));
            const outputs = await session.run(graphInputs(golden, prepared.graph.inputs, name));
            const tensor = outputs[prepared.graph.outputs[0]];
            if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
            await settleReleases(gpu);
            const diff = difference(
              unpatchifyTokens(tensor.data, golden.latentShape, WAN22_GEOMETRY),
              expected,
            );
            const normalized = normalizedRatio(diff, referenceRatio);
            measurements.push({
              output: "latents@scale-fault",
              maxAbs: diff.maxAbs,
              maxRel: diff.maxRel,
              tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
              stage: "karume",
            });
            note = `${
              formatNormalized(`故障注入 ${SCALE_FAULT} / f64 参照`, diff, referenceRatio)
            }（${
              band === undefined ? "帯は未導出" : `帯の ${(normalized / band).toFixed(1)} 倍`
            }）`;
            console.log(`[wan-ti2v-dit] ${id}: ${note}`);
            if (band === undefined) {
              throw new Error(`${id}: 帯が未導出（故障注入の r ${normalized.toPrecision(3)}）`);
            }
            assert(
              normalized > band,
              `${id}: 故障注入 ${SCALE_FAULT} の r ${
                normalized.toPrecision(3)
              } が帯 ${band} の内に収まった`,
            );
            return { status: "pass", note };
          },
        );
      });
    }
  } finally {
    await session.dispose();
    await settleReleases(gpu);
  }
};

/** 潜在の形の綴り（`48·21·30·52`）。 */
const shapeLabel = (shape: readonly number[]): string => shape.join("·");

/**
 * 組の r の形ごとの最悪の記録（ADR 0121 検収の段 2「形ごとの最悪も記録する」— T2V / I2V と潜在の形ごと。決定用と受入れを
 * 分けて並べる）。門ではない（帯の判定は各ケースの step）。r を出せなかったケースは数に入らないので、本数も並べる。
 */
const recordWorst = async (
  t: Deno.TestContext,
  label: string,
  id: string,
  ratios: readonly CaseRatio[],
): Promise<void> => {
  await t.step(`${label}: 形ごとの最悪 r（記録）`, async () => {
    await runRecordedCase(results, { id: `${REFERENCE_QUANT}/${id}/worst` }, () => {
      const worstOf = (rows: readonly CaseRatio[]): string => {
        if (rows.length === 0) return "なし";
        const worst = rows.reduce((left, right) => (right.ratio > left.ratio ? right : left));
        return `${worst.ratio.toPrecision(3)}（${worst.name}・${rows.length} 本）`;
      };
      const groups = (accept: boolean): string[] => {
        const rows = ratios.filter((row) => row.accept === accept);
        const shapes = [...new Set(rows.map(({ latentShape }) => shapeLabel(latentShape)))];
        return [
          ...(["t2v", "i2v"] as const).map((form) =>
            `${form.toUpperCase()} ${worstOf(rows.filter((row) => row.form === form))}`
          ),
          ...shapes.map((shape) =>
            `潜在 [${shape}] ${
              worstOf(rows.filter(({ latentShape }) => shapeLabel(latentShape) === shape))
            }`
          ),
        ];
      };
      const note = `決定用: ${groups(false).join("・")} / 受入れ: ${groups(true).join("・")}`;
      console.log(`[wan-ti2v-dit] ${label} の形ごとの最悪 r: ${note}`);
      return Promise.resolve({ status: "pass", note });
    });
  });
};

/**
 * 未導出の帯（`undefined`）の候補を出して赤で止める step（2.1 の `bandCandidate` と同じ規則 — 決定用だけから作り、受入れの r
 * は並べるが候補に使わない。決定用のどれかが r を出す前に落ちた回は候補を出さない）。帯が導出済みなら何もしない。
 */
const bandCandidate = async (
  t: Deno.TestContext,
  label: string,
  cases: readonly Ti2vCase[],
  ratios: readonly CaseRatio[],
  band: number | undefined,
): Promise<void> => {
  if (band !== undefined) return;
  await t.step(`${label}: 帯の候補（未導出）`, () => {
    const expected = cases.filter((testCase) => !isAccept(testCase)).length;
    const decided = ratios.filter(({ accept }) => !accept);
    const worst = Math.max(...decided.map(({ ratio }) => ratio));
    const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
    const candidate = decided.length === expected
      ? `決定用 ${expected} 本の最悪 r ${worst.toPrecision(3)} × 5 = ${
        (worst * 5).toPrecision(3)
      }` +
        `（有効数字 2 桁へ切り上げ ${Number((Math.ceil(worst * 5 / unit) * unit).toPrecision(2))}）`
      : `決定用 ${expected} 本のうち r が出たのは ${decided.length} 本 — 候補にしない（落ちたケースを先に調べる）`;
    const accepted = ratios.filter(({ accept }) => accept)
      .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`);
    throw new Error(
      `${label}: 帯が未導出 — ${candidate}。受入れの r（帯の決定に使わない）: ${
        accepted.join(" / ") || "なし"
      }`,
    );
  });
};

/**
 * Session を張って同じ入力で 2 本回す（1 本目 = パイプラインの生成と裏付け前のチャンク・2 本目 = 生成の 1 パスに近い 2 回目以降の
 * 形 — 2.1 の通常モードと同じ並び）。fdinfo の区間は構築・各 run・破棄で切る。戻りは 2 本の観測・出力・非有限の数・構築の所要・
 * 区間ごとの VRAM の山と、1 本目の後の行ブロックの枚数。
 */
const twoRuns = async (
  prepared: PreparedModel,
  gpu: GpuContext,
  inputs: Record<string, Tensor>,
  options: SessionOptions = {},
): Promise<{
  readonly runs: readonly Run[];
  readonly produced: readonly Float32Array[];
  readonly nonFinite: number;
  readonly buildMs: number;
  readonly vram: readonly string[];
  readonly rowBlocks: number;
}> => {
  const monitor = monitorDrmUsage();
  const runs: Run[] = [];
  const produced: Float32Array[] = [];
  let nonFinite = 0;
  let buildMs = 0;
  let rowBlocks = 0;
  let vram: readonly string[] = [];
  try {
    monitor.enter("構築");
    const built = performance.now();
    const session: Session = await prepared.createContainerSession(gpu, options);
    buildMs = performance.now() - built;
    try {
      for (const label of ["1 本目", "2 本目"]) {
        monitor.enter(label);
        const started = performance.now();
        const outputs = await session.run(inputs);
        const wallMs = performance.now() - started;
        const tensor = outputs[prepared.graph.outputs[0]];
        if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
        produced.push(tensor.data);
        nonFinite += tensor.data.reduce(
          (count, value) => count + (Number.isFinite(value) ? 0 : 1),
          0,
        );
        runs.push(observeRun(label, session.diagnostics(), wallMs, outputBytes(outputs)));
        rowBlocks = rowBlocksOf(session.diagnostics());
        // 次の run の確保の前に、この run の中間の解放を待つ。
        await settleReleases(gpu);
      }
    } finally {
      monitor.enter("破棄");
      await session.dispose();
      await settleReleases(gpu);
    }
  } finally {
    // 成功の経路でも失敗の経路でも interval を残さない（stop は標本を 1 つ足して止めるだけ）。
    const { peaks } = monitor.stop();
    vram = [...peaks].map(([phase, peak]) => `VRAM 山 [${phase}] ${formatDrmUsage(peak)}`);
  }
  return { runs, produced, nonFinite, buildMs, vram, rowBlocks };
};

Deno.test(
  "Wan2.2 TI2V DiT 行ブロックの期待値（GPU 不要）: B570 の束縛上限で S = 8,190 / 7,920 が 3 枚・S = 12,090 が 7 枚",
  () => {
    assertEquals(
      [...FULL_TOKENS_ORDER, LONG_TOKENS].map((tokens) =>
        expectedRowBlocks(tokens, B570_STORAGE_BINDING_LIMIT)
      ),
      [3, 3, 7],
    );
  },
);

Deno.test(
  "Wan2.2 TI2V DiT ケースの表（GPU 不要）: 実寸は S ごとに決定用 3 本 + 受入れ T2V 2 本 + I2V 2 本・名前の役割 / 形 / S が欄と一致",
  () => {
    for (const { tokens, cases } of FULL_GROUPS) {
      assertEquals(
        cases.filter((testCase) => !isAccept(testCase)).length,
        3,
        `S = ${tokens} の決定用`,
      );
      for (const form of ["t2v", "i2v"] as const) {
        assertEquals(
          cases.filter((testCase) => isAccept(testCase) && testCase.form === form).length,
          2,
          `S = ${tokens} の受入れの ${form}`,
        );
      }
    }
    const decided = WAN_TI2V_FULL_CASES.filter((testCase) => !isAccept(testCase));
    assertEquals(decided.filter(({ form }) => form === "t2v").length, 3, "決定用の T2V");
    assertEquals(decided.filter(({ form }) => form === "i2v").length, 3, "決定用の I2V");
    for (const { name, role, form, tokens } of WAN_TI2V_FULL_CASES) {
      assert(
        name.startsWith(`${role}-${form}-s${String(tokens).padStart(5, "0")}-t`),
        `${name}: 名前の役割 / 形 / S が欄（${role} / ${form} / ${tokens}）と食い違う`,
      );
    }
  },
);

Deno.test({
  name:
    `Wan2.2 TI2V DiT 故障注入の対象（GPU 不要）: ${SCALE_FAULT_WEIGHT} が容器の i8 の linear で companion scale を持つ`,
  ignore: !MODEL_PRESENT,
  fn: async () => {
    const opened = await openSeriesContainer(MODEL_URL);
    // 対象が無ければ scaledScaleContainer が名指しで落とす（GPU の門の前に、注入が空振りしないことを見る）。
    scaledScaleContainer(opened, graphName(), SCALE_FAULT_WEIGHT);
  },
});

// S = 12,090（opt-in）は確保が最大なので GPU テストの先頭に置く（ADR 0121 決定 12 — B570 の `destroy()` の解放遅れで、後ろに
// 置いた大きい確保が OOM になった前例）。実寸の照合 → 通常モード → 実用席 → S = 192 の順も確保の大きい順。

/** S = 12,090 の入力（乱数の潜在・実プロンプトの埋め込み・I2V の条件マスク — モジュール doc「実寸の計測と記録」）。 */
const longInputs = async (base: WanRopeBase, width: number): Promise<Record<string, Tensor>> => {
  const grid = wanTokenGrid(LONG_LATENT_SHAPE, WAN22_GEOMETRY);
  assertEquals(grid.count, LONG_TOKENS, "S = 12,090 の格子");
  const latents = gaussianLatents(
    LONG_LATENT_SHAPE.reduce((left, right) => left * right),
    LONG_SEED,
  );
  const tokens = patchifyLatents(latents, LONG_LATENT_SHAPE, WAN22_GEOMETRY);
  const tables = wanRopeTables(base, grid);
  const context = await loadTi2vGolden(LONG_CONTEXT_CASE);
  const textView = viewOf(context.io, "input.encoder_hidden_states", LONG_CONTEXT_CASE);
  const rope = [1, grid.count, 1, tables.cos.length / grid.count];
  return {
    tokens: f32Tensor([1, grid.count, tokens.length / grid.count], tokens),
    timesteps_proj: f32Tensor([1, width], timestepsProj(LONG_TIMESTEP, width)),
    encoder_hidden_states: f32Tensor(
      textView.shape,
      floatsOf(context.io, "input.encoder_hidden_states", LONG_CONTEXT_CASE),
    ),
    rope_cos: f32Tensor(rope, tables.cos),
    rope_sin: f32Tensor(rope, tables.sin),
    timesteps_proj_condition: f32Tensor([1, width], timestepsProj(0, width)),
    condition_mask: boolTensor([1, grid.count, 1], conditionMask(grid, "i2v")),
  };
};

/** 標準正規の乱数（mulberry32 + Box–Muller — 値は確保と非有限の検査にしか効かない・seed 固定で再現する）。 */
const gaussianLatents = (count: number, seed: number): Float32Array<ArrayBuffer> => {
  let state = seed >>> 0;
  const uniform = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let x = state;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return (((x ^ (x >>> 14)) >>> 0) + 0.5) / 2 ** 32;
  };
  const out = new Float32Array(count);
  for (let index = 0; index < count; index += 2) {
    const radius = Math.sqrt(-2 * Math.log(uniform()));
    const angle = 2 * Math.PI * uniform();
    out[index] = radius * Math.cos(angle);
    if (index + 1 < count) out[index + 1] = radius * Math.sin(angle);
  }
  return out;
};

Deno.test({
  name:
    `Wan2.2 TI2V DiT S = 12,090（832×480・121 フレーム・実 GPU・opt-in ${LONG_PROBE_ENV}=1）: 参照席と実用席の diag と` +
    "完走の可否を記録する（非有限 0・device lost なし）",
  ignore: !LONG_PROBE || !SMALL_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const opened = await openSeriesContainer(MODEL_URL);
    const prepared = prepareContainer(opened, graphName());
    const width = prepared.graph.inputs.find(({ name }) => name === "timesteps_proj")?.shape[1];
    if (typeof width !== "number") throw new Error("timesteps_proj の幅が静的な数でない");
    const inputs = await longInputs(await readRopeBase(opened), width);
    let deviceLost: string | undefined;
    const gpu = await acquireGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      const bindingLimit = gpu.device.limits.maxStorageBufferBindingSize;
      for (
        const [seat, options] of [
          [REFERENCE_QUANT, {}],
          [PRACTICAL_QUANT, practicalSessionOptions()],
        ] as const
      ) {
        await t.step(`${seat} S = ${LONG_TOKENS}`, async () => {
          let note = "";
          await runRecordedCase(
            results,
            {
              id: `${seat}/s${LONG_TOKENS}/capacity`,
              // 完走しなかった回（確保の失敗・device lost）は理由を記録に残す — 「可否」の否の側の記録。
              failureNote: (cause) =>
                note === ""
                  ? `完走せず: ${cause instanceof Error ? cause.message : String(cause)}`
                  : note,
            },
            async () => {
              const observed = await twoRuns(prepared, gpu, inputs, options);
              note = [
                capacityNote(seat, LONG_TOKENS, observed.runs),
                `構築 ${(observed.buildMs / 1000).toFixed(1)} s`,
                ...observed.runs.map((run) => formatRun(run)),
                ...observed.vram,
                `行ブロック ${observed.rowBlocks} 枚`,
                `非有限 ${observed.nonFinite}`,
              ].join(" / ");
              console.log(`[wan-ti2v-dit] ${seat} S = ${LONG_TOKENS}: ${note}`);
              assertEquals(observed.nonFinite, 0, `${seat} S = ${LONG_TOKENS}: 非有限`);
              if (seat === REFERENCE_QUANT) {
                assertEquals(
                  observed.rowBlocks,
                  expectedRowBlocks(LONG_TOKENS, bindingLimit),
                  `${seat}: self-attention の行ブロック枚数（束縛上限 ${bindingLimit} B）`,
                );
              }
              assertEquals(deviceLost, undefined, "device lost");
              return { status: "pass", note };
            },
          );
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    `Wan2.2 TI2V DiT 参照席 ${REFERENCE_QUANT} 実寸（実 GPU・計測モード / CPU f64）: S = 8,190 / 7,920 の 1 forward が` +
    "実寸の帯の内・故障注入 4 件は帯の外・行ブロック・1 submit の GPU 時間 ≤ 1 s",
  ignore: !FULL_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const graph = graphName();
    const opened = await openSeriesContainer(MODEL_URL);
    const prepared = prepareContainer(opened, graph);
    assertEquals(prepared.graph.outputs.length, 1, "製品のグラフの出力は 1 本");
    let deviceLost: string | undefined;
    // 計測モード（timestamp-query）: submit ごとの GPU 時間は既存の回収に相乗りして測る（2.1 の実寸の照合と同じ）。
    const gpu = await acquireGpu({
      gpuTiming: true,
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      // 期待値の元は device が実際に許した束縛上限（runtime の行ブロックが見るのと同じ値）。
      const bindingLimit = gpu.device.limits.maxStorageBufferBindingSize;
      const ratios: CaseRatio[] = [];
      for (const { tokens, cases, normalModeCase } of FULL_GROUPS) {
        ratios.push(
          ...await ratioGate({
            t,
            gpu,
            prepared,
            cases,
            band: FULL_BAND,
            full: { tokens, bindingLimit, deviceLost: () => deviceLost },
            normalModeCase,
          }),
        );
        await scaleFault(t, gpu, opened, graph, cases, FULL_BAND);
      }
      await recordWorst(t, "実寸の組", "full", ratios);
      await bandCandidate(t, `${REFERENCE_QUANT}/実寸の組`, WAN_TI2V_FULL_CASES, ratios, FULL_BAND);
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    `Wan2.2 TI2V DiT 参照席 ${REFERENCE_QUANT} 実寸 通常モード（実 GPU・計測なし）: S ごと（8,190 / 7,920）の所要と diag を` +
    "記録し（容量の閾値・外挿・I2V 対応の増分と並べる）、出力が計測モードと Uint32 で一致する",
  ignore: !FULL_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const prepared = prepareContainer(await openSeriesContainer(MODEL_URL), graphName());
    let deviceLost: string | undefined;
    const gpu = await acquireGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      for (const { tokens, normalModeCase: name } of FULL_GROUPS) {
        await t.step(`S = ${tokens}`, async () => {
          let note = "";
          await runRecordedCase(
            results,
            { id: `${REFERENCE_QUANT}/${name}/normal-mode`, failureNote: () => note },
            async ({ measurements }) => {
              const golden = await loadTi2vGolden(name);
              const inputs = graphInputs(golden, prepared.graph.inputs, name);
              const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
              const referenceRatio = ratioOf(referenceErrorOf(golden, name));
              const observed = await twoRuns(prepared, gpu, inputs);
              const tokensOut = observed.produced[observed.produced.length - 1];
              const diff = difference(
                unpatchifyTokens(tokensOut, golden.latentShape, WAN22_GEOMETRY),
                expected,
              );
              measurements.push(
                {
                  output: "latents",
                  maxAbs: diff.maxAbs,
                  maxRel: diff.maxRel,
                  tolerance: toleranceOf(diff, FULL_BAND, referenceRatio),
                  stage: "karume",
                } satisfies Measurement,
              );
              note = [
                formatNormalized("io の入力 / f64 参照", diff, referenceRatio),
                capacityNote(REFERENCE_QUANT, tokens, observed.runs),
                `構築 ${(observed.buildMs / 1000).toFixed(1)} s`,
                ...observed.runs.map((run) => formatRun(run)),
                ...observed.vram,
                `非有限 ${observed.nonFinite}`,
              ].join(" / ");
              console.log(
                `[wan-ti2v-dit] ${REFERENCE_QUANT} 実寸 S = ${tokens} の通常モード: ${note}`,
              );
              assertEquals(diff.nonFinite, 0, `${name}: 非有限`);
              // 帯が未導出の回は計測モードの照合が赤で止める（ここは所要と確保の記録を残す）。
              if (FULL_BAND !== undefined) {
                assert(
                  normalizedRatio(diff, referenceRatio) <= FULL_BAND,
                  `${name}: 通常モードの r ${
                    normalizedRatio(diff, referenceRatio).toPrecision(3)
                  } が実寸の帯 ${FULL_BAND} の外`,
                );
              }
              const measured = measuredTokens.get(name);
              if (measured === undefined) {
                throw new Error(
                  `${name}: 計測モードの照合の出力が無い（同じ回で計測モードの照合を先に回す — measuredTokens の MUST）`,
                );
              }
              for (const [index, data] of observed.produced.entries()) {
                assertEquals(
                  firstBitMismatch(data, measured),
                  -1,
                  `${name}: 通常モードの ${
                    observed.runs[index].label
                  }が計測モードの出力と Uint32 で一致しない` +
                    "（値は最初に割れる要素の添字）",
                );
              }
              assertEquals(deviceLost, undefined, "device lost");
              return { status: "pass", note };
            },
          );
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    `Wan2.2 TI2V DiT 実用席 ${PRACTICAL_QUANT}（実 GPU・通常モード・記録だけ）: S ごと（8,190 / 7,920）の所要と diag を記録する` +
    "（容量の閾値・外挿・I2V 対応の増分と並べる・非有限 0）",
  ignore: !FULL_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const options = practicalSessionOptions();
    const prepared = prepareContainer(await openSeriesContainer(MODEL_URL), graphName());
    let deviceLost: string | undefined;
    const gpu = await acquireGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      for (const { tokens, normalModeCase: name } of FULL_GROUPS) {
        await t.step(`${PRACTICAL_QUANT} S = ${tokens}`, async () => {
          let note = "";
          await runRecordedCase(
            results,
            { id: `${PRACTICAL_QUANT}/${name}/normal-mode`, failureNote: () => note },
            async () => {
              // 入力は参照席の golden の io（入力は席に依らない）。数値の門は段 7 の自機 A/B 門が持つ。
              const golden = await loadTi2vGolden(name);
              const observed = await twoRuns(
                prepared,
                gpu,
                graphInputs(golden, prepared.graph.inputs, name),
                options,
              );
              note = [
                capacityNote(PRACTICAL_QUANT, tokens, observed.runs),
                `構築 ${(observed.buildMs / 1000).toFixed(1)} s`,
                ...observed.runs.map((run) => formatRun(run)),
                ...observed.vram,
                `行ブロック ${observed.rowBlocks} 枚`,
                `非有限 ${observed.nonFinite}`,
              ].join(" / ");
              console.log(
                `[wan-ti2v-dit] 実用席 ${PRACTICAL_QUANT} S = ${tokens} の通常モード: ${note}`,
              );
              assertEquals(observed.nonFinite, 0, `${name}: 非有限`);
              assertEquals(deviceLost, undefined, "device lost");
              return { status: "pass", note };
            },
          );
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    `Wan2.2 TI2V DiT 参照席 ${REFERENCE_QUANT}（実 GPU / CPU f64）: S = 192 の組の 1 forward が帯の内・故障注入 4 件は帯の外`,
  ignore: !SMALL_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const graph = graphName();
    const opened = await openSeriesContainer(MODEL_URL);
    const prepared = prepareContainer(opened, graph);
    assertEquals(prepared.graph.outputs.length, 1, "製品のグラフの出力は 1 本");
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
      const ratios = await ratioGate({ t, gpu, prepared, cases: WAN_TI2V_CASES, band: SMALL_BAND });
      await scaleFault(t, gpu, opened, graph, WAN_TI2V_CASES, SMALL_BAND);
      await recordWorst(t, "S = 192 の組", "s192", ratios);
      await bandCandidate(t, `${REFERENCE_QUANT}/S = 192 の組`, WAN_TI2V_CASES, ratios, SMALL_BAND);
    } finally {
      // device を捨てる前に解放を待つ（後続のテストの予算を残す — {@link settleReleases}）。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});
