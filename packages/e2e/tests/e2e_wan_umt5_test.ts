/**
 * Wan2.1 の umT5（i8 系列のテキストエンコーダ）の**移植の門**（実 GPU・ADR 0119 段 10c — 決定 3 / 4 / 5 / 8）。
 *
 * 系列の容器（`outputs/series/umt5-xxl-i8-dyn/text_encoder/` — 重み i8 per-channel・活性 f32・入力は token id
 * `[1,L]` とバケット表 `[L,L]`・L は 2〜512 の記号次元）を karume runtime で回し、CPU の層逐次の参照（同じ i8 の
 * fake-quant 重み — recipe `wan/umt5_reference.py`）と突き合わせる。パイプラインへの結線は段 10d で、ここは
 * text_encoder 単体の forward を見る。
 *
 * 生成は `cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_export reference`（容器は
 * 段 10b の書き出し）。golden は 1 ケース 1 本の `reference.<case>.safetensors` で、中身と読み口・帯・判定は
 * `helpers/umt5-golden.ts`（単体の `umt5.Encoder` の通しの e2e と共有の 1 本）。
 *
 * ## 門
 *
 * 1. **入力の両経路の一致**（GPU 不要）: golden の原文を TS のトークナイザ（`WanPromptEncoder` — 10a の資産）に
 *    通した id 列と、TS のバケット表（`buildUmt5RelativePositionBuckets`）が golden の入力と**ビット一致**。GPU の
 *    門はこの TS の入力（製品の経路）で回す。golden が今の容器から採られたこと（part 0 の sha256）も見る。
 * 2. **正規化比 r の帯**（決定 8 ①）: r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷ max|f64|)（DiT の
 *    実寸の門と同じ指標 — `e2e_wan_dit_test.ts` の `DIT_FULL_NORMALIZED_BAND`）。帯は決定用 6 本の最悪 r × 5 を
 *    有効数字 2 桁へ切り上げた値で、受入れ 4 本は別に判定する（{@link UMT5_NORMALIZED_BAND}）。
 * 3. **故障注入**（受入れの 4 本で・帯の外へ出ること）: 相対位置の表を 1 ずらす・id 列の 1 トークンを別の id に・
 *    層 0 / 1 の相対位置の表の取り違え（容器の供給面で block を入れ替える — 別の Session）・i8 に固有の 1 件
 *    （中ほどの層の linear 1 本の per-channel scale を 2 倍 — {@link scaledScaleContainer}・別の Session）。
 *
 * 資産が無い環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。資産が**一部だけ**ある環境は
 * SKIP ではなく FAIL にする（完全性テスト）。
 *
 * ## Default run vs opt-in upstream parity
 *
 * Gate 2 (the r band against the CPU f64 reference) and gate 3 (the fault injections that prove the band
 * is live) compare against the recipe's goldens under a tolerance, so they are opt-in behind
 * `KARUME_UPSTREAM_PARITY=1` (helpers/upstream-parity.ts). The default run keeps asset completeness,
 * gate 1 (host tokenizer / bucket-table bit identity and the golden's container sha256), the graph's
 * input / output structure, and each case's TS-input forward having the golden's output shape with no
 * non-finite value.
 *
 * ## Public `umt5.Encoder` through the distribution mirror
 *
 * The last two tests of this file drive the public `umt5.Encoder` (`@karume/models/umt5`) built by
 * `fromPretrained(nodeDirectory(…))` over the distribution mirror `models/karume-umt5-xxl`, against the same 10
 * goldens and the same band. They cover what the series-level gate above does not: the hub mirror path (manifest
 * selection, tokenizer asset), the encoder's own tokenizer ids, and abort / dispose. Their skip conditions are
 * their own (mirror + goldens), independent of the series container and the 10a tokenizer asset. The band check
 * on the mirror is opt-in like the series-level band; ids, shape, finiteness and abort / dispose stay in the
 * default run.
 */

import { readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { fetchAssets, loadManifest, resolveSelection } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import {
  type BoundContainer,
  type GpuContext,
  type OpenedContainer,
  prepareContainer,
  type PreparedModel,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import * as umt5 from "../../models/umt5.ts";
import { parseUmt5TokenizerAsset, Umt5Tokenizer } from "../../models/src/umt5/tokenizer.ts";
import { parseWanTokenizerAsset, WanPromptEncoder } from "../../models/src/wan/text/tokenizer.ts";
import {
  buildUmt5RelativePositionBuckets,
  type I32Tensor,
} from "../../models/src/umt5/relative-position.ts";
import { type Umt5SessionInputs, umt5SessionInputs } from "../../models/src/umt5/session-io.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  BUCKETS_KEY,
  type CaseRole,
  CASES,
  COMPONENT,
  difference,
  filePresent,
  firstMismatch,
  formatRatio,
  GENERATE,
  type Golden,
  goldenUrl,
  INPUT_IDS_KEY,
  judgeBand,
  loadGolden,
  ratioOf,
  referenceRatioOf,
  SERIES_DIR,
  SERIES_NAME,
  UMT5_NORMALIZED_BAND,
} from "../../models/tests/helpers/umt5-golden.ts";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

const MODEL_FILE = "model.krm";
/**
 * 容器の中のグラフ名（表は helpers/series-graphs.ts の 1 本）。**使うときに引く** — 表に行が無いときにモジュール
 * 評価で落とすと、GPU 不要のテストまで巻き添えで落ちる。
 */
const graphName = (): string => seriesGraph(SERIES_NAME, COMPONENT);

/** 10a のトークナイザ資産（recipe `wan/umt5_tokenizer.py` が焼く）。 */
const TOKENIZER_URL = new URL(
  "../../../outputs/series/wan2.1-umt5-tokenizer/tokenizer.json",
  import.meta.url,
);

const GENERATE_TOKENIZER =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_tokenizer";

/**
 * 故障注入で取り違える 2 層の相対位置の表（層の添字の 1 ずれ — 先頭の 2 層）。容器の initializer 名は recipe の
 * `umt5_reference.layer_keys` と同じ綴り。
 */
const relativeBiasTable = (layer: number): string =>
  `encoder.block.${layer}.layer.0.SelfAttention.relative_attention_bias.weight`;
const SWAPPED_TABLES: readonly [string, string] = [relativeBiasTable(0), relativeBiasTable(1)];

/**
 * i8 に固有の故障注入で per-channel scale を倍にする linear（{@link scaledScaleContainer}）。24 層の中ほど（層 12）の
 * FFN の出口 `wo` を選ぶ: 出力は残差へ線形に足されるので、倍率 (1 + δ) はその層の FFN の寄与を δ だけずらす形で出て、
 * δ と r の関係が読める。q / k は避けた — umT5 は q / k の直後に norm もスケールも無く（上流
 * `UMT5Attention.forward` — `scores = q·kᵀ` に相対位置のバイアスを足して softmax）倍率は打ち消されないが、softmax の
 * 温度として非線形に効くので、微妙な水準で δ と r の関係が読みにくい（DiT が V を選んだ理由 — q / k の直後の
 * RMS norm — とは事情が違う）。
 */
const I8_SCALE_FAULT_WEIGHT = "encoder.block.12.layer.1.DenseReluDense.wo.weight";
/** 帯の外を門にする倍率（i8 の scale の取り違え — 軸・値・束縛 — の大きさ）。 */
const GROSS_SCALE_FACTOR = 2;

const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, SERIES_DIR));
const expectedFiles = CASES.map(({ name }) => goldenUrl(name));
const GOLDENS_PRESENT = expectedFiles.every(filePresent);
const ASSETS_AVAILABLE = MODEL_PRESENT && GOLDENS_PRESENT;
const ANY_PRESENT = MODEL_PRESENT || expectedFiles.some(filePresent);
const tokenizerText = await readTextIfPresent(TOKENIZER_URL);

if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_DIR.pathname} に umT5 の容器と golden が揃っていないため Wan の umT5 の e2e を SKIP ` +
      `する（重み 5.3 GB につきリポジトリ管理外）。生成: ${GENERATE}`,
  );
} else if (tokenizerText === undefined) {
  console.warn(
    `[karume] ${TOKENIZER_URL.pathname} が無いため Wan の umT5 の e2e（入力は TS のトークナイザで作る）を ` +
      `SKIP する。生成: ${GENERATE_TOKENIZER}`,
  );
}

const hex = (bytes: ArrayBuffer): string =>
  [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");

/** 容器の part 0（ヘッダ + 2 文書 — 全 block の sha256 を宣言する）の sha256（golden のメタと突き合わせる）。 */
const containerPart0Sha256 = async (): Promise<string> => {
  const [part0] = resolveParts(new URL(MODEL_FILE, SERIES_DIR));
  return hex(await crypto.subtle.digest("SHA-256", await readFile(part0)));
};

/** TS のトークナイザ（資産が無い回は ignore で SKIP 済み — ここへ来たら条件の食い違いとして落とす）。 */
const tokenizerEncoder = (): WanPromptEncoder => {
  if (tokenizerText === undefined) {
    throw new Error(`${TOKENIZER_URL.pathname} が無い（ignore の条件と食い違っている）`);
  }
  return new WanPromptEncoder(parseWanTokenizerAsset(JSON.parse(tokenizerText)));
};

/**
 * 製品の経路（TS のトークナイザ + TS のバケット表）でグラフ入力を作り、golden の入力（Python の経路）と
 * **ビット一致**することを見てから返す（門 1 — GPU の門は一致した入力でしか回さない）。
 */
const hostInputs = (encoder: WanPromptEncoder, golden: Golden, name: string) => {
  const ids = encoder.encode(golden.meta.prompt, name);
  const at = firstMismatch(ids, golden.ids);
  assertEquals(
    at,
    -1,
    `${name}: TS の id 列が golden と ${at} 番目で割れる（TS ${ids.length} 本）`,
  );
  const buckets = buildUmt5RelativePositionBuckets(ids.length);
  const cell = firstMismatch(buckets.data, golden.buckets);
  assertEquals(cell, -1, `${name}: TS のバケット表が golden と要素 ${cell} で割れる`);
  return { ids, buckets, inputs: umt5SessionInputs(ids, buckets) };
};

/** 相対位置の表を 1 ずらした表（`table'[i][j] = bucket(j + 1 − i)` — ホストの距離の 1 ずれの故障注入）。 */
const shiftedBuckets = (length: number): I32Tensor => {
  const wide = buildUmt5RelativePositionBuckets(length + 1);
  const data = new Int32Array(length * length);
  for (let row = 0; row < length; row += 1) {
    const start = row * (length + 1) + 1;
    data.set(wide.data.subarray(start, start + length), row * length);
  }
  return { dtype: "i32", shape: [length, length], data };
};

/**
 * id 列の中ほどの 1 トークンを、同じ列の別の id に置き換えた列（語彙の内の id を使う — 範囲外の id で落ちる
 * 形ではなく、別の語として黙って通る形を注入する）。
 */
const replacedToken = (ids: Int32Array<ArrayBuffer>): Int32Array<ArrayBuffer> => {
  const at = Math.floor((ids.length - 1) / 2);
  const other = ids.find((id, index) => index < ids.length - 1 && id !== ids[at]);
  if (other === undefined) throw new Error("置き換える別の id が列に無い");
  const replaced = ids.slice();
  replaced[at] = other;
  return replaced;
};

/**
 * 2 層の相対位置の表の block を入れ替えた供給面（故障注入 — 層の表の取り違え）。容器のファイルは書き換えず、
 * `readBlock` が返す block を入れ替える（どちらも sha256 の検証を通った本物の block — `BoundContainer.readBlock`
 * の MUST どおり、返された器は書き換えない）。
 */
const swappedTablesContainer = (
  opened: OpenedContainer,
  graph: string,
  [first, second]: readonly [string, string],
): BoundContainer => {
  const bound = Object.hasOwn(opened.graphs, graph) ? opened.graphs[graph] : undefined;
  const blockOf = (weight: string): string => {
    const supply = bound?.supplies.get(weight);
    if (supply === undefined || supply.blocks.length !== 1 || supply.scale !== undefined) {
      throw new Error(`グラフ '${graph}' の '${weight}' が丸ごと 1 block の F32 でない`);
    }
    return supply.blocks[0].id;
  };
  const swap = new Map([
    [blockOf(first), blockOf(second)],
    [blockOf(second), blockOf(first)],
  ]);
  return {
    graphs: opened.graphs,
    readBlock: (id) => opened.readBlock(swap.get(id) ?? id),
  };
};

/**
 * linear 1 本（`weight`）の per-channel scale を `factor` 倍にした供給面（i8 に固有の故障注入 —
 * {@link I8_SCALE_FAULT_WEIGHT}・DiT の `scaledScaleContainer` に倍率の引数を足した形）。容器のファイルは書き換えない:
 * 取得した block（sha256 の検証の後）の**写し**を倍にして返す（`BoundContainer.readBlock` の MUST — 返された器は書き換え
 * ない）。scale の payload は block の先頭からの f32 の列（Session 構築が `subarray(0, payloadBytes)` で読む形）。
 */
const scaledScaleContainer = (
  opened: OpenedContainer,
  graph: string,
  weight: string,
  factor: number,
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
      const scaled = bytes.slice();
      const values = new Float32Array(scaled.buffer, 0, scale.payloadBytes / 4);
      for (let index = 0; index < values.length; index += 1) values[index] *= factor;
      return scaled;
    },
  };
};

const gib = (bytes: number): string => `${(bytes / 2 ** 30).toFixed(2)} GiB`;

/** 1 forward の観測（壁時間と、run の直後に生きている確保）。 */
type Run = {
  readonly label: string;
  readonly wallMs: number;
  readonly weightBytes: number;
  readonly arenaBytes: number;
  readonly backingBytes: number;
};

const observeRun = (label: string, diagnostics: SessionDiagnostics, wallMs: number): Run => ({
  label,
  wallMs,
  weightBytes: diagnostics.weights.allocatedBytes,
  arenaBytes: diagnostics.lastRun?.allocatedBytes ?? 0,
  backingBytes: diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
});

const formatRun = (run: Run): string =>
  `${run.label}: 壁 ${(run.wallMs / 1000).toFixed(2)} s・確保 ${gib(
    run.weightBytes + run.arenaBytes + run.backingBytes,
  )}（重み ${gib(run.weightBytes)}・アリーナ ${gib(run.arenaBytes)}・常駐 ${gib(
    run.backingBytes,
  )}）`;

/** 1 forward を回して出力（f32 `[1, L, 4096]`）と観測を返す（次の run の前に解放を待つ）。 */
const forward = async (
  session: Session,
  prepared: PreparedModel,
  gpu: GpuContext,
  inputs: Umt5SessionInputs,
  label: string,
): Promise<{ readonly output: Extract<Tensor, { readonly dtype: "f32" }>; readonly run: Run }> => {
  const started = performance.now();
  const outputs = await session.run(inputs);
  const wallMs = performance.now() - started;
  const output = outputs[prepared.graph.outputs[0]];
  if (output.dtype !== "f32") throw new Error(`${label}: 出力が ${output.dtype}`);
  const run = observeRun(label, session.diagnostics(), wallMs);
  await settleReleases(gpu);
  return { output, run };
};

/**
 * A resource a describe's `beforeAll` opens (device, Session, encoder), or a loud failure when that
 * `beforeAll` did not get that far. The `afterAll` hooks read the raw `undefined` to close only what
 * was opened.
 */
const requireOpened = <T>(value: T | undefined, what: string): T => {
  if (value === undefined) throw new Error(`the ${what} was not opened by beforeAll`);
  return value;
};

const results = openResults("wan-umt5");

// 完全に空の環境だけ「生成していない」として SKIP。何か 1 つでもあれば欠けは FAIL。
test.skipIf(!ANY_PRESENT)(
  "Wan umT5 資産: i8 系列の容器と全ケース（決定用 6 本・受入れ 4 本）の golden が揃っている",
  () => {
    assert(MODEL_PRESENT, `${SERIES_DIR.pathname}${MODEL_FILE} が無い`);
    assertEquals(
      expectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `golden の欠け（生成: ${GENERATE}）`,
    );
  },
);

test.skipIf(!ASSETS_AVAILABLE || tokenizerText === undefined)(
  "Wan umT5 ホスト（GPU 不要）: TS のトークナイザの id 列と TS のバケット表が golden の入力（Python の経路）と" +
    "ビット一致し、golden が今の容器から採られている",
  async () => {
    const encoder = tokenizerEncoder();
    const part0 = await containerPart0Sha256();
    for (const { name, role } of CASES) {
      const golden = await loadGolden(name);
      assertEquals(golden.meta.role, role, `${name}: 役割`);
      assertEquals(
        golden.meta.part0Sha256,
        part0,
        `${name}: golden を採った容器（part 0 の sha256）が今の容器と違う — 生成し直す: ${GENERATE}`,
      );
      hostInputs(encoder, golden, name);
    }
  },
);

/** 1 ケースの r（帯の候補の材料 — 判定の前に積むので帯の外の回も残る）。 */
type CaseRatio = { readonly name: string; readonly role: CaseRole; readonly ratio: number };

/**
 * 未導出の帯の候補を出して赤で止める（決定用の最悪 r × 5 を有効数字 2 桁へ切り上げ — 受入れの r は並べるが
 * 候補には使わない）。決定用のどれかが r を出す前に落ちた回は候補を出さない（欠けた最悪は最悪ではない）。
 */
const bandCandidate = (ratios: readonly CaseRatio[]): string => {
  const expected = CASES.filter(({ role }) => role === "band").length;
  const decided = ratios.filter(({ role }) => role === "band");
  const accepted =
    ratios
      .filter(({ role }) => role === "accept")
      .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`)
      .join(" / ") || "なし";
  if (decided.length !== expected) {
    return (
      `決定用 ${expected} 本のうち r が出たのは ${decided.length} 本 — 候補にしない（落ちたケースを先に調べる）。` +
      `受入れの r: ${accepted}`
    );
  }
  const worst = Math.max(...decided.map(({ ratio }) => ratio));
  const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
  const candidate = Number((Math.ceil((worst * 5) / unit) * unit).toPrecision(2));
  return (
    `決定用 ${expected} 本（${decided
      .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`)
      .join(" / ")}）の最悪 r ${worst.toPrecision(3)} × 5 = ${(worst * 5).toPrecision(
      3,
    )}（有効数字 2 桁へ切り上げ ${candidate}）。` + `受入れの r（帯の決定に使わない）: ${accepted}`
  );
};

describe.skipIf(!ASSETS_AVAILABLE || tokenizerText === undefined || !GPU_AVAILABLE)(
  "Wan umT5 (real GPU, default run): the graph takes input_ids and the bucket table and returns one " +
    "output, and each case's TS-input forward has the golden's output shape with no non-finite value",
  () => {
    // Assigned in beforeAll; the its do not run when beforeAll throws.
    let encoder!: WanPromptEncoder;
    let prepared!: PreparedModel;
    let gpu: GpuContext | undefined;
    let session: Session | undefined;
    let deviceLost: string | undefined;

    beforeAll(async () => {
      encoder = tokenizerEncoder();
      const graph = graphName();
      prepared = prepareContainer(
        await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR)),
        graph,
      );
      assertEquals(
        prepared.graph.inputs.map(({ name }) => name),
        [INPUT_IDS_KEY, BUCKETS_KEY],
        "グラフ入力",
      );
      assertEquals(prepared.graph.outputs.length, 1, "グラフの出力は 1 本");
      gpu = await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(gpu);
      session = await prepared.createContainerSession(gpu);
    });

    afterAll(async () => {
      // Same order as the former finally blocks: the Session first, then the device.
      if (gpu === undefined) return;
      try {
        if (session !== undefined) {
          await session.dispose();
          await settleReleases(gpu);
        }
      } finally {
        await settleReleases(gpu);
        gpu.destroy();
      }
    });

    for (const { name, role } of CASES) {
      it(`${name}（${role}）`, async () => {
        const device = requireOpened(gpu, "device");
        const golden = await loadGolden(name);
        const { inputs } = hostInputs(encoder, golden, name);
        const { output } = await forward(
          requireOpened(session, "Session"),
          prepared,
          device,
          inputs,
          name,
        );
        assertEquals(output.shape, golden.outputShape, `${name}: 出力の形`);
        const nonFinite = output.data.filter((value) => !Number.isFinite(value)).length;
        assertEquals(nonFinite, 0, `${name}: 非有限`);
        assertEquals(deviceLost, undefined, "device lost");
      });
    }
  },
);

/**
 * Registers the hooks of one Session scope in the caller's describe: `beforeAll` opens the Session
 * over `prepared()` on `gpu()`, `afterAll` disposes it and waits for the releases. Each scope gets
 * its own describe so two Sessions never coexist (the scopes run one after another, in registration
 * order). Returns the accessor the its use.
 */
const registerSessionScope = (
  prepared: () => PreparedModel,
  gpu: () => GpuContext,
): (() => Session) => {
  let session: Session | undefined;
  beforeAll(async () => {
    session = await prepared().createContainerSession(gpu());
  });
  afterAll(async () => {
    if (session === undefined) return;
    await session.dispose();
    await settleReleases(gpu());
  });
  return () => requireOpened(session, "Session");
};

describe.skipIf(
  !UPSTREAM_PARITY || !ASSETS_AVAILABLE || tokenizerText === undefined || !GPU_AVAILABLE,
)(
  "Wan umT5 移植の門（実 GPU / CPU 層逐次 f64）: TS の入力で回した 1 forward の正規化比 r が帯の内・故障注入 4 件は" +
    "帯の外" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const band = UMT5_NORMALIZED_BAND;
    const ratios: CaseRatio[] = [];
    // Assigned in beforeAll; the its do not run when beforeAll throws.
    let encoder!: WanPromptEncoder;
    let graph!: string;
    let opened!: OpenedContainer;
    let prepared!: PreparedModel;
    let gpu: GpuContext | undefined;
    let deviceLost: string | undefined;
    const device = (): GpuContext => requireOpened(gpu, "device");

    beforeAll(async () => {
      encoder = tokenizerEncoder();
      graph = graphName();
      opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR));
      prepared = prepareContainer(opened, graph);
      assertEquals(
        prepared.graph.inputs.map(({ name }) => name),
        [INPUT_IDS_KEY, BUCKETS_KEY],
        "グラフ入力",
      );
      assertEquals(prepared.graph.outputs.length, 1, "グラフの出力は 1 本");
      gpu = await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(gpu);
    });

    afterAll(async () => {
      // Runs after the Session scopes below have closed theirs (the former outer finally block).
      if (gpu === undefined) return;
      await settleReleases(gpu);
      gpu.destroy();
    });

    describe("main Session", () => {
      const session = registerSessionScope(() => prepared, device);
      for (const { name, role } of CASES) {
        it(`${name}（${role}）`, async () => {
          let caseNote = "";
          await runRecordedCase(
            results,
            {
              id: name,
              failureNote: () => caseNote,
            },
            async ({ measurements }) => {
              const golden = await loadGolden(name);
              const { ids, inputs } = hostInputs(encoder, golden, name);
              const referenceRatio = referenceRatioOf(golden, name);
              const { output, run } = await forward(session(), prepared, device(), inputs, name);
              assertEquals(output.shape, golden.outputShape, `${name}: 出力の形`);
              const diff = difference(output.data, golden.expected);
              const diffF32 = difference(output.data, golden.reference);
              const referenceDiff = difference(golden.reference, golden.expected);
              const ratio = ratioOf(diff) / referenceRatio;
              ratios.push({ name, role, ratio });
              const unbounded = { atol: Number.POSITIVE_INFINITY, rtol: 0 };
              measurements.push(
                {
                  output: "output@f64-reference",
                  maxAbs: diff.maxAbs,
                  maxRel: diff.maxRel,
                  // 判定と同じ境界を絶対値へ写す（未導出の回は無限の帯 — 判定は下で赤にする）。
                  tolerance:
                    band === undefined
                      ? unbounded
                      : { atol: band * referenceRatio * diff.referenceMaxAbs, rtol: 0 },
                  stage: "karume",
                },
                {
                  output: "output@f32-reference",
                  maxAbs: diffF32.maxAbs,
                  maxRel: diffF32.maxRel,
                  tolerance: unbounded,
                  stage: "karume",
                },
                {
                  output: "f32-reference@f64-reference",
                  maxAbs: referenceDiff.maxAbs,
                  maxRel: referenceDiff.maxRel,
                  tolerance: unbounded,
                  stage: "karume",
                },
              );
              const notes = [
                `L ${ids.length}`,
                formatRatio("GPU / f64 参照", diff, referenceRatio),
                `CPU f32 参照 / f64 参照（正規化の分母）: 比 ${referenceRatio.toExponential(3)}`,
                formatRun(run),
              ];
              assertEquals(diff.nonFinite, 0, `${name}: 非有限`);

              const faults: { label: string; ratio: number }[] = [];
              if (role === "accept") {
                const injected: readonly (readonly [string, Umt5SessionInputs])[] = [
                  ["相対位置の表を 1 ずらす", umt5SessionInputs(ids, shiftedBuckets(ids.length))],
                  [
                    "id 列の中ほどの 1 トークンを別の id に",
                    umt5SessionInputs(
                      replacedToken(ids),
                      buildUmt5RelativePositionBuckets(ids.length),
                    ),
                  ],
                ];
                for (const [label, faulty] of injected) {
                  const { output: faultOutput } = await forward(
                    session(),
                    prepared,
                    device(),
                    faulty,
                    `${name} ${label}`,
                  );
                  const faultDiff = difference(faultOutput.data, golden.expected);
                  faults.push({ label, ratio: ratioOf(faultDiff) / referenceRatio });
                  notes.push(`故障注入 ${formatRatio(label, faultDiff, referenceRatio)}`);
                }
              }
              // 判定の前に出す（帯の外へ出た回も比が手元に残る）。
              caseNote = notes.join(" / ");
              console.log(`[wan-umt5] ${name}: ${caseNote}`);
              assertEquals(deviceLost, undefined, "device lost");
              judgeBand(band, name, ratio, faults);
              return { status: "pass", note: caseNote };
            },
          );
        });
      }
    });

    const swapLabel = "層 0 / 1 の相対位置の表の取り違え";
    // 本体の Session を畳んだ後に張る（2 本を同時に載せない — 重み 5.3 GiB × 2 は B570 の天井
    // 9,600 MiB を越える）。
    describe(`fault Session: ${swapLabel}`, () => {
      let swapped!: PreparedModel;
      beforeAll(() => {
        swapped = prepareContainer(swappedTablesContainer(opened, graph, SWAPPED_TABLES), graph);
      });
      const faultSession = registerSessionScope(() => swapped, device);
      for (const { name } of CASES.filter(({ role }) => role === "accept")) {
        const id = `${name}/table-swap`;
        it(`${id}（故障注入 ${swapLabel} → 帯の外）`, async () => {
          let note = "";
          await runRecordedCase(results, { id, failureNote: () => note }, async () => {
            const golden = await loadGolden(name);
            const { inputs } = hostInputs(encoder, golden, name);
            const referenceRatio = referenceRatioOf(golden, name);
            const { output } = await forward(faultSession(), swapped, device(), inputs, id);
            const faultDiff = difference(output.data, golden.expected);
            const ratio = ratioOf(faultDiff) / referenceRatio;
            note = `故障注入 ${formatRatio(swapLabel, faultDiff, referenceRatio)}（${
              band === undefined ? "帯は未導出" : `帯の ${(ratio / band).toFixed(1)} 倍`
            }）`;
            console.log(`[wan-umt5] ${id}: ${note}`);
            if (band === undefined) throw new Error(`${id}: 帯が未導出（故障注入の r ${ratio}）`);
            assert(ratio > band, `${id}: r ${ratio.toPrecision(3)} が帯 ${band} の内に収まった`);
            return { status: "pass", note };
          });
        });
      }
    });

    // i8 に固有の故障注入（scale の倍率）。前の Session を畳んでから別の Session を張る
    // （表の取り違えと同じ理由 — 2 本を同時に載せない）。
    const label = `${I8_SCALE_FAULT_WEIGHT} の scale を ${GROSS_SCALE_FACTOR} 倍`;
    describe(`fault Session: ${label}`, () => {
      let scaled!: PreparedModel;
      beforeAll(() => {
        scaled = prepareContainer(
          scaledScaleContainer(opened, graph, I8_SCALE_FAULT_WEIGHT, GROSS_SCALE_FACTOR),
          graph,
        );
      });
      const scaledSession = registerSessionScope(() => scaled, device);
      for (const { name } of CASES.filter(({ role }) => role === "accept")) {
        const id = `${name}/scale-fault`;
        it(`${id}（故障注入 ${label} → 帯の外）`, async () => {
          let note = "";
          await runRecordedCase(results, { id, failureNote: () => note }, async () => {
            const golden = await loadGolden(name);
            const { inputs } = hostInputs(encoder, golden, name);
            const referenceRatio = referenceRatioOf(golden, name);
            const { output } = await forward(scaledSession(), scaled, device(), inputs, id);
            const faultDiff = difference(output.data, golden.expected);
            const ratio = ratioOf(faultDiff) / referenceRatio;
            note = `故障注入 ${formatRatio(label, faultDiff, referenceRatio)}（${
              band === undefined ? "帯は未導出" : `帯の ${(ratio / band).toPrecision(3)} 倍`
            }）`;
            console.log(`[wan-umt5] ${id}: ${note}`);
            assertEquals(deviceLost, undefined, "device lost");
            if (band === undefined) {
              throw new Error(`${id}: 帯が未導出（故障注入の r ${ratio}）`);
            }
            assert(ratio > band, `${id}: r ${ratio.toPrecision(3)} が帯 ${band} の内に収まった`);
            return { status: "pass", note };
          });
        });
      }
    });

    if (band === undefined) {
      it("帯の候補（未導出）", () => {
        throw new Error(`帯が未導出 — ${bandCandidate(ratios)}`);
      });
    }
  },
);

/**
 * `umt5.Encoder`（`@karume/models/umt5`）の通しの e2e（実 GPU）— 配布形ミラー `models/karume-umt5-xxl` を
 * `fromPretrained(nodeDirectory(…))` で組み、Wan の umT5 の移植の門（`e2e_wan_umt5_test.ts`）と**同じ golden 10 本**を
 * **同じ判定**（正規化比 r が帯の内 — `helpers/umt5-golden.ts`）で見る。
 *
 * 入力は golden のメタの `cleaned`（上流の前処理 `prompt_clean` の後の文字列）: golden の `input_ids` はこれを
 * transformers 5.14.1 に通した列で、単体の encoder は前処理をしない（Wan の前処理の後の文字列は単体のトークナイザの
 * 受理集合に入る — `umt5_tokenizer_test.ts`）。
 *
 * ## 門
 *
 * 1. **ホスト（GPU 不要）**: ミラーのトークナイザ資産（manifest の `tokenizer` — hub が sha256 を照合して渡す）で組んだ
 *    トークナイザが、`cleaned` から golden の `input_ids` と**ビット一致**する列を出す。golden を採った容器の part 0 の
 *    sha256 が、ミラーの manifest が宣言する `text_encoder` の part 0 と同じ（= golden は配る重みから採った）。
 * 2. **実 GPU**: `tokenize(cleaned)` と `encode(cleaned).inputIds` が golden の `input_ids` と一致し、`hiddenStates`
 *    （`[L, 4096]`）の正規化比 r が帯の内・非有限 0（`judgeBand` — 故障注入は Wan の門が持つので、ここは渡さない）。
 * 3. **実 GPU**: 中断済みの signal は run の前に `signal.reason` を包まずに投げ、同じ encoder で次の `encode` が回る。
 *    `dispose` の後の `encode` は落ちる。
 *
 * ミラー・golden・GPU のどれかが無い環境は明示 SKIP（ADR 0005）— ミラーごと無い形は門番 `distribution_gate_test.ts`、
 * golden の欠けは `e2e_wan_umt5_test.ts` の完全性テストが FAIL にする。
 */

/** 配布形ミラー（`dist.py --pipeline umt5` の既定の出力先）。 */
const DIST_ROOT = new URL("../../../models/karume-umt5-xxl/", import.meta.url);
const ASSEMBLE = "cd tools/export-recipes && uv run python dist.py --pipeline umt5";
/** 席（明示 — 省略すると manifest の既定へ黙って移る。golden は i8 の重みから採った）。 */
const MIRROR_MODEL = "xxl";
const MIRROR_QUANT = "i8";
/** manifest の部品名とトークナイザ資産のキー（`src/umt5/encoder.ts` と同じ綴り）。 */
const TEXT_ENCODER = "text_encoder";
const TOKENIZER = "tokenizer";

const DIST_PRESENT = filePresent(new URL("karume.json", DIST_ROOT));

if (!DIST_PRESENT) {
  console.warn(
    `[karume] 配布形ミラー ${DIST_ROOT.pathname} が無いため umT5 の encoder の e2e を SKIP する。組み立て: ` +
      `${ASSEMBLE}（全 SKIP は門番 distribution_gate_test.ts が FAIL にする）`,
  );
} else if (!GOLDENS_PRESENT) {
  console.warn(
    `[karume] umT5 の golden（${
      goldenUrl(CASES[0].name).pathname
    } など）が揃っていないため umT5 の encoder の ` + `e2e を SKIP する。生成: ${GENERATE}`,
  );
}

/** ミラーの選択（manifest の宣言 — 席は明示）。 */
const loadMirror = async () => {
  const loaded = await loadManifest(nodeDirectory(DIST_ROOT));
  return {
    loaded,
    selection: resolveSelection(loaded.manifest, { model: MIRROR_MODEL, quant: MIRROR_QUANT }),
  };
};

/** 1 ケースの入力（golden のメタの前処理の後の文字列）と、その golden の id 列との最初の食い違い。 */
const idsMismatch = (ids: Int32Array, golden: Golden): number => firstMismatch(ids, golden.ids);

test.skipIf(!DIST_PRESENT || !GOLDENS_PRESENT)(
  "umT5 encoder ホスト（GPU 不要）: ミラーのトークナイザが golden の前処理の後の文字列から golden の id 列を出し、" +
    "golden はミラーの text_encoder の part 0 から採られている",
  async () => {
    const { loaded, selection } = await loadMirror();
    assert(Object.hasOwn(selection.assets, TOKENIZER), `ミラーに資産 '${TOKENIZER}' が無い`);
    const assets = await fetchAssets(loaded, { [TOKENIZER]: selection.assets[TOKENIZER] });
    const tokenizer = new Umt5Tokenizer(
      parseUmt5TokenizerAsset(JSON.parse(new TextDecoder().decode(assets[TOKENIZER]))),
    );
    const part0 = selection.containers[TEXT_ENCODER].parts[0].sha256;
    for (const { name } of CASES) {
      const golden = await loadGolden(name);
      assertEquals(
        golden.meta.part0Sha256,
        part0,
        `${name}: golden を採った容器（part 0 の sha256）がミラーの text_encoder と違う`,
      );
      const ids = tokenizer.tokenize(golden.meta.cleaned, name);
      const at = idsMismatch(ids, golden);
      assertEquals(at, -1, `${name}: id 列が golden と ${at} 番目で割れる（${ids.length} 本）`);
    }
  },
);

/**
 * Registers the hooks that build the mirror's `umt5.Encoder` on a test device in the caller's
 * describe: `beforeAll` acquires the device and runs `fromPretrained`, `afterAll` disposes the
 * encoder (idempotent — an it may already have disposed it), waits for the releases, fails on a
 * device loss and destroys the device. Returns the accessors the its use and the device-lost message (undefined while none).
 */
const registerMirrorEncoder = (): {
  readonly gpu: () => GpuContext;
  readonly encoder: () => umt5.Encoder;
  readonly deviceLost: () => string | undefined;
} => {
  let gpu: GpuContext | undefined;
  let encoder: umt5.Encoder | undefined;
  let deviceLost: string | undefined;
  beforeAll(async () => {
    gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    encoder = await umt5.Encoder.fromPretrained(nodeDirectory(DIST_ROOT), {
      gpu,
      model: MIRROR_MODEL,
      quant: MIRROR_QUANT,
    });
  });
  afterAll(async () => {
    if (gpu === undefined) return;
    try {
      if (encoder !== undefined) {
        await encoder.dispose();
        await settleReleases(gpu);
      }
      // The former test bodies checked this after the dispose (a device loss during dispose).
      assertEquals(deviceLost, undefined, "device lost");
    } finally {
      gpu.destroy();
    }
  });
  return {
    gpu: () => requireOpened(gpu, "device"),
    encoder: () => requireOpened(encoder, "encoder"),
    deviceLost: () => deviceLost,
  };
};

describe.skipIf(!DIST_PRESENT || !GOLDENS_PRESENT || !GPU_AVAILABLE)(
  "umT5 encoder (real GPU, default run): built by fromPretrained(nodeDirectory), tokenize and inputIds " +
    "match the golden ids and hiddenStates has the golden's shape with no non-finite value for all 10 " +
    "goldens; an aborted signal throws its reason before the run, and encode fails after dispose",
  () => {
    const mirror = registerMirrorEncoder();

    for (const { name } of CASES) {
      it(name, async () => {
        const encoder = mirror.encoder();
        const golden = await loadGolden(name);
        const text = golden.meta.cleaned;
        const tokenized = encoder.tokenize(text);
        assertEquals(idsMismatch(tokenized, golden), -1, `${name}: tokenize の id 列`);

        const encoding = await encoder.encode(text);
        assertEquals(idsMismatch(encoding.inputIds, golden), -1, `${name}: inputIds`);
        assertEquals(encoding.length, golden.ids.length, `${name}: length`);
        const [, rows, width] = golden.outputShape;
        assertEquals([encoding.length, encoding.hiddenSize], [rows, width], `${name}: 形`);
        const nonFinite = encoding.hiddenStates.filter((value) => !Number.isFinite(value)).length;
        assertEquals(nonFinite, 0, `${name}: 非有限`);
        assertEquals(mirror.deviceLost(), undefined, "device lost");
        await settleReleases(mirror.gpu());
      });
    }

    it("中断済みの signal は run の前に reason をそのまま投げ、次の encode は回る", async () => {
      const encoder = mirror.encoder();
      const controller = new AbortController();
      const reason = new Error("umt5-encoder: テストの中断");
      controller.abort(reason);
      const error = await assertRejects(() =>
        encoder.encode("a cat", { signal: controller.signal }),
      );
      assertStrictEquals(error, reason);
      const after = await encoder.encode("a cat");
      assertEquals(after.length, encoder.tokenize("a cat").length);
    });

    // The former test body's tail after its finally block (dispose, then the rejected encode).
    it("encode fails after dispose", async () => {
      const encoder = mirror.encoder();
      await encoder.dispose();
      await settleReleases(mirror.gpu());
      await assertRejects(() => encoder.encode("a cat"), Error, "after dispose");
      assertEquals(mirror.deviceLost(), undefined, "device lost");
    });
  },
);

describe.skipIf(!UPSTREAM_PARITY || !DIST_PRESENT || !GOLDENS_PRESENT || !GPU_AVAILABLE)(
  "umT5 encoder（実 GPU）: fromPretrained(nodeDirectory) で組み、golden 10 本の hiddenStates の正規化比 r が帯の内" +
    "（Wan の umT5 の移植の門と同じ golden・同じ帯）・tokenize と inputIds が golden の id 列" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const mirror = registerMirrorEncoder();

    for (const { name } of CASES) {
      it(name, async () => {
        const encoder = mirror.encoder();
        const golden = await loadGolden(name);
        const text = golden.meta.cleaned;
        const tokenized = encoder.tokenize(text);
        assertEquals(idsMismatch(tokenized, golden), -1, `${name}: tokenize の id 列`);

        const encoding = await encoder.encode(text);
        assertEquals(idsMismatch(encoding.inputIds, golden), -1, `${name}: inputIds`);
        assertEquals(encoding.length, golden.ids.length, `${name}: length`);
        const [, rows, width] = golden.outputShape;
        assertEquals([encoding.length, encoding.hiddenSize], [rows, width], `${name}: 形`);
        const diff = difference(encoding.hiddenStates, golden.expected);
        const referenceRatio = referenceRatioOf(golden, name);
        const ratio = ratioOf(diff) / referenceRatio;
        console.log(
          `[umt5-encoder] ${name}: L ${encoding.length} / ${formatRatio(
            "GPU / f64 参照",
            diff,
            referenceRatio,
          )}`,
        );
        assertEquals(diff.nonFinite, 0, `${name}: 非有限`);
        assertEquals(mirror.deviceLost(), undefined, "device lost");
        judgeBand(UMT5_NORMALIZED_BAND, name, ratio, []);
        await settleReleases(mirror.gpu());
      });
    }
  },
);
