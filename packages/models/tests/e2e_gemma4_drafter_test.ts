// 実重み Gemma 4 E2B の **MTP drafter**（投機デコードの draft 側）の検収門 — ADR
// [0096](../../../docs/decisions/0096-speculative-decoding.md) 段 2。
//
// 検収するのは「drafter が target の KV と埋め込み表を**借りて**回り、torch の drafter と同じ
// draft を出す」ことである。段 2 に投機ループ（verify と受理・棄却）はまだ無いので、ここが見るのは
// 1 サイクルぶんの draft だけ:
//
// ① 配布形（GPU 不要）… manifest の weights に `drafter` が居て、既定 quant が両方の役割を指す。
//    `ResolveOptions.weights` の絞り込みが効き、**投機を使わないロードで drafter の shard が
//    遅延資産へ落ちない**（落ちると `assertPleShardAssets` が「遅延資産 = PLE shard ちょうど」の
//    門で落ち、投機と無関係のロードまで壊れる — この検出器だけが GPU 無しで踏める）
// ② コンテナの形（GPU 不要）… 入力 6 本（順序込み）/ 出口 3 本 / external スロット 4 本が
//    target と同名同形 / 共有 initializer 1 本が target の initializer へ解決できる
// ③ **draft の一致**（実 GPU）… 3 ケース × 200 サイクル × 3 段 = 1,800 本の draft token が、
//    torch drafter の golden と **99% 以上**一致する。karume 側の hidden は i4 の target が出した
//    もの（golden の hidden は f32 の代理 target）なので、ビット一致は求められない — 一致率の
//    門である。target の token 列は golden の継続列で teacher forcing する
// ④ 寿命（実 GPU）… 借り手が生きている間は貸し手 context も貸し手 Session も dispose できず、
//    順序（借り手 → 貸し手）どおりなら通る
//
// MUST: 一致率の門は**実測に合わせない**。落ちたら段別・位置別の分布を報告に出し、赤のまま残す
// （drafter の質は投機の速度そのもので、門を緩めると「動くが速くならない」形が緑になる）。
//
// ## 資産
//
// 配布形ミラー `models/karume-gemma4/`（`dist.py --pipeline gemma4` が組む — drafter 込み）と
// golden `outputs/series/gemma4-e2b-drafter/`。どちらもリポジトリ管理外で、ミラーが無い環境では
// **明示 SKIP**、ミラーがあるのに golden が欠けている形は SKIP でなく **FAIL** にする。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import {
  type Manifest,
  MANIFEST_FILENAME,
  ManifestReferenceError,
  parseManifest,
  type ResolvedFiles,
  resolveFiles,
} from "@karume/hub";
import {
  acquireGpu,
  ExecutionError,
  type GenerationContext,
  parseSafetensors,
  prepareModel,
  type SafetensorsFile,
  type Session,
  type Tensor,
} from "@karume/runtime";
import { denoDirectory } from "@karume/hub/deno";
import { assertPleShardAssets, GEMMA4_STATE_ATTENTION_REDUCE } from "../src/gemma/pipeline.ts";
import type { Gemma4Assets } from "../src/gemma/pipeline.ts";
// MUST: 入口は**公開面**から取る（`src/...` を直に掴むと、面が痩せていても門が緑のままになる）。
import { type Gemma4ChatMessage, Gemma4Pipeline } from "../gemma.ts";
import { parseGemma4PipelineConfig } from "../src/gemma/config.ts";
import { createGemma4Ple, type Gemma4Ple, parseGemma4PleIndex } from "../src/gemma/ple.ts";
import { gemma4RopeInputNames, gemma4RopeInputs, type Gemma4RopeSpec } from "../src/gemma/rope.ts";
import {
  admitGemma4Drafter,
  draftOnce,
  GEMMA4_DRAFT_STEPS,
  type Gemma4Drafter,
  openDrafterContext,
} from "../src/gemma/speculative.ts";
import { planPrefillChunks } from "../src/generation/greedy.ts";
import { readShard, resolveShards, streamShards } from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { allResidentBytes, allResidentPleBytesOfMirror } from "./helpers/ple-budget.ts";
import { openPleShardAt } from "./helpers/ple-source.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4/", import.meta.url);
const GOLDEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-drafter/", import.meta.url);

/** SKIP 時にそのまま貼れる組み立てコマンド。 */
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4" +
  "（drafter 系列と golden は … python -m gemma4.export_drafter）";

/** 配布形の役割名（`Gemma4Pipeline` の同名定数と同じ綴り — 焼く側は `distribution.py`）。 */
const MODEL = "model";
const DRAFTER = "drafter";
/**
 * 全量常駐で受け取る assets（`Gemma4Pipeline` の `EAGER_ASSETS` と同じ 2 本）。
 *
 * ①の遅延資産シミュレーションで要る — この 2 本を除いた残りが「遅延側」で、そこが PLE shard
 * ちょうどであることを本番と同じ門（{@link assertPleShardAssets}）に掛ける。
 */
const EAGER_ASSETS: readonly string[] = ["tokenizer", "ple_index"];

/** golden のケース（正本は `export_drafter.py` の GOLDEN_CASES）。 */
const CASES = ["short-en", "readme-recipes", "readme-exporter"] as const;
const GOLDEN_PREFIX = "drafter-golden.";
const SUFFIX = ".safetensors";

/** 一致率の門（分母 = 3 ケース × 200 サイクル × 3 段）。 */
const MATCH_FLOOR = 0.99;

/** 実資産の形（config の `hidden_size` / `vocab_size` と、貸し手の共有 initializer 名）。 */
const HIDDEN = 1536;
const VOCAB = 262144;
const SHARED_TENSOR = "model.lm_head.weight";
const SHARED_LENDER = "p_model_lm_head_weight";

/** グラフ入力の名前（正本は `export_product.py` の定数）。 */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";
const CAPACITY_SYMBOL = "C";
/** prefill の刻み（配布形の宣言 `chunkLength` と同値 — 実行時ノブなので明示する）。 */
const CHUNK_LENGTH = 768;

const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const goldenPath = (name: string): URL => new URL(`${GOLDEN_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT);

const MIRROR_PRESENT = exists(new URL(MANIFEST_FILENAME, MIRROR_DIR)) &&
  exists(new URL("e2b/drafter/model.i8-00001-of-00002.safetensors", MIRROR_DIR));
const GOLDENS_PRESENT = CASES.every((name) => exists(goldenPath(name)));
const AVAILABLE = MIRROR_PRESENT && GOLDENS_PRESENT;

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] drafter 入りの配布形ミラー（${MIRROR_DIR.pathname}）が無いため MTP drafter 検収を` +
      ` SKIP する。組み立て: ${ASSEMBLE_COMMAND}`,
  );
}

const manifest: Manifest | undefined = MIRROR_PRESENT
  ? parseManifest(Deno.readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)))
  : undefined;

/** ミラーの manifest（SKIP しない位置でだけ呼ぶ）。 */
const mirrorManifest = (): Manifest => {
  assert(manifest !== undefined, "配布形ミラーの manifest が読めていない");
  return manifest;
};

/**
 * 依存資産の完全性（欠落を SKIP に畳まない）。ミラー（自系列に相当）があるのに golden が
 * 欠けているのは未生成ではなく欠損なので、SKIP でなく FAIL にする。
 */
Deno.test({
  name: "gemma4 drafter 資産: golden 3 ケースが揃っている",
  ignore: !MIRROR_PRESENT,
  fn: () => {
    for (const name of CASES) {
      assert(
        exists(goldenPath(name)),
        `${GOLDEN_PREFIX}${name}${SUFFIX} が ${GOLDEN_ROOT.pathname} に無い` +
          `（drafter 入りのミラーはあるのに期待値が欠けている — 生成: ${ASSEMBLE_COMMAND}）`,
      );
    }
  },
});

// ---------------------------------------------------------------------------
// ① 配布形（GPU 不要）— manifest の役割と `ResolveOptions.weights` の絞り込み
// ---------------------------------------------------------------------------

/**
 * `loadShardComponents` の遅延側（= 全量常駐させない資産）を取得キーの表から再現する。
 *
 * MUST: 本番と同じ式で作る — 「コンポーネントとして消費したキー」と `eagerAssets` を除いた残りが
 * 遅延側である（`src/hub/components.ts`）。drafter の shard は**コンポーネントに指定しない限り
 * 消費されない**ので、取得キーの表に残っているとここへ落ちる。
 */
const deferredKeys = (files: ResolvedFiles, componentKeys: readonly string[]): string[] => {
  const consumed = new Set<string>();
  for (const key of Object.keys(files)) {
    for (const component of componentKeys) {
      if (key === component || key.startsWith(`${component}[`)) consumed.add(key);
    }
  }
  return Object.keys(files).filter((key) => !consumed.has(key) && !EAGER_ASSETS.includes(key));
};

const pleIndex = () =>
  parseGemma4PleIndex(JSON.parse(Deno.readTextFileSync(new URL("e2b/ple/ple.json", MIRROR_DIR))));

Deno.test({
  name: "gemma4 drafter 配布形: manifest の weights に drafter が居て quant が両方を指す",
  ignore: !MIRROR_PRESENT,
  fn: () => {
    const entry = mirrorManifest().models[mirrorManifest().defaultModel];
    assertEquals(Object.keys(entry.weights).sort(), [DRAFTER, MODEL], "weights の役割");
    const quant = entry.quants[entry.defaultQuant];
    // 完全写像（parse が保証する）— target は混成（埋め込み i8 + linear i4）の "i4"、drafter は
    // **i8 単一**（linear まで i8 — i4 g32 に落とすと受理率が 1〜3 割落ちる・ADR 0096 追記）。
    assertEquals(Object.keys(quant.weights).sort(), [DRAFTER, MODEL], "quant の weights 写像");
    assertEquals(quant.weights[MODEL], "i4", "target の dtype ラベル");
    assertEquals(quant.weights[DRAFTER], "i8", "drafter の dtype ラベル");
    // requiredLimits は席に 1 つ（i8 は target の埋め込みと同じ席なので、宣言は増えない）。
    assert(quant.requiredLimits !== undefined, "既定 quant が requiredLimits を宣言していない");
  },
});

Deno.test({
  name: "gemma4 drafter 配布形: weights の絞り込みが drafter の shard を表から外す",
  ignore: !MIRROR_PRESENT,
  fn: () => {
    const target = resolveFiles(mirrorManifest(), { weights: [MODEL] });
    const both = resolveFiles(mirrorManifest(), { weights: [MODEL, DRAFTER] });

    const drafterKeys = (files: ResolvedFiles) =>
      Object.keys(files).filter((key) => key.startsWith(DRAFTER));
    assertEquals(drafterKeys(target), [], "絞ったのに drafter の取得キーが残っている");
    assertEquals(drafterKeys(both), [`${DRAFTER}[0]`, `${DRAFTER}[1]`], "drafter の shard 2 本");
    // 絞っても target 側と assets は 1 本も動かない。
    for (const key of Object.keys(target)) assertEquals(both[key], target[key], `${key} の参照`);

    assertThrows(
      () => resolveFiles(mirrorManifest(), { weights: [MODEL, "mtp"] }),
      ManifestReferenceError,
      "weights 'mtp' は manifest に無い",
    );
  },
});

Deno.test({
  name: "gemma4 drafter 配布形: 絞らないと投機なしのロードが遅延資産の門で落ちる",
  ignore: !MIRROR_PRESENT,
  fn: () => {
    const index = pleIndex();
    const where = "test";
    // 投機なし = コンポーネントは model 1 本。絞れば遅延側は PLE shard ちょうど。
    assertPleShardAssets(
      where,
      index,
      deferredKeys(resolveFiles(mirrorManifest(), { weights: [MODEL] }), [MODEL]),
    );
    // 投機あり = 2 本とも消費されるので、やはり PLE shard ちょうど。
    assertPleShardAssets(
      where,
      index,
      deferredKeys(resolveFiles(mirrorManifest(), { weights: [MODEL, DRAFTER] }), [MODEL, DRAFTER]),
    );
    // 絞らずに model だけを組むと drafter の shard が遅延側へ落ちる（= 投機と無関係のロードが
    // 壊れる）。この形が通ってしまうと、`weights` の絞り込みを外した退行が検出できない。
    assertThrows(
      () =>
        assertPleShardAssets(where, index, deferredKeys(resolveFiles(mirrorManifest()), [MODEL])),
      Error,
      "assets にあって索引に無い",
    );
  },
});

// ---------------------------------------------------------------------------
// ② コンテナの形（GPU 不要）
// ---------------------------------------------------------------------------

const targetShards = (): readonly URL[] =>
  resolveShards(new URL("e2b/model/model.i4.safetensors", MIRROR_DIR));
const drafterShards = (): readonly URL[] =>
  resolveShards(new URL("e2b/drafter/model.i8.safetensors", MIRROR_DIR));

/** ミラーの `pipelineConfig`（RoPE の宣言と容量の出どころ）。 */
const pipelineConfig = () => {
  const entry = mirrorManifest().models[mirrorManifest().defaultModel];
  return parseGemma4PipelineConfig(entry.pipelineConfig);
};

/** グラフ shard 1 本ずつを読んで宣言を取り出す（データ節は 1 バイトも要らない）。 */
const readGraphs = async () => ({
  target: prepareModel(await readShard(targetShards()[0])).graph,
  drafter: prepareModel(await readShard(drafterShards()[0])).graph,
});

Deno.test({
  name: "gemma4 drafter コンテナ: 入出力・external スロット・共有 initializer が target と組める",
  ignore: !MIRROR_PRESENT,
  fn: async () => {
    const { target, drafter } = await readGraphs();
    const config = pipelineConfig();

    // 入力 6 本（順序が契約）・出口 3 本（段順）。
    assertEquals(
      drafter.inputs.map((input) => input.name),
      ["token", "hidden", ...gemma4RopeInputNames()],
      "drafter グラフの入力（順序込み）",
    );
    assertEquals(drafter.outputs.length, GEMMA4_DRAFT_STEPS, "draft 出口の本数（= k）");
    assertEquals(drafter.symbols, [CAPACITY_SYMBOL], "記号（貸し手から継承する容量記号 1 本）");

    // external スロット 4 本（l13 sliding / l14 full の k/v）が target と同名同形。
    const external = Object.keys(drafter.states).filter((name) => drafter.states[name].external);
    assertEquals(external.length, 4, "external スロットの本数");
    assertEquals(external.length, Object.keys(drafter.states).length, "自前スロットが混ざっている");
    for (const name of external) {
      assertEquals(drafter.states[name].shape, target.states[name].shape, `${name} の形`);
    }

    // 共有 initializer は 1 本で、バイトを持たない（`tensor` が無い）。
    const shared = Object.keys(drafter.initializers).filter((name) =>
      drafter.initializers[name].shared !== undefined
    );
    assertEquals(shared.length, 1, "共有 initializer の本数");
    assertEquals(drafter.initializers[shared[0]].tensor, undefined, "共有宣言がバイトを持っている");
    assertEquals(
      drafter.initializers[shared[0]].shared?.tensor,
      SHARED_TENSOR,
      "共有 initializer が指す貸し手のテンソルキー",
    );

    // 門そのもの（`Gemma4Pipeline` が admission で通す 1 本）。貸し手の initializer 名まで解決する。
    const admitted = admitGemma4Drafter("test", drafter, {
      graph: target,
      rope: config.rope,
      hiddenSize: HIDDEN,
      capacitySymbol: CAPACITY_SYMBOL,
    });
    assertEquals(admitted.outputs, [...drafter.outputs], "admission が返す出口（段順）");
    assertEquals(admitted.hiddenSize, HIDDEN, "hidden の幅");
    assertEquals(admitted.sharedWeights, { [shared[0]]: SHARED_LENDER }, "共有重みの対応表");
  },
});

Deno.test({
  name: "gemma4 drafter コンテナ: 別世代の target と組んだ宣言は admission で落ちる",
  ignore: !MIRROR_PRESENT,
  fn: async () => {
    const { target, drafter } = await readGraphs();
    const config = pipelineConfig();
    const admit = (hiddenSize: number, capacitySymbol: string) =>
      admitGemma4Drafter("test", drafter, {
        graph: target,
        rope: config.rope,
        hiddenSize,
        capacitySymbol,
      });

    // hidden の幅が違う target（= 別世代）とは組めない。
    assertThrows(() => admit(HIDDEN + 1, CAPACITY_SYMBOL), Error, "drafter 入力 'hidden'");
    // 容量記号の綴りが違えば借り手 context の束縛点が無い。
    assertThrows(() => admit(HIDDEN, "K"), Error, "drafter グラフの記号");
    // 貸し手に共有テンソルが無い形（initializer を落とした target）。
    const stripped = {
      ...target,
      initializers: Object.fromEntries(
        Object.entries(target.initializers).filter(([name]) => name !== SHARED_LENDER),
      ),
    };
    assertThrows(
      () =>
        admitGemma4Drafter("test", drafter, {
          graph: stripped,
          rope: config.rope,
          hiddenSize: HIDDEN,
          capacitySymbol: CAPACITY_SYMBOL,
        }),
      Error,
      "共有 initializer",
    );
  },
});

// ---------------------------------------------------------------------------
// ②' pipeline の投機オプション（受理集合は資産を 1 バイトも読む前に落ちる）
// ---------------------------------------------------------------------------

Deno.test("Gemma4Pipeline: 段 2 の speculative は k = 3 だけ・fromAssets では受けない", async () => {
  // 受理集合の門は**取得元へ触る前**（`ref` は解決すらされない）。
  await assertRejects(
    () => Gemma4Pipeline.fromPretrained("owner/name", { speculative: { k: 2 } }),
    Error,
    "speculative.k 2 は段 2 では受けられない",
  );
  // `Gemma4Assets` に drafter の席が無いので、黙って投機なしで組まずに断る。
  const assets: Gemma4Assets = {
    config: {
      chunkLength: 2,
      maxChunkLength: 2,
      maxPosition: 2,
      capacity: 2,
      rope: {
        sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
        full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
      },
    },
    model: [],
    tokenizer: new Uint8Array(new ArrayBuffer(0)),
    pleIndex: new Uint8Array(new ArrayBuffer(0)),
    openPleShard: () => Promise.reject(new Error("test: 読まれないはず")),
  };
  await assertRejects(
    () => Gemma4Pipeline.fromAssets(assets, { speculative: {} }),
    Error,
    "speculative は受けられない",
  );
});

// ---------------------------------------------------------------------------
// ③④ 実 GPU — golden との突合と寿命
// ---------------------------------------------------------------------------

/** golden 1 ケース（`prompt [T]` / `tokens [N]` / `draft [N, k]`）。 */
type Golden = {
  readonly prompt: Int32Array<ArrayBuffer>;
  readonly tokens: Int32Array<ArrayBuffer>;
  readonly draft: Int32Array<ArrayBuffer>;
  readonly cycles: number;
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden に '${name}' が無い`);
  assertEquals(view.dtype, "I32", `golden '${name}' の格納 dtype`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const readGolden = async (name: string): Promise<Golden> => {
  const bytes = await Deno.readFile(goldenPath(name));
  const file = parseSafetensors(
    bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
      ? bytes.buffer
      : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const draftView = file.tensors.get("draft");
  assert(draftView !== undefined, `golden ${name} に 'draft' が無い`);
  assertEquals(draftView.shape[1], GEMMA4_DRAFT_STEPS, `${name}: draft の段数`);
  const tokens = goldenI32(file, "tokens");
  assertEquals(draftView.shape[0], tokens.length, `${name}: draft のサイクル数`);
  return {
    prompt: goldenI32(file, "prompt"),
    tokens,
    draft: goldenI32(file, "draft"),
    cycles: tokens.length,
  };
};

/** f32 の出力テンソルを取り出す（dtype の取り違えは黙って別の列になる）。 */
const f32Of = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  assert(tensor.data instanceof Float32Array, `${where}: f32 でない出力を f32 として読んでいる`);
  return tensor.data;
};

/** 1 ケースぶんの一致数（段別）。 */
type MatchCounts = { readonly perStep: number[]; readonly total: number; readonly cycles: number };

Deno.test({
  name: "gemma4 drafter 検収: draft が torch golden と 99% 以上一致する（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const config = pipelineConfig();
    const rope: Gemma4RopeSpec = config.rope;
    const { target: targetGraph, drafter: drafterGraph } = await readGraphs();
    const admitted = admitGemma4Drafter("test", drafterGraph, {
      graph: targetGraph,
      rope,
      hiddenSize: HIDDEN,
      capacitySymbol: CAPACITY_SYMBOL,
    });

    const index = pleIndex();
    const ple: Gemma4Ple = createGemma4Ple({
      index,
      openShard: (file) => openPleShardAt(new URL("e2b/ple/", MIRROR_DIR), file),
      vocabSize: VOCAB,
      // 全量常駐（読み直しゼロ — 予算は索引から導く）。
      maxResidentBytes: allResidentBytes(index),
    });

    const targetFiles = targetShards();
    const drafterFiles = drafterShards();
    const gpu = await acquireGpu();
    // 家族の既定（③PV の縮約形）で回す — 製品の decode と同じ実行形にする。
    const sessionOptions = { stateAttentionReduce: GEMMA4_STATE_ATTENTION_REDUCE };
    const target: Session = await prepareModel(await readShard(targetFiles[0])).createSession(
      gpu,
      streamShards(targetFiles.slice(1)),
      sessionOptions,
    );
    let drafterSession: Session | undefined;
    try {
      // 借りるのは埋め込み表 1 本（バイトは 1 つも複製されない）。
      const sharedWeights = Object.fromEntries(
        Object.keys(admitted.sharedWeights).map((
          borrower,
        ) => [borrower, target.exportWeight(admitted.sharedWeights[borrower])]),
      );
      drafterSession = await prepareModel(await readShard(drafterFiles[0])).createSession(
        gpu,
        streamShards(drafterFiles.slice(1)),
        { ...sessionOptions, sharedWeights },
      );
      const drafter: Gemma4Drafter = {
        session: drafterSession,
        outputs: admitted.outputs,
        rope,
        hiddenSize: admitted.hiddenSize,
      };
      // drafter はバイトを 1 本も常駐させ**ない**わけではない（自前の 4 層ぶんは持つ）が、
      // 借りた表ぶんは持たない = target の 1/20 未満に収まる。
      const drafterStorage = drafterSession.diagnostics().storage;
      const targetStorage = target.diagnostics().storage;
      assert(
        drafterStorage.residentCompressedBytes * 10 < targetStorage.residentCompressedBytes,
        `drafter の常駐 ${drafterStorage.residentCompressedBytes} バイトが target の ` +
          `${targetStorage.residentCompressedBytes} バイトに対して大きすぎる（表を借りていない）`,
      );

      const logitsName = targetGraph.outputs[0];
      const hiddenName = targetGraph.outputs[1];
      const counts: MatchCounts[] = [];

      /** 1 run ぶんの target 実行（返すのは選んだ行の hidden）。 */
      const runTarget = async (
        context: GenerationContext,
        ids: Int32Array<ArrayBuffer>,
        positions: Int32Array<ArrayBuffer>,
        queryLength: number,
        where: string,
      ): Promise<Float32Array<ArrayBuffer>> => {
        const perLayer = await ple.gather([...ids]);
        const outputs = await target.run(
          {
            [INPUT_IDS]: { dtype: "i32", shape: [1, ids.length], data: ids },
            [PER_LAYER_INPUTS]: perLayer,
            ...gemma4RopeInputs(rope, positions),
            [LAST_ROW]: { dtype: "i32", shape: [1], data: Int32Array.of(queryLength - 1) },
          },
          undefined,
          // MUST: `commit: "deferred"` は使わない（未 commit の run が残っていると draft は
          // 拒否される — draft は commit の後に採る）。
          { context, queryLength },
        );
        assertEquals(outputs[logitsName].shape, [1, 1, VOCAB], `${where}: logits の shape`);
        assertEquals(outputs[hiddenName].shape, [1, 1, HIDDEN], `${where}: hidden の shape`);
        return f32Of(outputs[hiddenName], `${where}: hidden`);
      };

      for (const name of CASES) {
        await t.step(`${name}: 200 サイクル × 3 段の draft を golden と突合`, async () => {
          const golden = await readGolden(name);
          const prompt = golden.prompt;
          const promptLength = prompt.length;
          // この会話が要る容量（prompt + 継続列ぶん — 配布形の既定 4096 では足りないケースがある）。
          const capacity = promptLength + golden.cycles;
          assert(
            capacity <= config.maxPosition,
            `${name}: 容量 ${capacity} が maxPosition ${config.maxPosition} を超える`,
          );
          const started = performance.now();
          const context = await target.createGenerationContext({
            bindings: { [CAPACITY_SYMBOL]: capacity },
            chunkLength: CHUNK_LENGTH,
          });
          const borrowed = await openDrafterContext(drafter, context);
          const perStep = Array.from({ length: GEMMA4_DRAFT_STEPS }, () => 0);
          /** 出た draft token の顔ぶれ（突合が自明でないことを毎回見るための欄）。 */
          const distinct = new Set<number>();
          try {
            // 相 1: prompt を prefill（最後の chunk の行が位置 T−1 の hidden を出す）。
            let hidden!: Float32Array<ArrayBuffer>;
            const chunks = planPrefillChunks(promptLength, CHUNK_LENGTH);
            for (const chunk of chunks) {
              const ids = new Int32Array(new ArrayBuffer(CHUNK_LENGTH * 4));
              const positions = new Int32Array(new ArrayBuffer(CHUNK_LENGTH * 4));
              for (let row = 0; row < chunk.queryLength; row += 1) {
                ids[row] = prompt[chunk.position + row];
                positions[row] = chunk.position + row;
              }
              hidden = await runTarget(
                context,
                ids,
                positions,
                chunk.queryLength,
                `${name} prefill@${chunk.position}`,
              );
            }

            // 相 2: サイクル t の draft。直前 token は t=0 が prompt 末尾・以降は継続列
            // （teacher forcing）で、query は論理位置 P−1 = T+t−1 に居る。
            for (let cycle = 0; cycle < golden.cycles; cycle += 1) {
              const position = promptLength + cycle - 1;
              const token = cycle === 0 ? prompt[promptLength - 1] : golden.tokens[cycle - 1];
              if (cycle > 0) {
                hidden = await runTarget(
                  context,
                  Int32Array.of(token),
                  Int32Array.of(position),
                  1,
                  `${name} decode@${cycle}`,
                );
              }
              assertEquals(context.pastLength, position + 1, `${name}: サイクル ${cycle} の論理長`);
              const draft = await draftOnce(drafter, borrowed, { token, hidden, position });
              assertEquals(draft.length, GEMMA4_DRAFT_STEPS, `${name}: draft の本数`);
              for (let step = 0; step < GEMMA4_DRAFT_STEPS; step += 1) {
                distinct.add(draft[step]);
                if (draft[step] === golden.draft[cycle * GEMMA4_DRAFT_STEPS + step]) {
                  perStep[step] += 1;
                }
              }
            }
          } finally {
            // MUST: 借り手 → 貸し手（逆順は runtime が拒否する）。
            await borrowed.dispose();
            await context.dispose();
          }
          // MUST: 突合が自明でないことを見る（drafter が 1 語を出し続ければ一致率は golden の
          // 受理率まで落ちるが、`argmax` が固まった形は「どの位置でも同じ語」として先に出る）。
          assert(
            distinct.size >= 32,
            `${name}: 出た draft token が ${distinct.size} 種類しかない（drafter の出口が固まっている）`,
          );
          const total = perStep.reduce((sum, value) => sum + value, 0);
          counts.push({ perStep, total, cycles: golden.cycles });
          console.log(
            `[e2e] gemma4 drafter ${name}: T=${promptLength} / ${golden.cycles} サイクル / 一致 ` +
              `${total}/${golden.cycles * GEMMA4_DRAFT_STEPS}（段別 ${
                perStep.map((hit, step) =>
                  `${step + 1}: ${(hit / golden.cycles * 100).toFixed(1)}%`
                ).join(" / ")
              }）/ ${(performance.now() - started).toFixed(0)}ms`,
          );
        });
      }

      await t.step("① 3 ケース合計の一致率が門を満たす", () => {
        const matched = counts.reduce((sum, entry) => sum + entry.total, 0);
        const drafts = counts.reduce(
          (sum, entry) => sum + entry.cycles * GEMMA4_DRAFT_STEPS,
          0,
        );
        assertEquals(counts.length, CASES.length, "突合したケース数");
        const rate = matched / drafts;
        console.log(
          `[e2e] gemma4 drafter 合計: ${matched}/${drafts} = ${(rate * 100).toFixed(2)}%` +
            `（門 ${(MATCH_FLOOR * 100).toFixed(0)}%）`,
        );
        assert(
          rate >= MATCH_FLOOR,
          `draft の一致率 ${(rate * 100).toFixed(2)}% が門 ${
            (MATCH_FLOOR * 100).toFixed(0)
          }% に満たない（段別: ${
            counts.map((entry, index) =>
              `${CASES[index]} ${entry.perStep.map((hit) => `${hit}/${entry.cycles}`).join(",")}`
            ).join(" / ")
          }）`,
        );
      });

      await t.step("④ 寿命: 借り手が生きている間は貸し手を畳めない", async () => {
        const context = await target.createGenerationContext({
          bindings: { [CAPACITY_SYMBOL]: CHUNK_LENGTH },
          chunkLength: CHUNK_LENGTH,
        });
        const borrowed = await openDrafterContext(drafter, context);
        // 貸し手 context は借り手が居る間 dispose できない（借り手の bind group が掴んでいる）。
        const refusedContext = await assertRejects(() => context.dispose(), ExecutionError);
        assert(refusedContext.message.includes("借りている"), refusedContext.message);
        // 順序どおりなら通る。
        await borrowed.dispose();
        await context.dispose();
      });
    } finally {
      // MUST: 借り手 Session を先に畳む（貸し手の重みを借りている間の dispose は拒否される）。
      if (drafterSession !== undefined) {
        const refused = await assertRejects(() => target.dispose(), ExecutionError);
        assert(refused.message.includes("貸し出している"), refused.message);
        await drafterSession.dispose();
      }
      await target.dispose();
      gpu.destroy();
      ple.dispose();
    }
  },
});

// ---------------------------------------------------------------------------
// ⑤ pipeline の投機オプション（実 GPU）— drafter Session まで組んで、生成は 1 つも変わらない
// ---------------------------------------------------------------------------

/** chat の検収ケース（`e2e_gemma4_directory_test.ts` と**同じ golden** — 投機の有無で動かない）。 */
const CHAT_CASE = {
  fixture: "single-user",
  maxNewTokens: 24,
  expected: "The capital of France is **Paris**.",
} as const;

type ChatFixture = {
  readonly chat: { readonly name: string; readonly messages: Gemma4ChatMessage[] }[];
};

Deno.test({
  name: "Gemma4Pipeline: speculative を渡すと drafter も載り、chat の golden は動かない（実 GPU）",
  ignore: !MIRROR_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    const fixture: ChatFixture = JSON.parse(
      await Deno.readTextFile(new URL("fixtures/gemma-text/gemma4-chat.json", import.meta.url)),
    );
    const chatCase = fixture.chat.find((row) => row.name === CHAT_CASE.fixture);
    assert(chatCase !== undefined, `フィクスチャに chat ケース '${CHAT_CASE.fixture}' が無い`);

    const started = performance.now();
    const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(MIRROR_DIR), {
      speculative: { k: GEMMA4_DRAFT_STEPS },
      maxResidentPleBytes: allResidentPleBytesOfMirror(MIRROR_DIR),
    });
    try {
      console.log(
        `[e2e] gemma4 drafter pipeline: 投機ありのロード ${
          (performance.now() - started).toFixed(0)
        }ms`,
      );
      await t.step(
        "① 生成は投機の有無で 1 文字も変わらない（段 2 に投機ループは無い）",
        async () => {
          const stream = pipeline.chat(chatCase.messages, {
            maxNewTokens: CHAT_CASE.maxNewTokens,
            sampler: { temperature: 0 },
          });
          assertEquals(await stream.text(), CHAT_CASE.expected, "温度 0 の出力");
        },
      );

      await t.step("② 見積りは段 2 では target ぶんだけ", () => {
        const report = pipeline.estimateSessionMemory();
        assert(report.resident.weights.totalBytes > 0, "常駐重みの見積りが 0 バイト");
      });
    } finally {
      // 順序（drafter Session → target Session）が正しければ、この 1 本は成功で返る
      // （逆順だと貸し出しの門に当たって `disposeSteps` が失敗を運ぶ）。
      await pipeline.dispose();
    }
  },
});
