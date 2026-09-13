// 実重み Gemma 4 E2B の**投機ループ**（MTP drafter を使った speculative decoding）の検収門 —
// ADR [0096](../../../docs/decisions/0096-speculative-decoding.md) 段 3。
//
// 段 2 の門（`e2e_gemma4_drafter_test.ts`）は「drafter が torch と同じ draft を出す」ことまでで、
// 投機ループ（verify・受理・棄却・commit）はこのファイルが受け持つ。見るのは 5 本:
//
// ① **同一性**（`stateAttentionReduce: "sequential"`）… 投機の唯一の契約は「速くなるだけで
//    出力は変わらない」である。3 ケースの golden prompt を 200 token 生成し、投機あり / 無しの
//    token id 列と絶対位置列が**厳密一致**する。併せて torch 継続列との先頭一致に床を置く
//    （「投機 == 非投機」だけでは、両側そろって別の列へ動く退行が緑のまま通る）。受理率
//    （token/cycle）は README 2 ケースに床を置く（床を割る = drafter か受理判定の退行で、
//    「動くが速くならない」形をここで落とす）。
//    NOTE: 席を `"sequential"` に倒すのは、decode（M=1）と verify（M=4）が**同じ縮約カーネル**を
//    通る形にするためである。既定の `"parallel"` は decode だけ別変種（KV 長を 16 レーンで分担）を
//    使うので、同じ行でもビット同一にならない（下の②）。
//    NOTE: 門は `speculative: "always"`（ゲート無しの常時投機）で回す — 床も厳密一致も「常に
//    投機」の契約だからである。既定の自己採算ゲート（段 4-B ④・`speculative: true`）は最後の
//    step が別に見る（列の一致だけが門で、落ちた step 数は壁時計依存なのでログのみ）。
// ② **既定席（`"parallel"`）の実測**（門ではない）… ①と同じ 3 ケースを既定席で回し、投機 /
//    非投機の相違添字を報告する。①′（decode の変種）と①（verify）の縮約順の差が argmax を
//    割る位置を数えるための実測席で、門にはしない（`docs/limitations.md`）。
// ③ **phase の観測**（`onRunDiagnostics`）… 1 sequence の生成で届く run 通知が
//    `speculation.draftRuns` / `cycles` と本数まで一致し、verify 形（M=4 / R=4）の計画が
//    2 cycle 目以降 LRU に**居続ける**（`lastRunPrepared.hit`）。state bind group の焼き直し
//    （`stateBacking.rebindCount`）が cycle 数に比例して伸びないことも同じ窓で見る。
// ④ **見積り**（`estimateSessionMemory`）… 常駐重みが target + drafter の 2 Session の診断と
//    **厳密一致**し、state は貸し手ぶん + 借り手の論理長 uniform 8 バイト、verify シナリオの
//    `ioBytes` が decode より「行が 3 本増えたぶん」ちょうど大きい。drafter を勘定に入れ忘れたら
//    どれも割れる。
// ⑤ **カーネル同値**（u32）… ③の同一性が成り立つ前提そのもの: 同じ token・同じ位置の行を
//    verify 形（M=4 の deferred run の行 0）と decode 形（M=1 の immediate run）で流した logits が
//    **u32 で厳密一致**する。①が割れたときに「受理判定の欠陥」と「カーネルの数値差」を切り分ける
//    のがこの門で、`Session` を直に回す（`commit(0)` は生成面が出さない）。
//
// ## 資産
//
// 配布形ミラー `models/karume-gemma4/`（drafter 込み）と golden
// `outputs/series/gemma4-e2b-drafter/`（prompt と torch 継続列）。どちらもリポジトリ管理外で、
// ミラーが無い環境では**明示 SKIP**、ミラーがあるのに golden が欠けている形は
// `e2e_gemma4_drafter_test.ts` の資産門が FAIL にする。

import { assert, assertEquals } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import {
  acquireGpu,
  type GenerationContext,
  parseSafetensors,
  prepareModel,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type StateAttentionReduce,
  type Tensor,
} from "@karume/runtime";
import { Gemma4Pipeline, type GenerationRunPhase, type GenerationStop } from "../gemma.ts";
import { parseGemma4PipelineConfig } from "../src/gemma/config.ts";
import { createGemma4Ple, parseGemma4PleIndex } from "../src/gemma/ple.ts";
import { gemma4RopeInputs, type Gemma4RopeSpec } from "../src/gemma/rope.ts";
import { GEMMA4_DRAFT_STEPS } from "../src/gemma/speculative.ts";
import type { SamplerSpec } from "../src/generation/sampler.ts";
import { readShard, resolveShards, streamShards } from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { allResidentBytes, allResidentPleBytesOfMirror } from "./helpers/ple-budget.ts";
import { openPleShardAt } from "./helpers/ple-source.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4/", import.meta.url);
const GOLDEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-drafter/", import.meta.url);

/** SKIP 時にそのまま貼れる組み立てコマンド（段 2 の門と同文）。 */
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4" +
  "（drafter 系列と golden は … python -m gemma4.export_drafter）";

/** golden のケース（正本は `export_drafter.py` の GOLDEN_CASES）。 */
const CASES = ["short-en", "readme-recipes", "readme-exporter"] as const;
type CaseName = (typeof CASES)[number];
const GOLDEN_PREFIX = "drafter-golden.";
const SUFFIX = ".safetensors";

/** 1 ケースで生成する token 数（受理率の分母を実用域まで積むための長さ）。 */
const MAX_NEW_TOKENS = 200;

/**
 * 受理率（token/cycle = `(accepted + cycles) / cycles`）の床。
 *
 * MUST: **実測に合わせない**。値は段 3 の着地時点の end-to-end 実測（README の 2 ケースで
 * 2.010 / 2.140 token/cycle・2026-09-08・RTX 3080 Ti）の**下**に置いた回帰検出線で、drafter の質か
 * 受理判定が退行すると割れる（golden 側の teacher forcing の値 2.125 / 2.110 とは別量）。
 * `short-en`（25 token の短文）は 1 ケースぶんの分散が大きすぎるのでログだけ（門は置かない）。
 *
 * NOTE: 基準は **RTX 3080 Ti / Vulkan バックエンド**の実測である。①は `"sequential"` 席でも
 * 縮約の**実行順**までは固定しない（grid-stride の分割が device の限界値で変わる）ので、別 GPU /
 * driver では受理列がずれて床を割りうる — そのときは退行ではなく環境差なので、床を下げる。
 */
const ACCEPTANCE_FLOOR: Partial<Record<CaseName, number>> = {
  "readme-recipes": 1.9,
  "readme-exporter": 1.9,
};

/** ③④で回す短いターン（phase の列と見積りの突合だけなので、長さは要らない）。 */
const PHASE_TOKENS = 40;

/** 実資産の形（config の `hidden_size`・貸し手の埋め込み表は borrower から借りる）。 */
const HIDDEN = 1536;
const VOCAB = 262144;

/**
 * ⑤の prefill 刻み（この門は 25 token の prompt を 1 chunk で入れるだけなので、配布形の 768 では
 * pad 行が 30 倍になる）。
 */
const CHUNK_LENGTH = 32;
/** verify run の物理行数（`k+1` — ⑤ではバケットとして宣言する）。 */
const VERIFY_ROWS = GEMMA4_DRAFT_STEPS + 1;

/** グラフ入力の名前（正本は `export_product.py` の定数）。 */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";
const CAPACITY_SYMBOL = "C";

/**
 * 借り手 context が GPU 上に持つバイト数（論理長 uniform だけ — 借り物スロットは 1 バイトも
 * 確保しない）。正本は runtime の `LENGTHS_BYTES`（公開面に無いのでここへ写す）。
 */
const LENGTHS_BYTES = 8;

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

if (!AVAILABLE) {
  console.warn(
    `[karume] drafter 入りの配布形ミラー（${MIRROR_DIR.pathname}）か golden が無いため投機ループ` +
      ` 検収を SKIP する。組み立て: ${ASSEMBLE_COMMAND}`,
  );
}

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden に '${name}' が無い`);
  assertEquals(view.dtype, "I32", `golden '${name}' の格納 dtype`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** golden 1 ケース（`prompt` と torch greedy の継続列 — 継続列は参考値で門ではない）。 */
type GoldenCase = {
  readonly prompt: number[];
  readonly continuation: number[];
};

const readGoldenCase = async (name: string): Promise<GoldenCase> => {
  const bytes = await Deno.readFile(goldenPath(name));
  const file = parseSafetensors(
    bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
      ? bytes.buffer
      : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    prompt: [...goldenI32(file, "prompt")],
    continuation: [...goldenI32(file, "tokens")],
  };
};

/** ミラーの manifest / `pipelineConfig`（RoPE と容量の出どころ）。 */
const mirrorManifest = (): Manifest =>
  parseManifest(Deno.readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)));

const pipelineConfig = () => {
  const manifest = mirrorManifest();
  return parseGemma4PipelineConfig(manifest.models[manifest.defaultModel].pipelineConfig);
};

const pleIndex = () =>
  parseGemma4PleIndex(JSON.parse(Deno.readTextFileSync(new URL("e2b/ple/ple.json", MIRROR_DIR))));

// ---------------------------------------------------------------------------
// 走行 1 本（sequence を開いて 1 ターン汲み、閉じる）
// ---------------------------------------------------------------------------

/** 1 ターンぶんの観測（token id 列・絶対位置列・停止理由・所要）。 */
type Turn = {
  readonly ids: number[];
  readonly positions: number[];
  readonly stop: GenerationStop;
  readonly ms: number;
};

const runTurn = async (
  pipeline: Gemma4Pipeline,
  prompt: readonly number[],
  options: {
    /**
     * `"always"` = ゲート無しの常時投機（門はこちら — 床も厳密一致も「常に投機」の契約である）。
     * `true` は自己採算ゲート付きの既定席で、①の最後の step だけがこちらを回す。
     */
    readonly speculative: boolean | "always";
    readonly capacity?: number;
    readonly tokens: number;
    readonly sampler?: SamplerSpec;
  },
): Promise<Turn> => {
  const sequence = await pipeline.sequence({
    speculative: options.speculative,
    ...(options.capacity === undefined ? {} : { capacity: options.capacity }),
  });
  try {
    const started = performance.now();
    // 3 ケースの突合は sampler を渡さない（低層の既定 = 温度 0）。投機は温度に依らず張るので
    // 抽選が走っても投機 / 非投機の列は一致するが、①は torch 継続列との突合も兼ねている — RNG は
    // splitmix64 で torch の Philox とは別物（同じ seed でも同じ列にならない）ので、parity は
    // 温度 0 でしか採れない。温度 > 0 の同一性は推奨 sampler の 1 本で別に見る。
    const stream = sequence.generate({
      prompt: [...prompt],
      maxNewTokens: options.tokens,
      ...(options.sampler === undefined ? {} : { sampler: options.sampler }),
    });
    const ids: number[] = [];
    const positions: number[] = [];
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      ids.push(event.id);
      positions.push(event.position);
    }
    return { ids, positions, stop: await stream.done, ms: performance.now() - started };
  } finally {
    await sequence.dispose();
  }
};

/** この会話が要る容量（`chunkLength` 未満は指定できないので下限で持ち上げる）。 */
const capacityFor = (pipeline: Gemma4Pipeline, promptLength: number): number =>
  Math.max(pipeline.program.chunkLength, promptLength + MAX_NEW_TOKENS);

/** 2 列の先頭一致長。 */
const commonPrefix = (left: readonly number[], right: readonly number[]): number => {
  let index = 0;
  while (index < left.length && index < right.length && left[index] === right[index]) index += 1;
  return index;
};

/**
 * torch 継続列と一致していることを要求する先頭の長さ（①の**外部**参照点）。
 *
 * ①の主門「投機 == 非投機」は、投機と非投機が**そろって**別の列へ動く退行（target 側の退行・
 * PLE / RoPE の取り違え）を拾えない — 両方が同じだけ壊れれば一致したままである。torch の greedy
 * 継続列と先頭が合っていることを併せて見ると、その形が落ちる。
 *
 * MUST: 実測（2026-09-08 の 3 ケースは 200/200 一致）に合わせない。i4 の target は torch と
 * ビット同値ではないので、いずれ長い列の途中で割れうる — 床は「別の列へ乗り換えた」級だけを
 * 拾う位置に置く。
 */
const GOLDEN_PREFIX_FLOOR = 16;

/** 投機の勘定を「1 cycle が確定させた token 数」に直す（= `(accepted + cycles) / cycles`）。 */
const tokensPerCycle = (stop: GenerationStop): number => {
  const speculation = stop.speculation;
  assert(speculation !== undefined, `投機が張られていないターン: ${JSON.stringify(stop)}`);
  return (speculation.accepted + speculation.cycles) / speculation.cycles;
};

// ---------------------------------------------------------------------------
// ① 同一性（sequential）— 投機あり / 無しの列が厳密一致する
// ---------------------------------------------------------------------------

Deno.test({
  name: "gemma4 投機①: sequential 席で投機 / 非投機の token 列が厳密一致する（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const started = performance.now();
    const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(MIRROR_DIR), {
      speculative: { k: GEMMA4_DRAFT_STEPS },
      // 参照経路（runtime の既定）— decode も verify も同じ縮約カーネルを通る席。
      stateAttentionReduce: "sequential",
      maxResidentPleBytes: allResidentPleBytesOfMirror(MIRROR_DIR),
    });
    console.log(
      `[e2e] gemma4 投機①: pipeline ロード ${(performance.now() - started).toFixed(0)}ms`,
    );
    try {
      for (const name of CASES) {
        await t.step(`${name}: 投機 / 非投機の ${MAX_NEW_TOKENS} token`, async () => {
          const golden = await readGoldenCase(name);
          const capacity = capacityFor(pipeline, golden.prompt.length);
          const speculative = await runTurn(pipeline, golden.prompt, {
            speculative: "always",
            capacity,
            tokens: MAX_NEW_TOKENS,
          });
          const plain = await runTurn(pipeline, golden.prompt, {
            speculative: false,
            capacity,
            tokens: MAX_NEW_TOKENS,
          });

          // 投機の唯一の契約（速くなるだけで列は変わらない）。
          assertEquals(speculative.ids, plain.ids, `${name}: 投機 / 非投機の token id 列`);
          assertEquals(speculative.positions, plain.positions, `${name}: 絶対位置列`);
          assertEquals(
            speculative.stop.reason,
            plain.stop.reason,
            `${name}: 停止理由`,
          );
          assertEquals(speculative.stop.tokens, plain.stop.tokens, `${name}: 生成 token 数`);
          // 非投機の sequence は勘定の欄ごと持たない（欄が生えていたら投機経路を通っている）。
          assertEquals(plain.stop.speculation, undefined, `${name}: 非投機ターンの speculation 欄`);

          const perCycle = tokensPerCycle(speculative.stop);
          const speculation = speculative.stop.speculation;
          assert(speculation !== undefined, "投機ターンに勘定が載っていない");
          const goldenPrefix = commonPrefix(speculative.ids, golden.continuation);
          console.log(
            `[e2e] gemma4 投機① ${name}: T=${golden.prompt.length} / ${speculative.ids.length} ` +
              `token / ${speculation.cycles} cycle / 受理 ${speculation.accepted}/` +
              `${speculation.drafted} / ${perCycle.toFixed(3)} token/cycle / 分布 [${
                speculation.acceptedHistogram.join(",")
              }] / 投機 ${speculative.ms.toFixed(0)}ms vs 非投機 ${plain.ms.toFixed(0)}ms / ` +
              `torch 継続列との先頭一致 ${goldenPrefix}`,
          );

          // 外部参照点（①が「投機 == 非投機」だけでは拾えない、両側そろっての退行を落とす）。
          assert(
            goldenPrefix >= GOLDEN_PREFIX_FLOOR,
            `${name}: torch 継続列との先頭一致が ${goldenPrefix} token しかない（床 ` +
              `${GOLDEN_PREFIX_FLOOR}）— 投機と非投機がそろって別の列へ動いている`,
          );

          const floor = ACCEPTANCE_FLOOR[name];
          if (floor !== undefined) {
            assert(
              perCycle >= floor,
              `${name}: 受理率 ${perCycle.toFixed(3)} token/cycle が床 ${floor.toFixed(2)} に` +
                `満たない（受理 ${speculation.accepted} / ${speculation.cycles} cycle・分布 [${
                  speculation.acceptedHistogram.join(",")
                }]）`,
            );
          }
        });
      }

      await t.step(
        "readme-recipes（配布形の推奨 sampler・温度 1）: 投機 / 非投機の列が同一",
        async () => {
          // 投機は温度に依らず張る — 受理の抽選は行ごとに非投機と同じ logits・history・順序で
          // `sampler.next` を呼ぶので、RNG の消費列まで一致する。torch との parity は採れない（RNG が
          // 別物）が、投機 / 非投機の同一性はここで見る。sampler は配布形の宣言（推奨値）そのもの。
          const sampler: SamplerSpec = { ...pipelineConfig().sampler, seed: 1 };
          assert(
            (sampler.temperature ?? 0) > 0,
            "配布形の推奨 sampler が温度 0（この門の前提が崩れている）",
          );
          const golden = await readGoldenCase("readme-recipes");
          const capacity = capacityFor(pipeline, golden.prompt.length);
          const speculative = await runTurn(pipeline, golden.prompt, {
            speculative: "always",
            capacity,
            tokens: MAX_NEW_TOKENS,
            sampler,
          });
          const plain = await runTurn(pipeline, golden.prompt, {
            speculative: false,
            capacity,
            tokens: MAX_NEW_TOKENS,
            sampler,
          });
          assertEquals(speculative.ids, plain.ids, "温度 1: 投機 / 非投機の token id 列");
          assertEquals(speculative.positions, plain.positions, "温度 1: 絶対位置列");
          assertEquals(speculative.stop.tokens, plain.stop.tokens, "温度 1: 生成 token 数");
          // 抽選が実際に走っていること（温度 0 の列と自分の長さの範囲で一致していれば RNG が
          // 消費されていない）。NOTE: 生テキストの継続は温度 1 だと早々に停止 token を引く
          // （2026-09-08 の実測は 3 token）ので受理率はここでは見ない — 推奨 sampler での受理率は
          // 段 4 の実測項目。
          const greedy = await runTurn(pipeline, golden.prompt, {
            speculative: false,
            capacity,
            tokens: MAX_NEW_TOKENS,
          });
          assert(
            commonPrefix(plain.ids, greedy.ids) < plain.ids.length,
            "温度 1 の列が温度 0 の列と一致（抽選が効いていない）",
          );
          const speculation = speculative.stop.speculation;
          assert(speculation !== undefined && speculation.cycles >= 1, "投機が張られていない");
          console.log(
            `[e2e] gemma4 投機① readme-recipes（温度 ${sampler.temperature}・topK ${sampler.topK}・` +
              `topP ${sampler.topP}）: ${plain.ids.length} token / ${speculation.cycles} cycle / ${
                tokensPerCycle(speculative.stop).toFixed(3)
              } token/cycle / 温度 0 との先頭一致 ${commonPrefix(plain.ids, greedy.ids)}`,
          );
        },
      );

      await t.step(
        "自己採算ゲート付き（既定の speculative: true）: always と同じ列を出す",
        async () => {
          // ゲートは cycle ごとに「投機 / decode 形」を壁時計で選ぶ。sequential 席では両者が
          // ビット同一（⑤の門）なので、どこで切り替わっても列は変わらない — ここが割れるなら
          // 切替そのもの（hidden の継ぎ・frontier の commit）が壊れている。
          const golden = await readGoldenCase("readme-recipes");
          const capacity = capacityFor(pipeline, golden.prompt.length);
          const gated = await runTurn(pipeline, golden.prompt, {
            speculative: true,
            capacity,
            tokens: MAX_NEW_TOKENS,
          });
          const always = await runTurn(pipeline, golden.prompt, {
            speculative: "always",
            capacity,
            tokens: MAX_NEW_TOKENS,
          });
          assertEquals(gated.ids, always.ids, "ゲート付き / always の token id 列");
          assertEquals(gated.positions, always.positions, "絶対位置列");
          assertEquals(gated.stop.tokens, always.stop.tokens, "生成 token 数");
          const speculation = gated.stop.speculation;
          assert(speculation !== undefined, "ゲート付きのターンに勘定が載っていない");
          // MUST: 落ちた step 数は**門にしない**（壁時計依存で、host と負荷で変わる）。この機で
          // どう出たかを記録するだけである。
          console.log(
            `[e2e] gemma4 投機① ゲート: ${gated.ids.length} token / ${speculation.cycles} cycle / ` +
              `plain step ${speculation.plainSteps} / 切替 ${speculation.switches} / ` +
              `ゲート ${gated.ms.toFixed(0)}ms vs always ${always.ms.toFixed(0)}ms`,
          );
        },
      );
    } finally {
      await pipeline.dispose();
    }
  },
});

// ---------------------------------------------------------------------------
// ②③④ 既定席（parallel）— 相違の実測・phase の観測・見積りの厳密門
// ---------------------------------------------------------------------------

/** run 1 本ぶんの観測（phase + その run の同期区間で読んだ診断）。 */
type PhaseRecord = {
  readonly phase: GenerationRunPhase;
  readonly preparedHit: boolean | undefined;
  readonly rebindCount: number;
  readonly residentCompressedBytes: number;
  readonly stateResidentBytes: number;
};

const recordOf = (diagnostics: SessionDiagnostics, phase: GenerationRunPhase): PhaseRecord => ({
  phase,
  preparedHit: diagnostics.lastRunPrepared?.hit,
  rebindCount: diagnostics.stateBacking.rebindCount,
  residentCompressedBytes: diagnostics.storage.residentCompressedBytes,
  stateResidentBytes: diagnostics.stateBacking.residentBytes,
});

/** 記録の中から 1 種類の phase を抜く。 */
const ofKind = (records: readonly PhaseRecord[], kind: GenerationRunPhase["kind"]) =>
  records.filter((record) => record.phase.kind === kind);

Deno.test({
  name: "gemma4 投機②③④: 既定席の相違・phase の本数・見積りの合算（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    /** 観測を集める窓（step が開けている間だけ積む）。 */
    let records: PhaseRecord[] | undefined;
    const started = performance.now();
    const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(MIRROR_DIR), {
      speculative: { k: GEMMA4_DRAFT_STEPS },
      maxResidentPleBytes: allResidentPleBytesOfMirror(MIRROR_DIR),
      onRunDiagnostics: (diagnostics, phase) => {
        records?.push(recordOf(diagnostics, phase));
      },
    });
    console.log(
      `[e2e] gemma4 投機②: pipeline ロード ${(performance.now() - started).toFixed(0)}ms`,
    );
    const k = GEMMA4_DRAFT_STEPS;
    try {
      await t.step(
        "② 既定席（parallel）の投機 / 非投機の相違を実測する（門ではない）",
        async () => {
          for (const name of CASES) {
            const golden = await readGoldenCase(name);
            const capacity = capacityFor(pipeline, golden.prompt.length);
            const speculative = await runTurn(pipeline, golden.prompt, {
              speculative: "always",
              capacity,
              tokens: MAX_NEW_TOKENS,
            });
            const plain = await runTurn(pipeline, golden.prompt, {
              speculative: false,
              capacity,
              tokens: MAX_NEW_TOKENS,
            });
            // 門にするのは「両方が要求ぶん出し切ること」だけ（列の一致は①の席でだけ契約する）。
            assertEquals(speculative.ids.length, MAX_NEW_TOKENS, `${name}: 投機側の token 数`);
            assertEquals(plain.ids.length, MAX_NEW_TOKENS, `${name}: 非投機側の token 数`);
            const first = commonPrefix(speculative.ids, plain.ids);
            let diverged = 0;
            for (let index = 0; index < MAX_NEW_TOKENS; index += 1) {
              if (speculative.ids[index] !== plain.ids[index]) diverged += 1;
            }
            console.log(
              `[e2e] gemma4 投機② ${name}: 相違 ${diverged}/${MAX_NEW_TOKENS} 個・最初の不一致 ` +
                `${first === MAX_NEW_TOKENS ? "無し" : `@${first}`} / ${
                  tokensPerCycle(speculative.stop).toFixed(3)
                } token/cycle / 投機 ${speculative.ms.toFixed(0)}ms vs 非投機 ` +
                `${plain.ms.toFixed(0)}ms`,
            );
          }
        },
      );

      // ③④は同じ 1 ターンの観測を共有する（見積りは「その走行が実際に抱えた常駐」と突き合わせる
      // ので、容量は既定のまま = `estimateSessionMemory` が既定で見積る形と同じにする）。
      const golden = await readGoldenCase("short-en");
      records = [];
      const turn = await runTurn(pipeline, golden.prompt, {
        speculative: "always",
        tokens: PHASE_TOKENS,
      });
      const observed = records;
      records = undefined;

      await t.step("③ run 通知の本数と種別が投機の勘定と一致する", () => {
        const speculation = turn.stop.speculation;
        assert(speculation !== undefined, "投機が張られていない");
        const drafts = ofKind(observed, "draft");
        const verifies = ofKind(observed, "verify");
        const prefills = ofKind(observed, "prefill");
        // 非投機の decode run は 1 本も混ざらない（投機ターンの decode は `k'=0` の verify として
        // 出る — 種別が混ざると「投機なのに decode へ落ちた cycle」が沈黙する）。
        //
        // MUST: この 0 本が成り立つのは上の `speculative: "always"`（ゲート無し）のターンだけ
        // である。既定の `auto` では自己採算ゲートが落とす plain step が `decode` 通知として
        // **正当に**出る（本数 = `speculation.plainSteps`・run の形が M=1・R=1 で decode そのもの
        // だから同じ枝を名乗る）ので、この席と下の総数の式（`plainSteps` の項が無い）はどちらも
        // ゲート付きのターンには当てはまらない。
        assertEquals(ofKind(observed, "decode").length, 0, "投機ターンに decode 通知が混ざった");
        assertEquals(drafts.length, speculation.draftRuns, "draft 通知の本数 = draftRuns");
        assertEquals(verifies.length, speculation.cycles, "verify 通知の本数 = cycles");
        assertEquals(
          observed.length,
          prefills.length + speculation.cycles + speculation.draftRuns,
          "通知の総数（prefill chunk + cycle + draft run）",
        );
        // cycle は 1 始まりの連番（同じ cycle の draft と verify が同じ番号を名乗る）。
        assertEquals(
          verifies.map((record) => record.phase.kind === "verify" ? record.phase.cycle : -1),
          Array.from({ length: speculation.cycles }, (_unused, index) => index + 1),
          "verify の cycle 番号",
        );
        // 受理数の合計が勘定と一致する（phase が運ぶ `accepted` と `GenerationStop` の合算が
        // 別々に数えられていないこと）。
        const accepted = verifies.reduce(
          (sum, record) => sum + (record.phase.kind === "verify" ? record.phase.accepted : 0),
          0,
        );
        assertEquals(accepted, speculation.accepted, "phase の accepted 合計");

        // verify 形（M=k+1 / R=k+1）の計画は 2 本目以降 LRU に居続ける（毎 cycle 導出へ落ちると
        // 例外は出ないまま decode のホットパスに計画導出が戻る）。
        const missed = verifies.slice(1).filter((record) => record.preparedHit !== true).length;
        assertEquals(
          missed,
          0,
          `2 本目以降の verify ${verifies.length - 1} 本のうち ${missed} 本が PreparedPlan を` +
            `外した（hit: [${verifies.map((record) => String(record.preparedHit)).join(",")}]）`,
        );
        // state bind group の焼き直しは **cycle 数に比例しない**。焼き直しが起きるのは
        // (context, backing) の組が変わったときだけで、同じ物理行数の verify が続く区間では
        // 1 度も増えない。
        const rowsSeries = verifies.map((record) =>
          record.phase.kind === "verify" ? record.phase.rows : -1
        );
        // MUST: 縛るのは「**M=k+1 の verify が続く区間**で焼き直さない」である。予算末尾の
        // `k' = 0` の cycle だけは decode 形（M=1）の run になり、その物理行数がこの context で
        // 初出なら bind group が 1 度だけ焼き直される — 末尾が `k'=0` に着地するかは受理列
        // （GPU / driver 依存）で決まるので、混ぜると環境差で割れる門になる。
        const bucketed = verifies.filter((record) =>
          record.phase.kind === "verify" && record.phase.rows > 1
        );
        const rebinds = bucketed.map((record) => record.rebindCount);
        // 実測（2026-09-08）では prefill 形 → verify 形の切替ぶん 1 回だけが 2 本目の verify までに
        // 現れ、そこから 1 度も動かない（`k'` が 2 / 1 に縮む cycle も、有効行が減るだけで物理
        // 行数はバケット `k+1` のままなので焼き直しは起きない）。
        assert(
          bucketed.length >= 4,
          `M=k+1 の verify が ${bucketed.length} 本では比例を見られない`,
        );
        const settled = new Set(rebinds.slice(1));
        assertEquals(
          settled.size,
          1,
          `2 本目以降の verify で rebindCount が動いた（cycle 数に比例して焼き直している）: [${
            rebinds.join(",")
          }]`,
        );
        assert(
          rebinds[1] - rebinds[0] <= 1,
          `verify 形へ落ち着くまでの焼き直しが ${rebinds[1] - rebinds[0]} 回ある: [${
            rebinds.join(",")
          }]`,
        );
        console.log(
          `[e2e] gemma4 投機③: prefill ${prefills.length} + draft ${drafts.length} + verify ` +
            `${verifies.length} 通（うち M=k+1 が ${bucketed.length} 本）/ 配送 ` +
            `${turn.stop.tokens} token / verify の行数 [${rowsSeries.join(",")}] / rebind [${
              rebinds.join(",")
            }]`,
        );
      });

      await t.step("④ 見積りが drafter の常駐と借り手 state と verify 形を合算する", () => {
        // 診断は phase で Session が分かれる（draft = 借り手・それ以外 = 貸し手）。
        const lender = ofKind(observed, "verify")[0];
        const borrower = ofKind(observed, "draft")[0];
        assert(
          lender !== undefined && borrower !== undefined,
          "貸し手 / 借り手の診断が採れていない",
        );
        // MUST: 観測値は見積りを取る**前**に出す（見積りが落ちる形でも突合の材料が残るように）。
        console.log(
          `[e2e] gemma4 投機④ 観測: 常駐重み target ${lender.residentCompressedBytes} B + ` +
            `drafter ${borrower.residentCompressedBytes} B / state 貸し手 ` +
            `${lender.stateResidentBytes} B + 借り手 ${borrower.stateResidentBytes} B`,
        );
        const report = pipeline.estimateSessionMemory();

        // 1. 常駐重み: 2 Session が実際に GPU 上へ置いたバイト数の和と厳密一致。
        assertEquals(
          report.resident.weights.compressedBytes,
          lender.residentCompressedBytes + borrower.residentCompressedBytes,
          `圧縮常駐の見積り（target ${lender.residentCompressedBytes} + drafter ` +
            `${borrower.residentCompressedBytes}）`,
        );
        // 2. 借り手 context は論理長 uniform だけ（借り物スロットは 1 バイトも確保しない）。
        assertEquals(borrower.stateResidentBytes, LENGTHS_BYTES, "借り手 context の常駐バイト数");
        // 3. state は貸し手ぶん + 借り手の 8 バイト。
        assertEquals(
          report.resident.stateBytes,
          lender.stateResidentBytes + LENGTHS_BYTES,
          `state の見積り（貸し手 ${lender.stateResidentBytes} + 借り手 ${LENGTHS_BYTES}）`,
        );

        // 4. verify シナリオの入出力は decode より「行が k 本増えたぶん」ちょうど大きい。
        //    行 1 本ぶんの**入力**バイト数は同じ報告から導く（prefill と decode は R=1 で同形・
        //    違いは chunk 行数だけなので、その差を行数で割ると 1 行ぶんになる）。式を写さずに
        //    報告の内側で閉じるので、`chunkLength` や PLE の幅が変わっても追随が要らない。
        const named = (name: string) => {
          const scenario = report.scenarios.find((entry) => entry.name === name);
          assert(scenario !== undefined, `シナリオ '${name}' が報告に無い`);
          return scenario;
        };
        const prefill = named("prefill");
        const decode = named("decode");
        const verify = named("verify");
        const chunkLength = pipeline.program.chunkLength;
        const perRowInputBytes = (prefill.ioBytes - decode.ioBytes) / (chunkLength - 1);
        assert(
          Number.isSafeInteger(perRowInputBytes),
          `1 行ぶんの入力バイト数が整数にならない（prefill ${prefill.ioBytes} − decode ` +
            `${decode.ioBytes} を ${chunkLength - 1} 行で割った ${perRowInputBytes}）`,
        );
        const rows = k + 1;
        assertEquals(
          verify.ioBytes - decode.ioBytes,
          // 入力: 行が k 本増える（token / PLE / RoPE）+ 行選択 `last_row` が k 要素増える。
          k * perRowInputBytes + k * 4 +
            // 出力: 選んだ行の logits と hidden を k 行ぶん余計に readback する。
            k * (VOCAB + HIDDEN) * 4,
          `verify（${rows} 行）と decode（1 行）の ioBytes 差`,
        );
        console.log(
          `[e2e] gemma4 投機④: 常駐重み ${report.resident.weights.compressedBytes} B（target ` +
            `${lender.residentCompressedBytes} + drafter ${borrower.residentCompressedBytes}）/ ` +
            `state ${report.resident.stateBytes} B / io prefill ${prefill.ioBytes}・decode ` +
            `${decode.ioBytes}・verify ${verify.ioBytes} B / 1 行 ${perRowInputBytes} B`,
        );
      });
    } finally {
      await pipeline.dispose();
    }
  },
});

// ---------------------------------------------------------------------------
// ⑤ カーネル同値（u32）— verify の行 0 と decode の 1 行
// ---------------------------------------------------------------------------

/** f32 の出力テンソルを取り出す（dtype の取り違えは黙って別の列になる）。 */
const f32Of = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  assert(tensor.data instanceof Float32Array, `${where}: f32 でない出力を f32 として読んでいる`);
  return tensor.data;
};

/** 同じ行を verify 形（M=k+1 の deferred run の行 0）と decode 形（M=1）で流した logits の対。 */
type RowPair = {
  readonly deferred: Float32Array<ArrayBuffer>;
  readonly immediate: Float32Array<ArrayBuffer>;
};

Deno.test({
  name: "gemma4 投機⑤: verify の行 0 と decode 1 行の logits が u32 一致する（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const config = pipelineConfig();
    const rope: Gemma4RopeSpec = config.rope;
    const golden = await readGoldenCase("short-en");
    const prompt = golden.prompt;
    const index = pleIndex();
    const ple = createGemma4Ple({
      index,
      openShard: (file) => openPleShardAt(new URL("e2b/ple/", MIRROR_DIR), file),
      vocabSize: VOCAB,
      maxResidentBytes: allResidentBytes(index),
    });
    const shards = resolveShards(new URL("e2b/model/model.i4.safetensors", MIRROR_DIR));
    const gpu = await acquireGpu();

    /**
     * 1 run（`last_row` の要素数が R を束縛する）。返すのは選んだ行の logits。
     *
     * 物理行数 `rows` は固定 chunk 契約（ADR 0066 決定 4）が許す 3 種のどれか — `chunkLength` /
     * 宣言したバケット / decode 形の 1 — で、有効行はその先頭 `ids.length` 本である（残りは
     * pad = token 0・位置 0 で、`queryLength` の外なので KV には入らない）。
     */
    const run = async (
      session: Session,
      logitsName: string,
      context: GenerationContext,
      spec: {
        readonly rows: number;
        readonly ids: readonly number[];
        readonly position: number;
        readonly lastRow: readonly number[];
        readonly commit: "immediate" | "deferred";
      },
      where: string,
    ): Promise<Float32Array<ArrayBuffer>> => {
      const tokens = new Int32Array(new ArrayBuffer(spec.rows * 4));
      const positions = new Int32Array(new ArrayBuffer(spec.rows * 4));
      for (let row = 0; row < spec.ids.length; row += 1) {
        tokens[row] = spec.ids[row];
        positions[row] = spec.position + row;
      }
      const outputs = await session.run(
        {
          [INPUT_IDS]: { dtype: "i32", shape: [1, spec.rows], data: tokens },
          [PER_LAYER_INPUTS]: await ple.gather([...tokens]),
          ...gemma4RopeInputs(rope, positions),
          [LAST_ROW]: {
            dtype: "i32",
            shape: [spec.lastRow.length],
            data: Int32Array.from(spec.lastRow),
          },
        },
        undefined,
        { context, queryLength: spec.ids.length, commit: spec.commit },
      );
      assertEquals(
        outputs[logitsName].shape,
        [1, spec.lastRow.length, VOCAB],
        `${where}: logits の shape`,
      );
      return f32Of(outputs[logitsName], where);
    };

    /**
     * 1 席ぶんの測定（prefill → 4 行の deferred run → `commit(0)` → 同じ行の 1 行 run）。
     *
     * 棄却（`commit(0)`）を挟むのは、投機ループが実際に通す順序だからである（受理 0 の cycle で
     * 次の decode が同じ値を出すことがこの門の意味）。
     */
    const measure = async (
      reduce: StateAttentionReduce,
      linearGemvReduce: "sequential" | "parallel" = "sequential",
      fuseRmsNormAdd = false,
    ): Promise<RowPair> => {
      const parsed = prepareModel(await readShard(shards[0]));
      // 出口 2 本の順序が契約（出力 0 = logits・出力 1 = 最終 norm 後 hidden）。
      const logitsName = parsed.graph.outputs[0];
      const session = await parsed.createSession(gpu, streamShards(shards.slice(1)), {
        stateAttentionReduce: reduce,
        linearGemvReduce,
        fuseRmsNormAdd,
      });
      try {
        const context = await session.createGenerationContext({
          bindings: { [CAPACITY_SYMBOL]: config.capacity },
          chunkLength: CHUNK_LENGTH,
          // verify 形（k+1 行）を pad 無しで流すための実行形。生成面（`sequence.ts`）が
          // 投機で選ぶ物理行数と同じ値である。
          chunkBuckets: [VERIFY_ROWS],
        });
        try {
          await run(session, logitsName, context, {
            rows: CHUNK_LENGTH,
            ids: prompt,
            position: 0,
            lastRow: [prompt.length - 1],
            commit: "immediate",
          }, "prefill");
          assertEquals(context.pastLength, prompt.length, "prefill 後の論理長");
          // frontier `b` と、その後ろに置く draft 3 本（値は任意 — 見るのは行 0 だけで、
          // 後続行は「同じ run に居る」ことだけが効く）。golden の継続列から採る。
          const frontier = golden.continuation[0];
          const drafts = golden.continuation.slice(1, 1 + GEMMA4_DRAFT_STEPS);
          const deferred = await run(session, logitsName, context, {
            rows: VERIFY_ROWS,
            ids: [frontier, ...drafts],
            position: prompt.length,
            lastRow: Array.from({ length: VERIFY_ROWS }, (_unused, row) => row),
            commit: "deferred",
          }, "verify");
          // 1 行も受理しない（= 全棄却の cycle）。論理長は prefill 直後のまま。
          context.commit(0);
          assertEquals(context.pastLength, prompt.length, "commit(0) 後の論理長");
          const immediate = await run(session, logitsName, context, {
            rows: 1,
            ids: [frontier],
            position: prompt.length,
            lastRow: [0],
            commit: "immediate",
          }, "decode");
          assertEquals(session.diagnostics().lastRunFusions?.rmsNormAdd, fuseRmsNormAdd ? 106 : 0);
          return {
            // 行 0 だけを写す（`deferred` の実体は 4 行ぶん）。
            deferred: deferred.slice(0, VOCAB),
            immediate: immediate.slice(0, VOCAB),
          };
        } finally {
          await context.dispose();
        }
      } finally {
        await session.dispose();
      }
    };

    /** 2 列を u32 で突き合わせる（不一致の要約 — 失敗時に 26 万要素の diff を出さないため）。 */
    const compare = (pair: RowPair) => {
      const left = new Uint32Array(pair.deferred.buffer);
      const right = new Uint32Array(pair.immediate.buffer);
      let mismatches = 0;
      let firstIndex = -1;
      let maxAbsDiff = 0;
      for (let index = 0; index < VOCAB; index += 1) {
        if (left[index] !== right[index]) {
          if (mismatches === 0) firstIndex = index;
          mismatches += 1;
        }
        const diff = Math.abs(pair.deferred[index] - pair.immediate[index]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
      }
      return { mismatches, firstIndex, maxAbsDiff, left, right };
    };

    try {
      await t.step("① sequential 席では u32 で厳密一致する", async () => {
        const started = performance.now();
        const pair = await measure("sequential");
        const { mismatches, firstIndex, maxAbsDiff, left, right } = compare(pair);
        assertEquals(
          mismatches,
          0,
          `verify の行 0 と decode 1 行が ${mismatches}/${VOCAB} 語で違う（最初 @${firstIndex}: ` +
            `${pair.deferred[firstIndex]} vs ${pair.immediate[firstIndex]}・最大絶対差 ` +
            `${maxAbsDiff}）`,
        );
        // 要約が恒真化していないこと（同じ突合を型どおりの厳密比較でもう一度通す）。
        assertEquals(left, right, "u32 の厳密比較");
        console.log(
          `[e2e] gemma4 投機⑤ sequential: u32 一致（${VOCAB} 語）/ ` +
            `${(performance.now() - started).toFixed(0)}ms`,
        );
      });

      for (const attention of ["sequential", "parallel"] as const) {
        await t.step(
          `GEMV並列加算 / attention=${attention} でverify行0とdecodeがu32一致する`,
          async () => {
            const { mismatches, left, right } = compare(await measure(attention, "parallel"));
            assertEquals(mismatches, 0);
            assertEquals(left, right);
          },
        );
      }

      for (const gemv of ["sequential", "parallel"] as const) {
        await t.step(`RMS融合 / GEMV=${gemv}でverify行0とdecodeがu32一致する`, async () => {
          const { mismatches, left, right } = compare(await measure("sequential", gemv, true));
          assertEquals(mismatches, 0);
          assertEquals(left, right);
        });
      }

      await t.step("② 既定席（parallel）の差は実測だけ（門ではない）", async () => {
        const started = performance.now();
        const { mismatches, firstIndex, maxAbsDiff } = compare(await measure("parallel"));
        console.log(
          `[e2e] gemma4 投機⑤ parallel: u32 相違 ${mismatches}/${VOCAB} 語（最初 @${firstIndex}）/ ` +
            `最大絶対差 ${maxAbsDiff} / ${(performance.now() - started).toFixed(0)}ms`,
        );
      });
    } finally {
      gpu.destroy();
      ple.dispose();
    }
  },
});
