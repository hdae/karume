// 実重み MiniCPM5-1B（states 形 decode グラフ）の実 GPU 検収門 — **固定 token id 列の parity**。
//
// 波 A の `packages/runtime/tests/e2e_minicpm5_test.ts` が 1-shot 形（prefill 相当の 1 本道）で
// GQA の数値を受け持つのに対し、こちらは**生成ループ全体**（ADR 0066 決定 4 の prefill-chunk と
// decode の 2 形 + GenerationContext の KV スロット）が実重みで**torch と同じ token を同じ順で
// 吐く**ことを受け持つ。検収の形は tolerance ではなく**離散の一致**で、`.lab` / PNG sha256 門と
// 同じ哲学（出口の成果物そのものを固定する — 数値がどれだけ動いても列が割れなければ緑、
// 割れたら赤）。
//
// 門は 5 本:
//
// ① 形の前提（入力 2 本 / 出力 2 本 / 出力 1 の供給元が argmax / states 48 スロット / 記号 M・C）
// ② **greedy parity**（3 ケース × K=16 step が torch の期待列と厳密一致） — この門の主張
// ③ prefill logits の tolerance 突合（`generateGreedy` を通さない手組みの 3 chunk・診断線）
// ④ census（states 形カーネル族が実際に走り、q 側だけ `:gqa` が付く）
// ⑤ decode 経路の計画性質（PreparedPlan が prefill 形 / decode 形の 2 本で安定する）
//
// ホスト側の生成ループ（`src/generation/greedy.ts`）は models の実装なので、この門は models 側に
// ある（`e2e_vowel_detector_chain_test.ts` と同じ分担 — runtime のテストが models の実装を
// 相対 import する逆向きの依存を作らない）。逆に runtime 側は合成グラフで states 形カーネルの
// 受入条件（`tests/gpu_state_acceptance_test.ts`）を持っており、そちらは実重みを要らない。
//
// ## 資産
//
// `outputs/series/minicpm5-1b-decode/`（f32 重み 4.03GiB につきリポジトリ管理外 — `.gitignore` の
// `outputs/`）。生成台本は `tools/export-recipes/minicpm5/export_decode.py` で、コマンドは下の
// {@link GENERATE_COMMAND} がそのまま正本。
//
// - `io.<case>.safetensors` — **無 pad 全長 1 回**の入出力（4 ケース）。chunk 形のグラフを
//   全長 1 発で回した参照ではなく「同じ重み・同じ位置で torch が出す値」の表で、こちらは
//   chunk へ割って**有効行だけ**を突き合わせる（pad 行で 0 に固定されるのは states 形
//   attention の出力だけ — ADR 0066 追記 8。後段の MLP / lm_head は pad 行にも意味のない値を
//   書くので、pad 行のグラフ出力は**読んではならない** — 参照側に対応物も無い）。
// - `greedy.<case>.safetensors` — decode 検収の正本（3 ケース）。`prompt` i32 `[T]` /
//   `expected` i32 `[K]` / `margin` f32 `[K]`。期待列は**全長 full re-forward** で採ってあり
//   （KV cache 経路を通さない — `export_decode.greedy_continuation` の MUST）、検収したい機構と
//   期待値を作る機構が独立している。
//
// 資産が無い環境では**明示 SKIP** する。ADR 0005 の「全 SKIP は明示 FAIL」門番は *GPU アダプタの
// 有無* だけを見ており、この SKIP とは独立。資産が**中途半端に**（ケース欠け）存在する場合は
// SKIP ではなく FAIL にする（下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  parseSafetensors,
  type PreparedModel,
  prepareModel,
  type SafetensorsFile,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import { generateGreedy } from "../src/generation/greedy.ts";
import {
  modelPresent,
  readShard,
  resolveShards,
  streamShards,
} from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

const SERIES_ROOT = new URL("../../../outputs/series/minicpm5-1b-decode/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（tools/export-recipes/minicpm5/README.md と同じ綴り）。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する** — 列挙だけに頼ると生成を
 * 一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `export_decode.py` の `GOLDEN_CASES`（io・T = 6 / 12 / 87 / 61）と `GREEDY_CASES`
 * （greedy — `capital-ja` は継続の余裕が足りず除外されている）。
 */
const EXPECTED_IO_CASES = ["capital-en", "capital-ja", "context-en", "context-ja"] as const;
const EXPECTED_GREEDY_CASES = ["capital-en", "context-en", "context-ja"] as const;

/** greedy golden の step 数（正本は `export_decode.GREEDY_STEPS`）。 */
const GREEDY_STEPS = 16;

/**
 * 固定長 prefill chunk の行数（ADR 0066 決定 4 の計画時定数）。
 *
 * **32 は「多 chunk prefill を必ず踏む」ための値**。context ケース（T = 87 / 61）が 3 本 / 2 本の
 * chunk に割れ、2 本目以降が `pastLength > 0` の prefill（`M > 1` かつ過去を持つ形 — decode の
 * `M = 1` とも 1 本目の `P = 0` とも別の経路）になる。全ケースが 1 chunk に収まる値にすると、
 * states 形 attention の「過去 + 今 chunk の両側述語」が実重みで 1 度も踏まれない。
 */
const CHUNK_LENGTH = 32;

/**
 * KV スロットの容量記号（`export_decode.CAPACITY_SYMBOL`）と、この門が束縛する値。
 *
 * 128 は全ケースの `T + K`（最長は context-en の 87 + 16 = 103）を覆う。覆えているかは
 * ケースごとに実測で見る（下の {@link CAPACITY} 検査）— prompt が伸びた再エクスポートで
 * 容量が足りなくなったとき、無音で丸まらず落ちる位置をここに置く。
 */
const CAPACITY_SYMBOL = "C";
const CAPACITY = 128;

/**
 * この系列が引ける絶対位置の排他的上限（正本は `export_decode.ROPE_TABLE_POSITIONS` = RoPE 表の
 * 行数）。表の外の gather は非有限を返し argmax が沈黙誤 token に畳むので、`generateGreedy` は
 * この値を要求して入口で落とす。
 */
const MAX_POSITION = 512;

/** グラフ入力の名前（正本は `export_decode.INPUT_IDS` / `POSITION_IDS`）。 */
const INPUT_IDS = "input_ids";
const POSITION_IDS = "position_ids";

/** 層数（config の `num_hidden_layers`）と state スロット本数（層ごとに k / v の 2 本）。 */
const LAYERS = 24;
const SLOTS = LAYERS * 2;

/** q 側 / k・v 側の head 数（16:2 = 真の GQA 形。census の `:gqa` の根拠）。 */
const HEADS = 16;
const KV_HEADS = 2;

/**
 * prefill logits（`[1,M,130560]`）の torch CPU 期待値との突合に使う許容誤差。
 *
 * 波 A の 1-shot 門（`e2e_minicpm5_test.ts` の `MINICPM5_TOLERANCE`）と**同じ閾値**にしてある —
 * 比べているものが同じ（同じ重み・同じ位置の logits）で、違うのは経路（1 本道 か chunk +
 * KV スロット）だけなので、閾値が違うと「経路差なのか閾値差なのか」が読めなくなる。
 *
 * 実測（2026-08-18・context-en / T=87 を 3 chunk で流した有効 87 行）は **maxAbs 7.439e-5** で、
 * 1-shot 門が同じケースで測った 7.44e-5 と**同値**（atol 1e-3 の 13 分の 1）。chunk 分割 +
 * KV スロット経由でも 1 本道と同じ誤差しか出ない、が実測の中身。
 *
 * **rtol は 0**（判定は atol 単独 — 理由は 1-shot 側の docstring に書いた「参照値ちょうど 0.0 の
 * 要素が数本ある」がそのまま当てはまる）。
 */
const PREFILL_ATOL = 1e-3;

/**
 * ディレクトリの列挙。存在しない場合だけ空に縮退する。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      return [];
    }
    throw cause;
  }
};

const discoverCases = (prefix: string): readonly string[] =>
  listDir(SERIES_ROOT)
    .filter((entry) => entry.isFile && entry.name.startsWith(prefix) && entry.name.endsWith(SUFFIX))
    .map((entry) => entry.name.slice(prefix.length, entry.name.length - SUFFIX.length))
    .sort();

/** 登録時点で必要なので同期列挙する（`Deno.test` の ignore 判定と同じ理由）。 */
const IO_CASES = discoverCases(IO_PREFIX);
const GREEDY_CASES = discoverCases(GREEDY_PREFIX);

/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const AVAILABLE = IO_CASES.length > 0 || GREEDY_CASES.length > 0;
/**
 * **何か 1 つでも**残っているか（完全性テストの SKIP 述語 — Codex 波 H 指摘 H-02）。
 * golden が全滅してモデルだけ残った欠損は `AVAILABLE` では偽になり、完全性テスト自身が
 * SKIP される — 欠損を FAIL にする述語は「完全に空」でだけ寝てよい。
 */
const ANY_PRESENT = AVAILABLE || modelPresent(new URL(MODEL_FILE, SERIES_ROOT));

if (!AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため MiniCPM5-1B decode 検収を ` +
      `SKIP する（f32 重み 4.03GiB につきリポジトリ管理外）。生成: ${GENERATE_COMMAND}`,
  );
}

/**
 * ファイル 1 本を `ArrayBuffer` として読む。
 *
 * MUST: view が buffer 全体を覆っているなら slice しない — model.safetensors は 4.03GiB で、
 * 無条件の `slice` は峰値を 8GiB に倍増させる（1-shot 門と同じ理由）。
 */
const readBuffer = async (file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, SERIES_ROOT));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/**
 * golden の 1 テンソルを宣言 dtype の view で取る（コピーしない）。
 *
 * MUST: 格納 dtype が食い違ったら落とす — 要素は f32 も i32 も 4 バイトなので、黙って読み替えると
 * ビット列の再解釈が「通ってしまう」。runtime 側テストの同名ヘルパ（`helpers/golden-io.ts`）は
 * import しない（パッケージのテストが他パッケージのテスト内部に依存しない — `helpers/gpu.ts` の
 * NOTE と同じ分担）。
 */
const goldenF32 = (file: SafetensorsFile, name: string): Float32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden に '${name}' が無い`);
  assertEquals(view.dtype, "F32", `golden '${name}' の格納 dtype`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden に '${name}' が無い`);
  assertEquals(view.dtype, "I32", `golden '${name}' の格納 dtype`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** `greedy.<case>.safetensors`（decode 検収の正本）。 */
type GreedyGolden = {
  readonly prompt: Int32Array<ArrayBuffer>;
  readonly expected: Int32Array<ArrayBuffer>;
  readonly margin: Float32Array<ArrayBuffer>;
};

const loadGreedy = async (caseName: string): Promise<GreedyGolden> => {
  const file = parseSafetensors(await readBuffer(`${GREEDY_PREFIX}${caseName}${SUFFIX}`));
  assertEquals(
    [...file.tensors.keys()].sort(),
    ["expected", "margin", "prompt"],
    `${GREEDY_PREFIX}${caseName}${SUFFIX} のテンソルキー`,
  );
  return {
    prompt: goldenI32(file, "prompt"),
    expected: goldenI32(file, "expected"),
    margin: goldenF32(file, "margin"),
  };
};

/** `io.<case>.safetensors`（無 pad 全長 1 回の入出力）。 */
type IoGolden = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly positions: Int32Array<ArrayBuffer>;
  readonly logits: Float32Array<ArrayBuffer>;
  readonly tokens: Int32Array<ArrayBuffer>;
};

const loadIo = async (caseName: string): Promise<IoGolden> => {
  const file = parseSafetensors(await readBuffer(`${IO_PREFIX}${caseName}${SUFFIX}`));
  assertEquals(
    [...file.tensors.keys()].sort(),
    [`input.${INPUT_IDS}`, `input.${POSITION_IDS}`, "output.0", "output.1"].sort(),
    `${IO_PREFIX}${caseName}${SUFFIX} のテンソルキー`,
  );
  return {
    ids: goldenI32(file, `input.${INPUT_IDS}`),
    positions: goldenI32(file, `input.${POSITION_IDS}`),
    logits: goldenF32(file, "output.0"),
    tokens: goldenI32(file, "output.1"),
  };
};

/**
 * states 形 decode グラフの形の前提（**数値が合ったまま静かに壊れる**性質だけを見る）。
 *
 * 完全な形検査は書き手側（`export_decode.assert_ir_form_decode`）が持つ。読み手側で要るのは
 * 「この門が回している 2 形が、期待どおりの入口・出口・スロットを持つグラフか」の確認だけで、
 * ここが緩むと**別のグラフを検収してしまう**（例: mask 入力の残った従来形は chunk 局所の
 * causal になり、prompt が 1 chunk に収まるケースだけ数値が合う）。
 */
const assertDecodeForm = (parsed: PreparedModel): void => {
  const graph = parsed.graph;
  assertEquals(graph.inputs.map((spec) => spec.name), [INPUT_IDS, POSITION_IDS], "グラフ入力");
  assertEquals(graph.outputs.length, 2, "graph.outputs の本数（logits / token の 2 本）");

  // 出力 1 の供給元が argmax（ADR 0068 決定 4 の decode 出口）。`generateGreedy` はこの出力を
  // そのまま次 step の入力にするので、ここが別 op（例: 1-shot 形の logits）だと「形の合う
  // 別テンソルを読んだ token 列」が例外なしに出来上がる。
  const producer = new Map<string, string>();
  for (const node of graph.nodes) {
    for (const out of node.outs) producer.set(out, node.op);
  }
  assertEquals(producer.get(graph.outputs[1]), "argmax", `出力 1 ('${graph.outputs[1]}') の供給元`);

  // states は層ごとに k / v の 2 本。attention は全て states 欄を持ち、真の GQA 形（16:2）。
  assertEquals(Object.keys(graph.states).length, SLOTS, "states スロットの本数");
  const attentions = graph.nodes.filter((node) => node.op === "attention");
  assertEquals(attentions.length, LAYERS, "attention ノードの本数（= 層数）");
  attentions.forEach((node, index) => {
    assertEquals(
      Object.keys(node.states).sort(),
      ["k", "v"],
      `attention[${index}]: states 欄（mask 込みの従来形が残っていないこと）`,
    );
    const heads = node.ins.map((name) => graph.values[name].shape[1]);
    assertEquals(heads, [HEADS, KV_HEADS, KV_HEADS], `attention[${index}]: head 軸`);
  });
  assertEquals(
    graph.nodes.filter((node) => node.op === "state_append").length,
    SLOTS,
    "state_append ノードの本数（スロット 1 本につき 1 本）",
  );

  // 記号は 2 つ: `M`（物理 chunk 行数 — 入力 shape から解ける）と `C`（容量 — states にしか
  // 現れない専用記号で、束縛点は createGenerationContext だけ・ADR 0066 追記 7）。
  assertEquals([...graph.symbols].sort(), ["C", "M"], "graph.symbols");
};

/** chunk 1 本ぶんの i32 入力（token id 列も絶対位置列も `[1, rows]`）。 */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** 実行後の診断から PreparedPlan の実績を取る（未実行なら fail loudly）。 */
const preparedOf = (diagnostics: SessionDiagnostics, where: string) => {
  const prepared = diagnostics.lastRunPrepared;
  assert(prepared !== undefined, `${where}: lastRunPrepared が無い（run が決着していない）`);
  return prepared;
};

Deno.test({
  name: "MiniCPM5 decode 資産: 期待するケースとモデル本体が揃っている",
  // 完全に空の環境だけ SKIP。**何か 1 つでも**あれば欠けは FAIL（`ANY_PRESENT` の JSDoc）。
  ignore: !ANY_PRESENT,
  fn: () => {
    assertEquals(IO_CASES, [...EXPECTED_IO_CASES], `${SERIES_ROOT.pathname} の io ケース`);
    assertEquals(
      GREEDY_CASES,
      [...EXPECTED_GREEDY_CASES],
      `${SERIES_ROOT.pathname} の greedy ケース`,
    );
    assert(modelPresent(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} が無い`);
  },
});

/**
 * ①形 / ②greedy parity / ③prefill logits / ⑤計画性質。
 *
 * MUST: Session は **1 本だけ組んで全ステップで共有する**。この資産は 4.03GiB の f32 重みで、
 * ステップごとに組み直すと同じアップロードを何度も払う（1-shot 門が 4 ケースで 1 Session に
 * したのと同じ理由）。ステップ間で共有するのは Session（不変の重みと計画キャッシュ）だけで、
 * **GenerationContext は用途ごとに作り分ける** — 生成 1 本 = context 1 本（ADR 0066 決定 6）で、
 * 使い回すと前の生成の KV が残った状態から prefill を始めることになる。
 */
Deno.test({
  name: "MiniCPM5 decode 検収: 固定 token id 列の parity（実 GPU / torch CPU 期待値）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const shards = resolveShards(new URL(MODEL_FILE, SERIES_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    const [logitsName, tokenName] = parsed.graph.outputs;

    await t.step("① 形の前提: states 形 chunk グラフである", () => {
      assertDecodeForm(parsed);
    });

    const gpu = await acquireGpu();
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    try {
      await t.step("② greedy parity: 3 ケース × 16 step が torch の期待列と厳密一致", async () => {
        /** ケースごとの prefill chunk 本数（多 chunk prefill を踏んだことの実測）。 */
        const chunkCounts: number[] = [];
        for (const caseName of EXPECTED_GREEDY_CASES) {
          const golden = await loadGreedy(caseName);
          const prompt = [...golden.prompt];
          const expected = [...golden.expected];
          assertEquals(expected.length, GREEDY_STEPS, `${caseName}: golden の step 数`);

          // golden 側の資格（**恒真でない**）: torch 側の 1 位 / 2 位の差が atol の 2 倍を
          // 超えていれば、③の許容内の数値差では 1 位が動けない = この列は数値差で割れない。
          // 台本側は `MARGIN_FLOOR = 1e-2` で採るケースを選んでおり、実測の最小余裕は
          // context-ja の 0.0261（step 15）= atol 1e-3 の 26 倍。
          const minMargin = Math.min(...golden.margin);
          assert(
            minMargin > 2 * PREFILL_ATOL,
            `${caseName}: golden の最小余裕 ${minMargin} が atol と同程度 — ` +
              `この形では token 列の一致が数値差で反転しうる（門として成立しない）`,
          );
          // 容量が prompt + 継続を覆っていること（覆えないと state スロットが溢れる）。
          assert(
            prompt.length + GREEDY_STEPS <= CAPACITY,
            `${caseName}: T ${prompt.length} + K ${GREEDY_STEPS} が容量 ${CAPACITY} を超える`,
          );

          const started = performance.now();
          const generated = await generateGreedy({
            session,
            inputIds: INPUT_IDS,
            // この系列は RoPE 表を焼いたままなのでグラフ入力は今も `position_ids` 1 本だが、
            // 位置の供給口は `GreedySpec.derive` に一本化された（ADR 0091 決定 1 — gemma4 側が
            // 位置から cos / sin を組む形になり、`positionIds` 欄は退役した）。
            derive: (_ids, positions) => ({ [POSITION_IDS]: i32Row(positions.length, positions) }),
            token: tokenName,
            chunkLength: CHUNK_LENGTH,
            maxPosition: MAX_POSITION,
            // 容量記号は context 生成だけへ渡る（run の bindings には出ない — 追記 7）。
            bindings: { [CAPACITY_SYMBOL]: CAPACITY },
            prompt,
            maxNewTokens: GREEDY_STEPS,
          });
          const elapsed = performance.now() - started;

          // 検収そのもの。tolerance を持たない離散の一致で、GQA の head 写像違い・RoPE の
          // 位置ずれ・KV スロットの行写像違い・pad 行の混入はどれもここで列を割る。
          assertEquals(generated, expected, `${caseName}: 生成 token 列`);

          const chunks = Math.ceil(prompt.length / CHUNK_LENGTH);
          chunkCounts.push(chunks);
          console.log(
            `[e2e] minicpm5 decode greedy ${caseName}: T=${prompt.length} ` +
              `(prefill ${chunks} chunk) + K=${GREEDY_STEPS} step / 最小余裕 ` +
              `${minMargin.toExponential(3)} / ${elapsed.toFixed(0)}ms`,
          );

          // ⑤ decode 経路の計画性質: 生成 1 本で導出される計画は **prefill 形（M=32）と
          // decode 形（M=1）の 2 本だけ**（ADR 0066 決定 4）。ここが step 数に比例して伸びる
          // 形（例: queryLength が計画鍵に混ざる）は、値は正しいまま毎 step 再導出になる。
          // ケースを跨いでも 2 本のまま（プロンプト長は計画鍵に効かない）。
          const prepared = preparedOf(session.diagnostics(), caseName);
          assertEquals(prepared.cachedPlans, 2, `${caseName}: 導出済み計画の本数`);
          assertEquals(
            prepared.hit,
            true,
            `${caseName}: 最終 decode step が計画キャッシュに当たる`,
          );
        }
        // 多 chunk prefill（`pastLength > 0` かつ `M > 1`）を最低 1 ケースが踏んだこと。
        assert(
          Math.max(...chunkCounts) >= 2,
          `全ケースが 1 chunk に収まっている（chunk 本数 ${chunkCounts}）— ` +
            `chunkLength ${CHUNK_LENGTH} では過去つき prefill が 1 度も走らない`,
        );
      });

      await t.step(
        "③ prefill logits: context-en の 3 chunk を有効行だけ突合（診断線）",
        async () => {
          // ②が離散の一致だけを見るのに対し、ここは**連続値**を見る。列が割れないまま数値が
          // じわりとずれる変更（融合の外れ・縮約順序の変化）は②を素通りするので、その線を
          // ここが受け持つ。`generateGreedy` は通さず、chunk の組み立て（pad 0・絶対位置・
          // 物理行数固定）を**この場で書く** — 生成ループと同じヘルパを使うと、割り方の誤りが
          // 期待値の側にも同じように入って恒真化する。
          //
          // 故障注入の実測（2026-08-18）: 2 本目以降の chunk だけ絶対位置を +1 すると、最初に
          // ずれた行（位置 33）の argmax が即座に割れる。一方**全行を一様に +1** しても
          // maxAbs 8.9e-5 で素通りする — RoPE は相対位置しか見ないので、これは門の弱さでは
          // なく不変性そのもの（この経路で捕まえられるのは chunk 間の相対ずれ・行の取り違え・
          // pad 行の混入で、絶対位置の一様なオフセットではない）。
          const io = await loadIo("context-en");
          const total = io.ids.length;
          const vocab = io.logits.length / total;
          assertEquals(io.tokens.length, total, "io の token 行数");
          assert(
            Number.isSafeInteger(vocab),
            `io の logits 長 ${io.logits.length} が行数で割れない`,
          );

          const context = await session.createGenerationContext({
            bindings: { [CAPACITY_SYMBOL]: CAPACITY },
            chunkLength: CHUNK_LENGTH,
          });
          let worst = 0;
          let worstAt = "";
          let chunks = 0;
          let totalQueryLength = 0;
          try {
            for (let position = 0; position < total; position += CHUNK_LENGTH) {
              const queryLength = Math.min(CHUNK_LENGTH, total - position);
              totalQueryLength += queryLength;
              // pad 行は 0 のまま（ADR 0066 追記 6 の値契約）— `Int32Array` の初期値がその契約。
              const ids = new Int32Array(CHUNK_LENGTH);
              const positions = new Int32Array(CHUNK_LENGTH);
              for (let row = 0; row < queryLength; row += 1) {
                ids[row] = io.ids[position + row];
                positions[row] = position + row;
              }
              const outputs = await session.run(
                {
                  [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
                  [POSITION_IDS]: i32Row(CHUNK_LENGTH, positions),
                },
                undefined,
                { context, queryLength },
              );
              chunks += 1;

              const logits = outputs[logitsName];
              assertEquals(logits.dtype, "f32", "logits の dtype");
              assertEquals(
                logits.shape,
                [1, CHUNK_LENGTH, vocab],
                "logits の shape（物理 chunk 形）",
              );
              const tokens = outputs[tokenName];
              assertEquals(tokens.dtype, "i32", "token の dtype");
              assertEquals(tokens.shape, [1, CHUNK_LENGTH, 1], "token の shape");

              for (let row = 0; row < queryLength; row += 1) {
                const where = `位置 ${position + row}`;
                // token は離散なので厳密一致（②と同じ判定を chunk の途中行にも掛ける）。
                assertEquals(tokens.data[row], io.tokens[position + row], `${where}: argmax token`);
                const base = row * vocab;
                const reference = (position + row) * vocab;
                for (let column = 0; column < vocab; column += 1) {
                  const delta = Math.abs(
                    logits.data[base + column] - io.logits[reference + column],
                  );
                  // MUST: NaN を「最悪値」として**残す**書き方にする（`!(delta <= worst)` だと
                  // 一度 NaN を掴んでも次の有限値で上書きされ、非有限が黙って消える）。
                  if (Number.isNaN(delta) || delta > worst) {
                    worst = delta;
                    worstAt = `${where} / 列 ${column}`;
                  }
                }
              }
            }
          } finally {
            // MUST: 途中で落ちても返す（KV 容量ぶんの VRAM を抱えたままにしない）。
            await context.dispose();
          }
          // 恒真回避: `chunks` はループ反復数そのものなので自己比較になる（G4 L-4）。
          // 各 chunk の queryLength の総和が有効行数 total と一致することを実効に検査する。
          assertEquals(totalQueryLength, total, "chunk の queryLength 総和");
          console.log(
            `[e2e] minicpm5 decode prefill context-en: ${chunks} chunk / 有効 ${total} 行 / ` +
              `logits maxAbs ${worst.toExponential(3)}（最悪 ${worstAt}）`,
          );
          // 非有限（NaN / inf）はここで必ず落ちる（`worst` が NaN なら `<=` は偽）。
          assert(
            worst <= PREFILL_ATOL,
            `context-en: prefill logits の maxAbs ${worst} が atol ${PREFILL_ATOL} を超えた` +
              `（最悪 ${worstAt}）`,
          );
        },
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** census 1 回ぶんの内訳（states 形カーネル族の dispatch 数）。 */
type StateCensus = {
  readonly qk: number;
  readonly stats: number;
  readonly pv: number;
  readonly append: number;
};

/**
 * ④census（ADR 0058 決定 4 / 0067 受入条件 ③）。states 形の 4 族が**実際に走った**ことと、
 * q 側の 2 族に `:gqa` が付いたことをパイプラインキーの側から見る。
 *
 * 数値（②③）はキーを区別しない — `repeat_kv` を実体化した形も、融合 attention の従来形も、
 * 正しく実装されていれば同じ値を出す。**キーだけがこの区別をする**。
 *
 * NOTE: `:gqa` が付くのは ①QK と ③PV だけ。②行統計と `state_append` は GQA の軸を持たない
 * （キー生成が `stateVariantKeyPart(sliding, false)` — src/kernels/state-attention.ts）ので、
 * 「4 族すべてに `:gqa`」は成立しない。ここは**付くべき 2 族に付き、付かないべき 2 族に
 * 付かない**の両側で見る。
 */
const assertStateCensus = (diagnostics: SessionDiagnostics, where: string): StateCensus => {
  const entries = diagnostics.lastRunTiming?.entries;
  // MUST: 計測が無効な run を「0 本」として数えない — キー検査が黙って空振りする。
  assert(entries !== undefined, `${where}: lastRunTiming が無い（計測が有効な device のはず）`);
  assert(entries.length > 0, `${where}: 内訳が空（キー検査が空振りしている）`);

  const family = (prefix: string) => entries.filter((entry) => entry.key.startsWith(prefix));
  const qk = family("attention_state_qk");
  const stats = family("attention_state_stats");
  const pv = family("attention_state_pv");
  const append = family("state_append");
  const dispatches = (of: readonly { readonly dispatchCount: number }[]): number =>
    of.reduce((total, entry) => total + entry.dispatchCount, 0);
  const census: StateCensus = {
    qk: dispatches(qk),
    stats: dispatches(stats),
    pv: dispatches(pv),
    append: dispatches(append),
  };

  const shown = entries.map((entry) => `${entry.key}×${entry.dispatchCount}`).join(" / ");
  // ① 4 族が実効（0 本の族があれば、その段は別経路で走ったか走っていない）。
  assert(
    census.qk > 0 && census.stats > 0 && census.pv > 0 && census.append > 0,
    `${where}: states 形の族に 0 本がある ${JSON.stringify(census)}（走った内訳: ${shown}）`,
  );
  // 全層ぶん出ている（行ブロック分割 — ADR 0060 / 0067 決定 7 — は増える側なので下限で見る）。
  assert(
    census.qk >= LAYERS && census.stats >= LAYERS && census.pv >= LAYERS,
    `${where}: attention の dispatch が層数 ${LAYERS} に足りない ${JSON.stringify(census)}`,
  );
  assert(
    census.append >= SLOTS,
    `${where}: state_append の dispatch がスロット数 ${SLOTS} に足りない ${JSON.stringify(census)}`,
  );

  // ② 16:2 なので ①QK / ③PV は全て `:gqa` 変種。1 本でも素のキーが出たら、その層は k / v を
  // H まで広げた形で走ったか、GQA 判定が落ちている。
  assertEquals(
    [...qk, ...pv].filter((entry) => !entry.key.endsWith(":gqa")).map((entry) => entry.key),
    [],
    `${where}: 16:2 の GQA なのに非 GQA キーが走った（走った内訳: ${shown}）`,
  );
  // 逆側 — 行統計と append に `:gqa` は付かない（GQA の軸を持たない）。
  assertEquals(
    [...stats, ...append].filter((entry) => entry.key.includes(":gqa")).map((entry) => entry.key),
    [],
    `${where}: 行統計 / state_append に GQA 変種が出た（走った内訳: ${shown}）`,
  );
  // `window` attrs を持たない full 形なので sliding 変種は 1 本も出ない（出たら窓外を切る）。
  assertEquals(
    entries.filter((entry) => entry.key.includes(":sliding")).map((entry) => entry.key),
    [],
    `${where}: full 形なのに sliding 変種が走った（走った内訳: ${shown}）`,
  );
  // ③ 非 states の attention 系が 0 本（融合 attention の別族へ落ちていない）。
  assertEquals(
    entries
      .filter((entry) =>
        entry.key.startsWith("attention") && !entry.key.startsWith("attention_state")
      )
      .map((entry) => entry.key),
    [],
    `${where}: states 形でない attention のキーが混ざっている（走った内訳: ${shown}）`,
  );
  return census;
};

if (AVAILABLE && GPU_AVAILABLE && !TIMESTAMP_QUERY_AVAILABLE) {
  console.warn(
    "[karume] アダプタが 'timestamp-query' を列挙しないため MiniCPM5 decode の census を " +
      "SKIP する（parity と logits の 2 本は残る — ADR 0021 の計測は device 作成時にしか" +
      "要求できない）",
  );
}

/**
 * ④census（実 GPU / timestamp-query）。
 *
 * MUST: 計測を要求しない device では**明示 SKIP** する。`acquireGpu({ gpuTiming: true })` は
 * feature 不在で fail loudly するので、渡し忘れや縮退で `lastRunTiming` が undefined になった
 * 形は上の {@link assertStateCensus} が落とす。
 * NOTE: 数値の門（上のテスト）は素の `acquireGpu()` で走らせる — 常用と同じ device 構成のまま
 * 突合したいので、timestamp 書き込みが混ざる構成はこのテストに閉じる（1-shot 門と同じ分担で、
 * 4.03GiB の Session をもう 1 本組む代償はそこで払う）。
 */
Deno.test({
  name: "MiniCPM5 decode census: states 形カーネル族が GQA 変種で走る（実 GPU / timestamp-query）",
  ignore: !AVAILABLE || !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const shards = resolveShards(new URL(MODEL_FILE, SERIES_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    const [, tokenName] = parsed.graph.outputs;
    // 最短ケース（T=6）で足りる — 見るのは走ったパイプラインの種類と本数。
    const golden = await loadGreedy("capital-en");
    assert(golden.prompt.length <= CHUNK_LENGTH, "census は 1 chunk で踏むケースを使う");

    const gpu = await acquireGpu({ gpuTiming: true });
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    const context = await session.createGenerationContext({
      bindings: { [CAPACITY_SYMBOL]: CAPACITY },
      chunkLength: CHUNK_LENGTH,
    });
    try {
      const ids = new Int32Array(CHUNK_LENGTH);
      const positions = new Int32Array(CHUNK_LENGTH);
      for (let row = 0; row < golden.prompt.length; row += 1) {
        ids[row] = golden.prompt[row];
        positions[row] = row;
      }
      const prefill = await session.run(
        {
          [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
          [POSITION_IDS]: i32Row(CHUNK_LENGTH, positions),
        },
        undefined,
        { context, queryLength: golden.prompt.length },
      );
      const prefillCensus = assertStateCensus(session.diagnostics(), "prefill");

      // decode 形（M=1）は prefill とは別の計画なので、キーの検査も 1 度ずつ要る。
      const first = prefill[tokenName].data[golden.prompt.length - 1];
      assertEquals(first, golden.expected[0], "prefill の最終有効行が greedy golden の 1 手目");
      await session.run(
        {
          [INPUT_IDS]: i32Row(1, Int32Array.of(first)),
          [POSITION_IDS]: i32Row(1, Int32Array.of(golden.prompt.length)),
        },
        undefined,
        { context, queryLength: 1 },
      );
      const decodeCensus = assertStateCensus(session.diagnostics(), "decode");
      console.log(
        `[e2e] minicpm5 decode census: prefill ${JSON.stringify(prefillCensus)} / ` +
          `decode ${JSON.stringify(decodeCensus)}`,
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});
