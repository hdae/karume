/**
 * Session の公開型面。executor.ts の前半にあった宣言のみの公開 API 型群と、
 * {@link SessionOptions} の鍵になる unique symbol をここへ移した（executor.ts は再輸出で
 * 公開面を保つ）。MUST: executor.ts へ import を張らない（循環 import の禁止）。
 */

import type { IrDtype } from "../format/ir.ts";
import type { ArenaStats } from "../gpu/arena.ts";
import type { BatchScope, ResidentTensor } from "../gpu/device.ts";
import type { GpuTimingStats, SubmitPolicy, SubmitStats } from "../gpu/submit.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import type { FusionCounts } from "./fusion.ts";
import type { GenerationContext } from "./generation-context.ts";
import type { SymbolBindings } from "./plan.ts";
import type { SharedWeight } from "./weight-residency.ts";

type TensorOf<D extends IrDtype, A> = {
  readonly dtype: D;
  /** 束縛解決済みの具体値。 */
  readonly shape: readonly number[];
  readonly data: A;
};

/**
 * 意味論 dtype で判別するテンソル（ADR 0009 — ADR 0008 の公開面の部分改訂）。
 *
 * 要素は全型 4 バイトで、**bool は u32 の 0 / 1**（WebGPU のストレージバッファに 1bit 型が
 * 無いため。GPU 側の格納と同じ規約）。入力・出力とも同じ形で扱う。
 *
 * MUST NOT: **入力として渡した `data` は `run` / `enqueue` の戻り Promise が settle するまで
 * 書き換えない**（borrowed — ランタイムは写しを取らない）。GPU への `writeBuffer` は発行の
 * 同期区間ではなくマイクロタスクの先で出るので、書き換えは例外も警告も無い沈黙誤値になる。
 * `shape` の方は発行時に複製されるため、書き換えても実行中の run には影響しない（契約の全体は
 * `Session.run` の「入力の寿命」節）。出力として受け取った `Tensor` は呼び出し側の所有物で、
 * この制約は掛からない。
 */
export type Tensor =
  | TensorOf<"f32", Float32Array<ArrayBuffer>>
  | TensorOf<"i32", Int32Array<ArrayBuffer>>
  | TensorOf<"bool", Uint32Array<ArrayBuffer>>;

/**
 * 実行の入力 1 本。ホスト配列（{@link Tensor}）か、GPU 常駐のまま束ねる
 * {@link ResidentTensor}（第 4 の寿命クラス）。
 *
 * MUST: 常駐入力は **`writeBuffer` を 1 度も出さない** — バッファをそのまま bind group へ
 * 焼き込む。したがってホスト側に shape も dtype も無く、検査できるのは**大きさだけ**
 * （宣言 shape ぶんと厳密一致）。記号次元はその常駐入力からは束縛されないので、他の入力か
 * `bindings` で決まっていなければ fail loudly（{@link bindSymbols} の `deferredInputs`）。
 */
export type RunInput = Tensor | ResidentTensor;
/**
 * 実行の入力の束（名前 → 入力 1 本）。
 *
 * **この Record の member 構成は `run` / `enqueue` の発行時に写し取られる**ので、戻り Promise を
 * 待たずに差し替えても、その実行は発行時点の顔ぶれで走る（同じく shape と `bindings` も固定
 * される）。写すのは metadata だけで、{@link Tensor} の `data` は borrowed のまま — 契約の全体は
 * `Session.run` の「入力の寿命」節。
 */
export type RunInputs = Readonly<Record<string, RunInput>>;
export type RunOutputs = Readonly<Record<string, Tensor>>;

/**
 * generation実行1回ぶんの指定（Session.runの第3引数、EnqueueOptions.generation）。
 *
 * `queryLength` は今 step の実 token 数（prefill は `1..chunkLength`・decode は 1）で、
 * **`pastLength` は渡さない** — 論理長の進行はcontextが所有し、runまたはbatchの最終成功でのみ進む
 * （ADR 0066 決定 6 の二重簿記の禁止）。
 */
export type GenerationRun = {
  readonly context: GenerationContext;
  readonly queryLength: number;
  /**
   * 論理長を進める時点（既定 `"immediate"` = 従来 — run が例外なく返った時点で `queryLength` 行
   * ぶん進む）。enqueueではbatchが例外なく完了するまで進めない。
   *
   * `"deferred"` は進行を保留し、**受理した行数**を後から `GenerationContext.commit(rows)` で
   * 確定させる（投機デコードの検証形 — draft の何行が受理されるかは、その run の出力を読んで
   * 初めて決まる）。保留がある間は次の run と `rewind` を拒否するので、論理長を動かす経路は
   * 依然 1 本のまま（ADR 0066 決定 6 の二重簿記の禁止）。
   */
  readonly commit?: "immediate" | "deferred";
};

/** {@link Session.enqueue} の指定。 */
export type EnqueueOptions = {
  /** 束ねる区間（{@link GpuContext.beginBatch}）。フェンスはこの区間の決着 1 本だけ。 */
  readonly batch: BatchScope;
  /**
   * 会話状態を使う実行。使用予約は発行時からbatchの最終決着まで保つ。
   * 論理長のadvance/deferはenqueueの戻り時でなく、finish/finishAndReadの成功時に行う。
   * 同じcontextを未確定のまま重ねて使えない。context/Sessionのdisposeはfinish後に行う。
   */
  readonly generation?: GenerationRun;
  /**
   * 記号次元の明示指定。常駐入力は束縛源にならないので、その入力**だけ**が持つシンボルは
   * ここで与える（`run` の第 2 引数と同じ意味）。
   */
  readonly bindings?: SymbolBindings;
  /**
   * グラフ出力 → 書き出し先の常駐テンソル。dispatch 列の**後**に同じコマンド列へ
   * `copyBufferToBuffer` を積む（readback もフェンスも伴わない）。
   *
   * MUST: 大きさは宣言 shape ぶんと厳密一致（fail loudly）。`enqueue` は readback をしないので、
   * ここに載せなかった出力は次の同一 signature の enqueue で slot ごと上書きされて消える。
   */
  readonly copyOutputs?: Readonly<Record<string, ResidentTensor>>;
};

/**
 * {@link Session.enqueueRead} の戻り — 受理とグラフ出力の 2 本の決着。
 *
 * `admitted` は {@link Session.enqueue} の戻りと同じ意味（エンコードの受理・失敗はここにも出る）。
 * `outputs` は batch の**決着後**にだけ解決する（区間の失敗では同じ理由で拒否する）。
 * MUST NOT: `finish` / `finishAndRead` を呼ぶ前に `outputs` を await する（区間が閉じるまで
 * 解決しないので、その await は永久に返らない）。
 */
export type EnqueueRead = {
  readonly admitted: Promise<void>;
  readonly outputs: Promise<RunOutputs>;
};

/**
 * {@link Session.createGenerationContext} の指定（ADR 0066 決定 6）。
 *
 * スロット容量（`graph.states` の記号次元）と `chunkLength` を確定して物理確保する。context は
 * 入力を 1 本も持たないので、記号次元は**ここで与えた束縛だけ**で決まる（states は束縛源に
 * ならない — ADR 0066 決定 2）。
 */
export type GenerationContextSpec = {
  /**
   * state スロットの shape に現れる記号次元の値。`graph.symbols` に無い名前は fail loudly。
   * 数値次元だけで容量が決まるグラフでは省略できる。
   */
  readonly bindings?: SymbolBindings;
  /**
   * 固定長 prefill chunk の行数（ADR 0066 決定 4 — 計画時定数で、末尾 chunk は pad で埋める）。
   * decode は `queryLength = 1` 固定形なので、この値とは独立に走る。
   *
   * 短い chunk を pad 無しで回す追加の物理形は {@link GenerationContextSpec.chunkBuckets}。
   */
  readonly chunkLength: number;
  /**
   * prefill 形として `chunkLength` に**加えて**許す物理 chunk 行数（ADR 0066 決定 4 /
   * 追記〈バケット〉）。各要素は 2 以上 `chunkLength` 未満の整数で、**狭義昇順**。
   * 省略 / 空配列 = 追加なし（従来どおり prefill 形 1 本 + decode 形の 2 本）。
   *
   * 短い prompt は `chunkLength` 行へ pad すると pad 行ぶんの仕事がそのまま無駄になる
   * （行局所な linear / pointwise / norm は物理行数に比例する）。バケットを宣言すると、
   * 呼び出し側は chunk ごとに `queryLength` 以上の最小バケットを物理行数に選べる。
   * 既定を下げる形を採らないのは、長い prompt では大きい chunk の一括が最速だから。
   *
   * MUST: 本数は「同時に定常化させてよい PreparedPlan の本数」でもある（実行形 1 本 =
   * 別鍵の計画 1 本 — ADR 0042 決定 2 の LRU）。増やしすぎると decode のホットパスが
   * 追い出しで静かに再導出へ落ちる。
   */
  readonly chunkBuckets?: readonly number[];
  /**
   * **借り先の context**（ADR 0096 段 2 §2.1 — drafter が読む target の生成 context）。
   *
   * 指定できるのは「グラフの全スロットが external」のときだけで、逆も MUST（external が
   * あるのに `borrow` 無しは fail loudly）。借り手は自前のスロットを 1 本も確保せず、
   * external スロットを**名前**で貸し手のスロットに束ねる。`bindings` は貸し手のものを継承
   * するので**渡せない**・`chunkLength` は 1 ちょうど・`chunkBuckets` は宣言できない
   * （借り手の実行形は decode 1 本だけ）。
   *
   * 借り手の run は貸し手の run リースを取るので、貸し手の run / commit / rewind と直列化され、
   * 貸し手に `pendingCommit` が残っている間は拒否される（draft は commit の後）。
   */
  readonly borrow?: GenerationContext;
};

/**
 * 整数内積の変種（w8a8 経路）。**両者は同じ整数を返す**ので、これは速度の選択でしかない
 * （src/kernels/linear-i8a8.ts）。
 */
export type I8a8Dot = "dp4a" | "emu";

/**
 * **テスト専用の非公開面**（mod.ts からは輸出しない — ADR 0008 の「薄い面」を汚さない）。
 *
 * i8a8 の整数内積変種を**強制する**（linear / 融合 attention の**全 i8a8 カーネル共通** —
 * 指定した変種が両族に等しく載る）。拡張のある機で `dot4I8Packed` 版とエミュ版を実走して
 * atol=0 で突合するのが「エミュは数値同一」という主張の唯一の機械的検出器で、環境変数では
 * なく Session 単位のノブにしてあるのは 1 プロセス内で両方を回すため。
 *
 * MUST: 強制の意味は族を跨いで 1 つ（既定が族ごとに分かれても、このノブは分けない）。
 * **既定**は族ごとに別々に決まる — linear は言語機能の列挙、attention は device 単位の
 * 実走カナリア（src/gpu/attention-dp4a-canary.ts）。指定するとカナリアは走らない
 * （テストが「この変種で回す」と言っている以上、環境の判定を挟むと何を測ったのか消える）。
 */
export const I8A8_DOT: unique symbol = Symbol("karume.i8a8Dot");

/**
 * **テスト専用の非公開面**（mod.ts からは輸出しない — {@link I8A8_DOT} と同じ流儀）。
 *
 * 分解 attention の行ブロック枚数（src/runtime/fusion.ts の `rowBlockAttention`）を強制する。
 * 既定の枚数は device の `maxStorageBufferBindingSize` から静的に決まるので、**上限に余裕の
 * ある機では常に 1 枚**になり、2 枚以上の経路（行窓カーネル・ブロック跨ぎの full-write）が
 * 1 度も走らない。強制分割はその経路を実機で回して 1 枚実行と Uint32 一致させるための唯一の
 * 手段で、環境変数ではなく Session 単位のノブにしてあるのは 1 プロセス内で両方を回すため。
 *
 * MUST: 上限に収まらない枚数は fail loudly（緩める向きには使えない）。
 */
export const ROW_BLOCK_SPLIT: unique symbol = Symbol("karume.rowBlockSplit");

/**
 * op 族ごとの計算精度ノブ（ADR 0028 / attention の i8a8 は設計 §9.2）。**重み格納の f16
 * （ADR 0018）とは別の軸**で、`"f16"` は共有タイルを f16 に落として内積を回す変種
 * （累積は f32）、`"a8"` は活性を per-token i8 へ量子化して整数内積で回す変種を選ぶ。
 *
 * MUST: 3 値は**相互排他**（直積ではない）。attention の q/k/v は全て活性で格納軸を持たない
 * ので、「f16 かつ a8」という組み合わせは表現する対象がそもそも存在しない。
 * MUST: `"f16"` は `acquireGpu({ shaderF16: true })` を伴う（Session 構築時に fail loudly）。
 * **`"a8"` は `shader-f16` を要求しない**（feature ゲートに混ぜないこと）。
 *
 * NOTE: 値の綴りは 0.5.0 で `"i8a8"` → `"a8"` へ改名した（ADR 0074 決定 3）— このノブが
 * 決めているのは**活性の扱いだけ**で、重みの格納形は資産ヘッダが決める。カーネル側の内部
 * 識別子（`linear-i8a8.ts` / パイプラインキーの `:i8a8:`）は実行変種の名前なので不変。
 */
export type ComputePrecision = "f32" | "f16" | "a8";

/** states 形 attention ①QK / ③PV の縮約形（{@link SessionOptions.stateAttentionReduce}）。 */
export type StateAttentionReduce = "sequential" | "parallel" | "parallel-fused";

/** 実行とメモリ見積りで共有する値域。 */
export const STATE_ATTENTION_REDUCES: Readonly<Record<StateAttentionReduce, true>> = {
  sequential: true,
  parallel: true,
  "parallel-fused": true,
};

/** 量子化 GEMV の K 加算順（ADR 0098）。 */
export type LinearGemvReduce = "sequential" | "parallel" | "parallel-subgroup32";

/** RMS正規化の縮約方式。subgroup32は参照と加算順が異なる（ADR 0100）。 */
export type RmsNormReduce = "workgroup" | "subgroup32";

export type SessionOptions = {
  /** submit の時間予算政策（TDR / watchdog 対策 — ADR 0004）。既定は DEFAULT_SUBMIT_POLICY。 */
  readonly submitPolicy?: SubmitPolicy;
  /**
   * linear の実行形（既定 `"f32"` = 従来どおり）。
   *
   * `"a8"` は **整数常駐（i8 / i4）の重みの linear** に効き、活性を per-token i8 へ
   * 量子化して整数内積で回す。ノブが指すのは「活性の i8 化 + 整数内積」だけで、**重みの
   * 格納形は別軸** — 格納形で数値契約の違う 2 変種に分かれる:
   * - i8 常駐 → **w8a8**: 縮約全体が 1 つの i32（整数部は丸め 0 回）で、dequant は
   *   `xs · wscale` を先に 1 つの f32 へ畳んでから `f32(acc)` へ掛ける形 — 丸めは**その積と
   *   最後の fma の 2 回**（設計 = docs/research/2026-08-03-dp4a-w8a8-design.md。丸め回数の
   *   勘定は fma 融合実装での性質 — 仕様保証ではない。src/kernels/linear-i8a8.ts の MUST）。
   * - i4 常駐 → **w4a8**: group ごとに i32 内積して group 境界で f32 へ flush するので、
   *   丸めは `k/g + 1` 回（同じく融合実装での勘定）。wscale が group ごとに変わり畳めないため
   *   `xs` は**最後の fma 1 回**へ回る。数値契約の差は 2 つ — ①縮約の粒度（全体 vs group 部分縮約 = 丸め回数が
   *   k と g に依る）②`xs` の掛け位置（ADR 0076 決定 2）。起票は
   *   docs/perf-ledger.md の Q-8、契約は src/kernels/linear-i8a8.ts の「w4a8 変種」節。
   *
   * `"f16"` は共有タイルを f16 に落とす計算変種（ADR 0028）で、重み格納が f32 / f16 の
   * linear に効く（**i8 常駐の重みとは組めない** — w8a16 は未実装なので fail loudly）。
   * MUST: 既定は `"f32"` — i8 / i4 / f16 資産を自動で低精度実行にすると既存の PNG sha256 門と
   * E2E tolerance が黙って変わる。opt-in 以外はあり得ない。
   */
  readonly linearCompute?: "f32" | "a8" | "f16";
  /**
   * 融合 attention（ADR 0023 の 3 カーネル）の実行形（既定 `"f32"` = 従来どおり）。
   *
   * `"f16"` は ①QK / ②行統計 / ③PV の共有タイルを f16 にし、**S も f16 で受け渡す**
   * （① が書き ②③ が読む — transient が半減する）。`linearCompute` と**別の軸**なのは、
   * 1024px の内訳が attention 46% / linear 42% で片方だけ f16 にしたい場面が実際にあるため。
   *
   * `"a8"` は q / k / v を i8 へ量子化して整数内積で回す変種
   * （設計 = docs/research/2026-08-04-attention-a8-design.md）。**現時点の意味論は
   * 「①QK と ③PV が i8a8・②行統計は f32 のまま」**（③ の A 側 = P̃ は scale が 1/127 に
   * 構造縮退するので量子化カーネルを通らず、V だけが Vᵀ 経由の per-column i8 になる）。
   * `"f16"` と違い `shader-f16` を要求せず、資産の格納形（f32 / f16 / i8）とも独立に効く —
   * attention の入力は全て活性だから。
   * **適格判定は段ごとに独立で、満たさない段だけが f32 経路へ沈黙で縮退する**
   * （① は `D % 4 == 0`・③ は `N % 4 == 0` — i8 ペイロードの語境界条件で、パック方向が
   * 段ごとに違う）。したがって **`D % 4 == 0` かつ `N % 4 != 0` なら「①QK は i8a8・③PV は
   * f32」の混成**になる。linear の `k % 4` と同じ流儀で、落ちたことは診断のパイプライン
   * キーにだけ出る。
   * MUST: 既定は `"f32"`。
   */
  readonly attentionCompute?: ComputePrecision;
  /**
   * 融合 attention の中間バッファ **S（スコア）の格納形**（既定 `"f32"` = 従来どおり）。
   * `attentionCompute`（計算形）と**直交する第 2 の軸**で、S は中間バッファなので
   * 「どの精度で計算するか」と「どの精度で置くか」を別々に選べる。
   *
   * `"f16"` は S を `array<u32>` に **`pack2x16float` で 2 要素／語**詰める（core WGSL・
   * **`shader-f16` を要求しない** — ADR 0030 決定 1「i8a8 は shader-f16 を要求しない」を
   * 保ったまま `attentionCompute: "a8"` と組める。本命はこの組）。丸めは格納の 1 回だけで、
   * 読み側の `unpack2x16float` は厳密。したがって出力は「**S をホストで f16 に丸めた
   * f32 変種**」とビット単位で一致する。
   *
   * MUST: 既定は `"f32"` — S の格納形を自動で落とすと既存の PNG sha256 門と E2E tolerance が
   * 黙って変わる（ADR 0028 決定 1 が auto を禁じたのと同じ理由）。
   * MUST: `attentionCompute: "f16"`（`:c16`）との併用は **fail loudly**。あちらは S を
   * `array<f16>` で持つ**別の格納形**なので、冗長かつ矛盾する組になる。
   * **適格判定は `D % 4 == 0 && N % 4 == 0`**（書き手 ①QK が v4 経路を取る条件 — 1 スレッドが
   * 4 連続列 = 2 語ちょうどを排他に書く）で、満たさない形は f32 格納へ**沈黙で**縮退する
   * （linear の `k % 4`・ADR 0030 決定 5 と同じ流儀で、落ちたことは診断のパイプライン
   * キーにだけ出る）。
   */
  readonly attentionScoreStorage?: ScoreStorage;
  /**
   * states 形 attention（ADR 0067 決定 4 — 生成 context の KV スロットを読む形）の **①QK と
   * ③PV の縮約形**（既定 `"sequential"` = 参照経路）。**1 つのノブが 2 段を一緒に切り替える**
   * （2026-09-06 裁定 — 段ごとに席を割らない）。
   *
   * `"parallel"` はどちらの段も**縮約を workgroup 内の 16 レーンで分担し固定順の木で畳む**変種:
   * - ③PV は KV 長方向を分担（perf-ledger K-12 — src/kernels/state-attention.ts「③' KV 並列
   *   縮約変種」節）。decode（M=1）で 1 スレッドの逐次長が KV 長に比例して伸びる形を潰す。
   * - ①QK は内積の D 方向を分担（perf-ledger K-14 — 同「①' D 並列縮約変種」節）。1 invocation
   *   が D 本の積和を逐次で回す遅延を縮める。
   *
   * **2 段で適用範囲が違う**（席は 1 つのまま）: **①' が選ばれるのは `M`（物理 chunk 行数）が
   * 8 以下の計画**（decode の M=1 と投機の verify M ≤ 8）で、prefill 計画（M > 8）は `"parallel"`
   * を指定しても ①（逐次）のまま走る。prefill では ① が既に行 × 列で埋まっており、①' に替えると
   * 遅くなると実測したため（2026-09-06 — 適用条件と実測は `stateQkParallelEligible` の WHY）。**③' が選ばれるのは
   * `M < 16` の計画だけ**で、`M ≥ 16` は席に依らず ③ₜ（GEMM 骨格の V 行タイル共有 —
   * perf-ledger K-13 段 2）が取る。③ₜ は ③ と**ビット同一**なので、席が `"parallel"` でも
   * prefill 計画の値は参照経路のものになる（decode = M 1 の計画では従来どおり ③' が効く）。
   *
   * どちらも縮約順が変わるので **参照経路とビット同一ではない**（決定性は保つ）。融合
   * attention（`attentionCompute`）とは別族なので直交する。
   * MUST: 既定は `"sequential"`（ADR 0058 決定 2 — 数値を変える経路の自動選択禁止）。
   *
   * parallel-fused は parallel の加算順を保ち、M<=8・列上限<=1024 の states 形だけ
   * 行統計と PV を融合する。readonly / その他の形は parallel の経路。ADR 0102。
   */
  readonly stateAttentionReduce?: StateAttentionReduce;
  /**
   * 量子化 GEMV の加算順（既定 sequential）。parallel は実測済みの INT2/4/8 形状と M=1..8 に
   * 限る任意指定。f32 演算のみで、対象外形状・格納は従来経路（診断キーで区別）。
   * 加算順が変わるため、QAT の再量子化を含め生成列は既定と一致しない場合がある。
   * M=1 と M=4/8 は同一の加算順を使う。追加重みコピーは無い。ADR 0098。
   * parallel-subgroup32はparallelと同じ配分・加算木を32レーン内の値交換で実行する（ADR 0101）。
   * acquireGpu({ subgroups: true })が必要。不足時は拒否し、自動でparallelへ戻さない。
   */
  readonly linearGemvReduce?: LinearGemvReduce;
  /**
   * 隣接するf32のrms_norm→addを融合する（既定false、ADR 0099）。
   * 最終次元256/1536/2560、同一shape、内部値が専有される形だけに適用する。
   * 丸め保持を実測しているが、未検証GPUで参照とビット同一とは保証しない。
   * submitPolicyは独立の設定で、この指定だけでは投入上限を変更しない。
   */
  readonly fuseRmsNormAdd?: boolean;
  /**
   * 隣接するlinear→static_quantizeを融合する（既定false、ADR 0103）。
   * linearGemvReduce: parallel / linearCompute: f32との組合せのみ対応する。
   * 検収済みINT2/4/8の形状、M=1..8、内部値が専有される形だけ。既存scaleを借用する。
   * 未検証GPUのビット一致は保証せず、falseで非融合の参照へ戻せる。
   */
  readonly fuseLinearStaticQuantize?: boolean;
  /**
   * 固定SRQの活性をpacked int8（u32 1語にint8コード4個）で並列GEMVへ渡す（既定false、ADR 0105）。
   * linearGemvReduce: parallel / linearCompute: f32との組合せのみ対応する。
   * 対象は「素のstatic_quantizeノードで、消費先が全て並列GEMVへ落ちるlinearの活性」だけで、
   * 1本でも別の消費先が混ざる形・k が16の倍数でない形・scale 0（恒等）は従来のf32のまま。
   * 重み1語あたりの活性ロードがi2 16→4本・i4 8→2本・i8 4→1本に減る（research 2026-09-19 §14）。
   * 復元は`f32(code) * scale`で、現行SRQ出力と要素ごとにu32一致する（ADR 0105）。
   * 例外は int8 に席が無い2値 — -0.0は+0.0へ落ち（積和の結果は動かない）、NaNは飽和する。
   */
  readonly packedStaticQuantize?: boolean;
  /**
   * RMSの縮約方式（既定workgroup）。subgroup32は幅128超のRMSと任意のRMS→add融合に適用。
   * acquireGpu({ subgroups: true })が必要。不足時は拒否し、自動で参照へ戻さない。
   * 加算順が変わるため生成列は参照と異なりうる。M2・投機生成の採用は別途検収する。
   */
  readonly rmsNormReduce?: RmsNormReduce;
  /**
   * 行ブロック gemv（linear の GEMV 族・M ≥ 2）の**並列度の目標**（スレッド数 = 出力列 n ×
   * y タイル数）。既定 16384 = 参照 device（RTX 3080 Ti）の飽和点。
   *
   * 飽和点が小さい GPU（内蔵 GPU・Apple M 系）では下げると `rows`（1 スレッドが持つ行数）が
   * 増えて重みの読み直しが減る（機序は src/kernels/linear-gemv.ts の
   * `linearGemvRowsForShape`）。**静的**なノブ — Session 生成時に固定し、実行時に変えない・
   * device を見て自動選択しない（ADR 0022 の実行時オートチューン禁止）。
   * 目標が変われば選ばれる `rows` が変わり、`rows` はパイプラインキーに載るので
   * 「同一キー → バイト同一 WGSL」は保たれる。
   * MUST: 1 以上の安全な整数でなければ fail loudly。
   */
  readonly linearGemvRowsThreadTarget?: number;
  /**
   * slot backing（導出済み計画にヒットした run が使う中間バッファ束）を**同時に保持する予算**
   * （バイト・既定 {@link DEFAULT_PLAN_BACKING_BUDGET_BYTES} = 256 MiB）。
   *
   * run の形（signature）ごとに 1 本の backing があり、生成では prefill 形 ↔ decode 形の切替が
   * 毎ターン起きる。予算内なら切り替えても作り直さず保持し（切替 1 回 ≈ 40 ms の作り直しが
   * 消える — perf-ledger H-15）、超える分は古い順に退役する。**新規 1 本だけで予算を超える形は
   * 他を全て退役させてその 1 本だけを持つ**（= 従来の容量 1 の挙動）ので、常駐は
   * `max(予算, 最大 1 本)` を超えない。予算 0 は従来どおり常に 1 本。
   * 勘定するのは各 backing が抱える VRAM = **領域の総和 + 所有する入力バッファ**
   * （{@link PlanBackingStats.residentBytes} + {@link PlanBackingStats.inputBytes}）。
   * MUST: 非負の安全な整数でなければ fail loudly。
   */
  readonly planBackingBudgetBytes?: number;
  /**
   * **共有 initializer の実体**（ADR 0096 段 2 §1.3 / §2.2 — 借り手の initializer 名 →
   * 貸し手 `Session.exportWeight()` の戻り）。
   *
   * グラフの `shared` 宣言**全部**に対して過不足なく与える MUST。門は 5 点（同一 device・
   * 宣言 shape・消費席（貸し手の codec から導く）・行の軸 — `resolveSharedWeights`）で、どれも破れは
   * 例外ではなく別の値として出るため fail loudly。
   *
   * 寿命: 借り手 Session が生きている間、貸し手 Session の `dispose()` は fail loudly になる
   * （借用計数 — `dispose` の冪等・非 throw 契約からの意図的な逸脱）。
   */
  readonly sharedWeights?: Readonly<Record<string, SharedWeight>>;
  /** テスト専用（{@link I8A8_DOT}）。既定は wgslLanguageFeatures の列挙から決める。 */
  readonly [I8A8_DOT]?: I8a8Dot;
  /** テスト専用（{@link ROW_BLOCK_SPLIT}）。既定は device の limit から静的に決まる枚数。 */
  readonly [ROW_BLOCK_SPLIT]?: number;
};

/**
 * 低精度格納（f16 — ADR 0018 / i8 — ADR 0019）の実績。**ADR 0006 が義務づける常設診断**で、
 * 「f16 / i8 指定なのに適格 0MB」を沈黙させないための唯一の観測点。
 *
 * 対象は圧縮格納の initializer だけ（格納 f32 / i32 はどちらにも数えない）。両方 0 なら
 * 「そのモデルに低精度格納が 1 本も無い」、`resident` が 0 で `hostExpanded` が大きければ
 * 「低精度と宣言したのに適格判定で全部落ちている」— この 2 つが区別できる形にしてある。
 */
export type StorageDiagnostics = {
  /**
   * 圧縮のまま GPU 常駐した重みの **GPU バッファ上のバイト数**（整列のゼロ詰め込み。
   * i8 は **per-channel scale のバッファぶんも加算**する — 実際に GPU が抱えるバイト数を
   * 表す欄なので、scale を除くと VRAM 実績と食い違う）。
   * f32 で持ったときの 1/2（f16）・約 1/4（i8）になるのがこの経路の目的。
   */
  readonly residentCompressedBytes: number;
  /**
   * 適格外でロード時に CPU で f32 展開した重みの、**展開後**のバイト数（= 実際に GPU が
   * 抱えるバイト数）。VRAM 削減はゼロで、縮んだのは配信サイズだけ。
   */
  readonly hostExpandedBytes: number;
};

/**
 * Session 構築相（`Session.build` = 重みアップロードの明示 async ステージ）の費用内訳。
 * 「構築に 2.50s 掛かった」を**どこで消えたか**へ分解するための常設診断で、値は Session の
 * 寿命を通じて不変（構築が終わった時点で確定し、run では 1 つも動かない）。
 *
 * ## 帰属の限界（この分解が到達できる上限）
 *
 * 計測はホスト時計（`performance.now()`）だけで、**GPU フェンスを 1 本も追加しない**
 * （追加すると part ごとの submit 1 回という契約が崩れ、瞬間ピークが重み 1 本ぶん押し上がる —
 * ADR 0108 決定 9）。したがって `queue.writeBuffer` の**実 GPU 転送時間はホスト時計から分離
 * できず**、part 末尾のフェンス待ち（{@link SessionBuildStats.uploadFenceMs}）に丸ごと吸われる。
 * MUST: この 7 席から「転送そのものの速度」を読まないこと。分解の上限は
 * **「ホストが費やした時間」（shardWait / decode / bufferCreate / writeBufferIssue）と
 * 「GPU の完了を待った時間」（uploadFence）の 2 区分**で、後者の内訳（実転送 / キュー待ち /
 * ドライバの都合）はランタイムからは見えない。
 *
 * NOTE: バイト数・本数の席は**既存の診断を流用する**（ここには置かない）— CPU 展開後のバイト数は
 * {@link StorageDiagnostics.hostExpandedBytes}、圧縮常駐のバイト数は
 * {@link StorageDiagnostics.residentCompressedBytes}、確保したバッファ本数は
 * {@link SessionDiagnostics.weights} の `allocCount`。
 * NOTE: グラフ検証・パイプライン生成の費用はここに**入らない** — 前者は構築相の外
 * （`prepareContainer` 相）で、後者は構築相にそもそも存在しない（パイプラインは初回 run で作られる。
 * 構築直後の `pipelineCount` は 0）。混ぜると「構築費」の定義が濁る。
 */
export type SessionBuildStats = {
  /**
   * 消費した供給単位（容器の part）の本数。
   *
   * NOTE: 欄名は旧名（shard）の据え置き — 公開面なので改名は breaking。
   */
  readonly shardCount: number;
  /**
   * 供給の**次の 1 本を待った**時間の総和（`for await` の反復待ち）。待ちは 2 段あり、どちらも
   * 足す — part の列の次の 1 本と、part の中の次の block。block は引かれるたびに 1 本ずつ読まれる
   * （lazy）ので、取得と検証（未検証の取得元での sha256・三値の符号検査）の費用はほぼ全部
   * block 側の待ちに入る。ネットワーク / ディスクの費用がここに集まるので、この席が構築費から
   * その帰属を分離する唯一の点になる（大きければ遅いのは Karume ではなく供給側）。欄名は旧名の
   * 据え置き。
   */
  readonly shardWaitMs: number;
  /**
   * 適格外の重みを CPU で f32 展開した時間の総和（f16 / i8 / i4 の decode）。
   * **適格判定に全部通っていれば 0** — 0 でないことは VRAM 削減が落ちている
   * （{@link StorageDiagnostics.hostExpandedBytes} が正）ことと表裏。
   */
  readonly decodeMs: number;
  /** `createBuffer`（重み本体 + scale）に費やした時間の総和。 */
  readonly bufferCreateMs: number;
  /**
   * `queue.writeBuffer` の**発行**に費やした時間の総和。呼び出しは staging への複製を伴うが
   * GPU 転送の完了は待たないので、これは「ホスト側の複製費用」であって転送時間ではない
   * （上の「帰属の限界」）。
   */
  readonly writeBufferIssueMs: number;
  /**
   * `queue.writeBuffer` で実際に渡したバイト数の総和（重み本体 + scale・整列のゼロ詰め込み）。
   * {@link StorageDiagnostics} の 2 席の和とは一致しない — あちらは GPU が抱えるバイト数で、
   * こちらは**ホストから流したバイト数**（適格外の重みは展開後の f32 を流すので後者が膨らむ）。
   */
  readonly uploadedBytes: number;
  /**
   * 供給単位（part）ごとの明示 submit の完了を待った時間の総和（ADR 0108 決定 9 のフェンス）。
   * 実 GPU 転送時間はここに吸われている（上の「帰属の限界」）。
   */
  readonly uploadFenceMs: number;
};

/**
 * params バッファの内容アドレスキャッシュの実績（1 run ぶん）。params は実行時のテンソル値に
 * 依存しないので、shape が変わらない限り 2 run 目以降の `allocCount` は 0 に落ちる。
 *
 * MUST: 常設診断として出す。キャッシュが外れても値は正しいまま（毎 dispatch 確保に戻るだけ）
 * で、例外も警告も出ない — ここが唯一の観測点。
 *
 * NOTE: 導出済み計画がヒットした run（{@link PreparedPlanStats}）では**導出相そのものが
 * 走らない**ため `allocCount` / `reuseCount` とも 0 になる。値の意味は変わらず、「その run が
 * params に対して行った GPU 操作がゼロ」という事実の報告。
 */
export type ParamsCacheStats = {
  /** この run で新規に確保 + writeBuffer した params の本数。 */
  readonly allocCount: number;
  /** この run でキャッシュから配り直した params の本数（GPU 操作ゼロ）。 */
  readonly reuseCount: number;
};

/**
 * 導出済み実行計画（Session 常駐）の実績。同一 bindings で走り直す run は計画・融合判定・
 * レシピ導出を丸ごと飛ばし、レシピ列をそのまま実行相へ渡す。
 *
 * MUST: 常設診断として出す。キャッシュが外れても値は正しいまま（毎 run 導出に戻るだけ）で、
 * 例外も警告も出ない — 性能だけが静かに戻る。ここが唯一の観測点。
 */
export type PreparedPlanStats = {
  /** この run が導出済み計画に当たったか。 */
  readonly hit: boolean;
  /** この run の決着時点で Session が抱えている導出済み計画の本数（上限あり）。 */
  readonly cachedPlans: number;
};

/**
 * 中間バッファの GPU backing（Session 常駐の領域バッファ群 — ADR 0093）の実績。導出済み計画に
 * ヒットした run は中間バッファをここから配るので、run ごとの領域確保・createBuffer / destroy が
 * ゼロになる。
 *
 * MUST: 常設診断として出す。**signature が交互に切り替わる形では毎 run 作り直しになり**、値は
 * 正しいまま run ごとに数百 MiB の createBuffer / destroy が復活する（例外も警告も出ない）。
 * `buildCount` が run 数に比例して伸びていないことが、その沈黙劣化の唯一の観測点。
 */
export type PlanBackingStats = {
  /**
   * 保持中の backing 全てが常駐させている**領域の総和**（未構築 / 破棄済みなら 0）。
   * MUST: 定義は「計画の領域の総和」— backing が併せて常駐させる入力バッファは含めない
   * （理由と門は {@link ActiveBacking.bytes}）。{@link SessionOptions.planBackingBudgetBytes} が
   * 勘定するのも同じ量。
   */
  readonly residentBytes: number;
  /**
   * 保持中の backing が所有する入力バッファの総和（常駐入力は所有しないので含めない）。
   * 予算（{@link SessionOptions.planBackingBudgetBytes}）が勘定するのは `residentBytes + inputBytes`。
   */
  readonly inputBytes: number;
  /** 保持中の backing の本数（予算内で複数保持する — {@link SessionOptions.planBackingBudgetBytes}）。 */
  readonly retainedCount: number;
  /** Session の生存中に backing を構築した累計回数（run ごとではなく累計）。 */
  readonly buildCount: number;
};

/**
 * {@link SessionOptions.planBackingBudgetBytes} の既定（256 MiB）。gemma4 E2B では decode 形
 * （3 MiB）と prefill バケット 32 / 64 / 128 形（capacity 16K で 23 / 45 / 89 MiB）が収まり、
 * chunk 768 形（capacity 16K で ≈ 500 MiB）は 1 本だけ持つ側に落ちる。
 */
export const DEFAULT_PLAN_BACKING_BUDGET_BYTES = 256 * 1024 * 1024;

/**
 * state backing（{@link GenerationContext} 所有バッファ群）の実績。ADR 0066 決定 5 が
 * {@link PlanBackingStats} と同格で置くよう定めた診断席で、KV の常駐量と焼き直しの沈黙劣化に
 * 対する唯一の観測点になる。
 *
 * MUST: 常設診断として出す。**context の識別子は計画鍵に入らない**（決定 5 の MUST）ので、
 * context を切り替えてもレシピ再導出は起きない — その代わり「state を含む bind group の焼き直し」
 * だけが起きる形になっており、そこが暴走しても例外も警告も出ない。
 */
export type StateBackingStats = {
  /**
   * 生存中の context が常駐させている総バイト数（未生成 / 全て dispose 済みなら 0）。
   * MUST: 定義は「state スロットの総バイト数 + 論理長 uniform」= context が実際に GPU 上で
   * 抱えるバイト数（{@link StorageDiagnostics.residentCompressedBytes} が scale ぶんを数えるのと
   * 同じ規律）。生存集合から毎回導出し、独立に更新するカウンタは持たない。
   */
  readonly residentBytes: number;
  /** Session の生存中に生成した context の累計本数（現存数ではなく累計）。 */
  readonly contextCount: number;
  /**
   * state を含む bind group を焼き直した累計回数（ADR 0066 決定 5 の焼き込み単位の分離）。
   *
   * 焼き直しが起きるのは **(context, backing 実体) の組が変わったとき**だけ — 初回・context の
   * 切替・backing の再構築（別 signature への切替 / LRU 追い出し / 構築失敗からの復帰）。
   * MUST: **run 数に比例して伸びていないこと**が、決定 5 の分離が効いていることの唯一の観測点。
   * 比例して伸びる形は「切替のたびに全部組み直す」状態そのもので、値は正しいまま decode の
   * ホットパスに createBindGroup が戻る（例外も警告も出ない）。
   */
  readonly rebindCount: number;
};

export type SessionDiagnostics = {
  /**
   * **この Session が使った**パイプラインキーの本数（グラフと opt-in の組み合わせに対して
   * 何本のカーネルが立ったか）。
   *
   * MUST: 定義は「使ったキーの本数」であって「この Session が生成させた本数」ではない。
   * パイプラインキャッシュは device 寿命（GpuContext 所有）なので、同一 device の先行 Session が
   * 既に作っていればこの Session は 1 本も生成しない — 生成側の定義にすると、同じグラフの
   * Session が「1 本目は 3・2 本目は 0」と報告する形になり、カーネル本数の観測点として使えない。
   * NOTE: パイプラインは初回 run で作られるので、構築直後は 0（{@link SessionBuildStats}）。
   */
  readonly pipelineCount: number;
  /**
   * この device 上のキャッシュが抱えるパイプラインキーの総本数（全 Session の和集合）。
   *
   * dispose 済み Session が使ったキーもここには残る（キャッシュの寿命は GpuContext と一致し、
   * 解放は `GpuContext.destroy` のみ）。{@link SessionDiagnostics.pipelineCount} との差が
   * 「同一 device の他 Session ぶん」で、run を跨いで単調に増え続ける形は codegen 決定性の破れ
   * （キーに載せるべきでない値が載っている）の唯一の観測点になる。
   */
  readonly devicePipelineCount: number;
  readonly submit: SubmitStats;
  /**
   * 重み（initializer）アリーナの実績。**params キャッシュ（Session 常駐）の実体もここが
   * 所有する**ので、`allocCount` は initializer 本数 + 生成済み params 本数になる。
   */
  readonly weights: ArenaStats;
  /** 低精度格納の適格 / 適格外の内訳（ADR 0006 の常設診断）。 */
  readonly storage: StorageDiagnostics;
  /**
   * 構築相の費用内訳（Session の寿命を通じて不変 — 構築が終わった時点で確定する）。
   * 帰属の限界は {@link SessionBuildStats} の docstring。
   */
  readonly buildStats: SessionBuildStats;
  /**
   * 直近 run の中間バッファ実績。未実行なら undefined。
   *
   * NOTE: slot backing に乗った run（{@link PlanBackingStats}）では中間バッファも入力バッファも
   * アリーナを通らないため、ここに残るのは readback staging のぶんだけになり、計画から写す
   * 3 欄（`reuseCount` / `transientBytes` / `peakTransientBytes`）は 0 になる
   * （値の意味は不変 — 「その run がアリーナで確保したもの」と「その run の計画」）。
   */
  readonly lastRun: ArenaStats | undefined;
  /**
   * 直近 run の **op 別 GPU 実時間内訳**（パイプラインキー別 — ADR 0021）。
   * 計測が無効な device（`acquireGpu` の `gpuTiming` / feature 不在）では undefined。
   * `lastRun` と同じ寿命で、run の開始でリセットされる。
   */
  readonly lastRunTiming: GpuTimingStats | undefined;
  /**
   * 直近 run の**計画時**に適用が決まった融合 / 別名化の回数（ルール別 —
   * src/runtime/fusion.ts）。未実行なら undefined で、run のたびに丸ごと置き換わる。
   *
   * MUST: 常設診断として出す。融合はエクスポータのノード発行順が 1 つ変わるだけで黙って
   * 外れ、値は正しいまま性能だけが戻る（例外も警告も出ない）。ここが唯一の観測点。
   */
  readonly lastRunFusions: FusionCounts | undefined;
  /**
   * 直近 run の params キャッシュ実績。未実行なら undefined で、run のたびに置き換わる。
   * 実体（バッファ）は Session 常駐なので、ここは「その run が何本作り、何本使い回したか」。
   */
  readonly lastRunParams: ParamsCacheStats | undefined;
  /**
   * 直近 run の導出済み計画キャッシュ実績。run の開始でリセットされ、導出相が決着した時点で
   * 埋まる（未実行、および導出相の途中で落ちた run では undefined）。
   */
  readonly lastRunPrepared: PreparedPlanStats | undefined;
  /**
   * transient slot の GPU backing の実績（run ごとではなく Session の現況 + 累計）。
   * 未構築の Session では `{ residentBytes: 0, retainedCount: 0, buildCount: 0 }`。
   */
  readonly planBacking: PlanBackingStats;
  /**
   * state backing（{@link GenerationContext} 所有）の実績（Session の現況 + 累計 —
   * ADR 0066 決定 5）。context を 1 本も作っていない Session では
   * `{ residentBytes: 0, contextCount: 0, rebindCount: 0 }`。
   */
  readonly stateBacking: StateBackingStats;
};
