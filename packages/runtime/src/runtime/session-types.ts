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
import type { SymbolBindings } from "./plan.ts";

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
export type RunInputs = Readonly<Record<string, RunInput>>;
export type RunOutputs = Readonly<Record<string, Tensor>>;

/** {@link Session.enqueue} の指定。 */
export type EnqueueOptions = {
  /** 束ねる区間（{@link GpuContext.beginBatch}）。フェンスはこの区間の決着 1 本だけ。 */
  readonly batch: BatchScope;
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
   */
  readonly chunkLength: number;
};

/**
 * 整数内積の変種（w8a8 経路）。**両者は同じ整数を返す**ので、これは速度の選択でしかない
 * （src/kernels/linear-i8a8.ts）。
 */
export type I8a8Dot = "dp4a" | "emu";

/**
 * **テスト専用の非公開面**（mod.ts からは輸出しない — ADR 0008 の「薄い面」を汚さない）。
 *
 * i8a8 の整数内積変種を強制する（linear / 融合 attention の**全 i8a8 カーネル共通** —
 * 選択は device の言語機能の列挙 1 つで決まるので Session 単位で 1 つ）。拡張のある機で
 * `dot4I8Packed` 版とエミュ版を実走して atol=0 で突合するのが「エミュは数値同一」という
 * 主張の唯一の機械的検出器で、環境変数ではなく Session 単位のノブにしてあるのは
 * 1 プロセス内で両方を回すため。
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
 * （累積は f32）、`"i8a8"` は活性を per-token i8 へ量子化して整数内積で回す変種を選ぶ。
 *
 * MUST: 3 値は**相互排他**（直積ではない）。attention の q/k/v は全て活性で格納軸を持たない
 * ので、「f16 かつ i8a8」という組み合わせは表現する対象がそもそも存在しない。
 * MUST: `"f16"` は `acquireGpu({ shaderF16: true })` を伴う（Session 構築時に fail loudly）。
 * **`"i8a8"` は `shader-f16` を要求しない**（feature ゲートに混ぜないこと）。
 */
export type ComputePrecision = "f32" | "f16" | "i8a8";

export type SessionOptions = {
  /** submit の時間予算政策（TDR / watchdog 対策 — ADR 0004）。既定は DEFAULT_SUBMIT_POLICY。 */
  readonly submitPolicy?: SubmitPolicy;
  /**
   * linear の実行形（既定 `"f32"` = 従来どおり）。
   *
   * `"i8a8"` は **i8 で GPU 常駐している重みの linear** だけに効き、活性を per-token i8 へ
   * 量子化して整数内積で回す（設計 = docs/research/2026-08-03-dp4a-w8a8-design.md）。
   * `"f16"` は共有タイルを f16 に落とす計算変種（ADR 0028）で、重み格納が f32 / f16 の
   * linear に効く（**i8 常駐の重みとは組めない** — w8a16 は未実装なので fail loudly）。
   * MUST: 既定は `"f32"` — i8 / f16 資産を自動で低精度実行にすると既存の PNG sha256 門と
   * E2E tolerance が黙って変わる。opt-in 以外はあり得ない。
   */
  readonly linearCompute?: "f32" | "i8a8" | "f16";
  /**
   * 融合 attention（ADR 0023 の 3 カーネル）の実行形（既定 `"f32"` = 従来どおり）。
   *
   * `"f16"` は ①QK / ②行統計 / ③PV の共有タイルを f16 にし、**S も f16 で受け渡す**
   * （① が書き ②③ が読む — transient が半減する）。`linearCompute` と**別の軸**なのは、
   * 1024px の内訳が attention 46% / linear 42% で片方だけ f16 にしたい場面が実際にあるため。
   *
   * `"i8a8"` は q / k / v を i8 へ量子化して整数内積で回す変種
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
   * 保ったまま `attentionCompute: "i8a8"` と組める。本命はこの組）。丸めは格納の 1 回だけで、
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
 * transient slot の GPU backing（Session 常駐バッファ群）の実績。導出済み計画にヒットした run は
 * 中間バッファをここから配るので、アリーナの確保・参照計数・createBuffer / destroy がゼロになる。
 *
 * MUST: 常設診断として出す。**signature が交互に切り替わる形では毎 run 作り直しになり**、値は
 * 正しいまま run ごとに数百 MiB の createBuffer / destroy が復活する（例外も警告も出ない）。
 * `buildCount` が run 数に比例して伸びていないことが、その沈黙劣化の唯一の観測点。
 */
export type PlanBackingStats = {
  /**
   * 活性 backing が常駐させている **slot の**総バイト数（未構築 / 破棄済みなら 0）。
   * MUST: 定義は「slot 表の総バイト数」— backing が併せて常駐させる入力バッファは含めない
   * （理由と門は {@link ActiveBacking.bytes}）。
   */
  readonly residentBytes: number;
  /** Session の生存中に backing を構築した累計回数（run ごとではなく累計）。 */
  readonly buildCount: number;
};

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
   * state を含む bind group を焼き直した累計回数。
   *
   * NOTE: 波 C の時点では **0 固定**（state を束ねる dispatch がまだ無い — ノードの `states` 欄 /
   * `state_append` / attention 統合は波 D）。波 D で ADR 0066 決定 5 の「焼き込み単位の分離」を
   * 実装した時点でここが埋まる。
   */
  readonly rebindCount: number;
};

export type SessionDiagnostics = {
  readonly pipelineCount: number;
  readonly submit: SubmitStats;
  /**
   * 重み（initializer）アリーナの実績。**params キャッシュ（Session 常駐）の実体もここが
   * 所有する**ので、`allocCount` は initializer 本数 + 生成済み params 本数になる。
   */
  readonly weights: ArenaStats;
  /** 低精度格納の適格 / 適格外の内訳（ADR 0006 の常設診断）。 */
  readonly storage: StorageDiagnostics;
  /**
   * 直近 run の中間バッファ実績。未実行なら undefined。
   *
   * NOTE: slot backing に乗った run（{@link PlanBackingStats}）では中間バッファも入力バッファも
   * アリーナを通らないため、ここに残るのは readback staging のぶんだけになる
   * （値の意味は不変 — 「その run がアリーナで確保したもの」）。
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
   * 未構築の Session では `{ residentBytes: 0, buildCount: 0 }`。
   */
  readonly planBacking: PlanBackingStats;
  /**
   * state backing（{@link GenerationContext} 所有）の実績（Session の現況 + 累計 —
   * ADR 0066 決定 5）。context を 1 本も作っていない Session では
   * `{ residentBytes: 0, contextCount: 0, rebindCount: 0 }`。
   */
  readonly stateBacking: StateBackingStats;
};
