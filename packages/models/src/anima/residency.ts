/**
 * Anima の DiT 常駐（ADR 0112）— 決定の純関数と、常駐の席 1 本の状態機械。
 *
 * 既定（`"per-stage"`）は従来どおり段ごとに Session を張っては畳む。opt-in の `"transformer"` は
 * DiT の Session を generate を跨いで持ち続け、2 回目以降の generate で重みの読み直し・導出・
 * backing の作り直しを消す（ADR 0112 決定 1）。
 *
 * ここに置くのは「いつ持ち、いつ手放し、いつ退避するか」だけで、GPU に触る処理は注入された
 * 口（`open` / `dispose` / `settleRelease`）越しに呼ぶ。GPU 無しの単体テストで状態遷移を
 * 故障注入つきで縛るため（`tests/anima_residency_test.ts`）。
 *
 * ## 状態
 *
 * | 欄           | 意味                                                                                   |
 * | ------------ | -------------------------------------------------------------------------------------- |
 * | `held`       | 常駐中の DiT（無ければ undefined）                                                     |
 * | `downgraded` | 退避したことがある（OOM / 空き不足 — 以後この pipeline の寿命の間は常駐しない: 格下げ） |
 *
 * MUST: 格下げは戻さない。戻すと、VRAM が足りない機で毎回 OOM を踏み直してから（または空きを
 * 試し確保してから）遅い経路へ落ちる（退避の費用を generate ごとに払う）。
 * MUST: 常駐しない判断は必ず名乗る（`residency` イベント）。黙って遅い経路へ落ちると、利用者は
 * opt-in が効いていないことに気付けない。例外は既定の段ごと運転で常駐と無関係な generate
 * （既定の挙動にイベントを足さない — 既存の購読側のイベント列を変えない）と、この generate で作った
 * DiT を段の失敗で畳むとき（席に載る前の失敗で、常駐の状態は何も変わらない — 次の generate は
 * また常駐を試みるので遅い経路へ落ちてもいない）だけ。
 *
 * ## 退避の 2 本の線
 *
 * 1. **先回り**（{@link TransformerResidency.ensureHeadroom}）: 常駐 DiT がある時点で、次に張る段が要る量を
 *    試し確保で量り、入らなければ段を張る前に退避する。量る点は generate の中に 2 つ:
 *    - text 段の前（常駐 DiT を持ち越した generate）— text_encoder / text_conditioner の必要量の最大
 *      （{@link generateMemoryNeed}）。
 *    - DiT 段の後・VAE 段の前（その時点で常駐 DiT があるとき）— vae_decoder の必要量。この点が要るのは、
 *      最初の generate（DiT をこの generate で作って席に載せ、その上に VAE が乗る）と、持ち越した DiT が
 *      新しい解像度を回した generate（DiT 段の中で計画の backing が育つので、text 段の前に量った空きは
 *      VAE の時点では古い）を覆うため。VAE を text 段の前の量りに含めないのは、遅く退避するほうが得だから
 *      （DiT 段は常駐の恩恵を受け終えている）。
 * 2. **反応**（{@link TransformerResidency.runStage} / `runTransformer`）: 段が `GpuOutOfMemoryError` を
 *    投げたら退避してやり直す（2 本目の線 — 先回りの見積りが外れたとき）。
 *
 * どちらの線の退避も同じ格下げを立てる（以後この pipeline は常駐しない — 上の「格下げは戻さない」MUST は
 * 先回りの退避にもそのまま効く）。
 *
 * WHY 先回りが主線か: WebGPU で device を失わずに OOM を返すのは `createBuffer` / `createTexture` だけで、
 * 重みのアップロード（`queue.writeBuffer`）の staging の OOM は device を無効化する（wgpu / Dawn —
 * runtime の `gpu/headroom.ts` 冒頭）。反応の線はその OOM を踏んだ時点で手遅れになりうる。
 */

import { type AdmissionReport, GpuOutOfMemoryError } from "@karume/runtime";
import { ModelInputError } from "../errors.ts";

/**
 * 受理する綴りの並び（語彙の正本 — 型はここから導く）。
 *
 * NOTE: 公開面には型だけを出し、この並びは出さない（`ANIMA_SAMPLER_TYPES` とは非対称）— 公開面は薄く
 * 保つ（ADR 0008）ので、選択肢を列挙する消費者（CLI / UI）が現れてから同じ凍結した並びとして足す。
 */
const ANIMA_RESIDENCIES = Object.freeze(["per-stage", "transformer"] as const);

/**
 * DiT の Session を generate の後に持ち続けるか。
 *
 * - `"per-stage"`（既定）: 段ごとに張って畳む。VRAM のピークは最大の段 1 本ぶん。
 * - `"transformer"`: DiT を**この generate の後も**持ち続ける。text / VAE の段は常駐 DiT の上に乗る。
 */
export type AnimaResidency = typeof ANIMA_RESIDENCIES[number];

/** 省略時の値（従来の挙動 — 既定の VRAM の前提を変えない）。 */
export const DEFAULT_ANIMA_RESIDENCY: AnimaResidency = "per-stage";

/** `residency` イベントの動作。 */
export type AnimaResidencyAction =
  /** DiT をこの generate の後も持ち続ける。 */
  | "retained"
  /**
   * DiT を手放した（常駐を求められなかった — DiT 段の終わりか、DiT 段の前で失敗した generate の後 /
   * 求められたが格下げ済み）。
   */
  | "released"
  /** 常駐中の DiT を途中で捨てた（OOM で退避した / 持ち越した DiT の段が失敗した）。 */
  | "evicted";

/** `residency` イベントの理由。 */
export type AnimaResidencyReason =
  /** 実効 residency（request ?? pipeline 既定）どおり。 */
  | "request"
  /** 常駐を求められたが、この pipeline は退避（`out-of-memory` / `headroom`）で格下げ済み。 */
  | "downgraded"
  /** 他の段か、持ち越した DiT の run が `GpuOutOfMemoryError` を投げたので退避した。 */
  | "out-of-memory"
  /**
   * 段を張る前の試し確保で、次の段が要る量が常駐 DiT の上に入らなかったので先に退避した（OOM を踏む前 —
   * {@link TransformerResidency.ensureHeadroom}）。量る点は text 段の前（text 系 2 段の必要量）と、DiT 段の
   * 後・VAE 段の前（VAE の必要量）の 2 つ。
   */
  | "headroom"
  /**
   * 持ち越した常駐 DiT を使った DiT 段が失敗した（denoise ループ内の `onEvent` の throw による中断を
   * 含む）ので、壊れうる Session を捨てた。
   */
  | "failure";

/** 状態機械が発する通知（pipeline が `residency` イベントへ写す）。 */
export type ResidencyNotice = {
  readonly action: AnimaResidencyAction;
  readonly reason: AnimaResidencyReason;
};

/**
 * `residency` の綴りを検査する（構築オプションと request の両方の入口）。
 *
 * MUST: 未知の綴りは既定へ縮退させず落とす — 綴り違いの `"transformers"` が黙って段ごと運転に
 * なると、利用者は常駐が効いていないことに速度でしか気付けない。打つ手は呼び手の側にあるので
 * 入力起因（ADR 0107）。
 */
export const assertAnimaResidency = (value: unknown, where: string): AnimaResidency => {
  const accepted = typeof value === "string"
    ? ANIMA_RESIDENCIES.find((candidate) => candidate === value)
    : undefined;
  if (accepted === undefined) {
    throw new ModelInputError(
      `${where}: 期待 ${ANIMA_RESIDENCIES.map((name) => `'${name}'`).join(" / ")}` +
        `（実際 ${typeof value === "string" ? `'${value}'` : String(value)}）`,
    );
  }
  return accepted;
};

/** DiT 段の Session の出所。 */
export type TransformerSource =
  /** 前の generate から持ち越した常駐 DiT を使う（読み直さない）。 */
  | "resident"
  /** 新しく作り、段が成功したら常駐の席に載せる（席に載るのは段の後 — 段の中の失敗は段ごと運転と同じ）。 */
  | "build-resident"
  /** 段ごとに作って畳む（常駐の席に載せない）。 */
  | "per-stage";

/** 決定の入力（DiT 段の**開始時**の事実）。 */
export type ResidencyFacts = {
  /** この generate の実効 residency（request ?? pipeline 既定）。 */
  readonly effective: AnimaResidency;
  /** 退避（OOM / 空き不足）で格下げ済みか。 */
  readonly downgraded: boolean;
  /** 常駐の席に DiT が載っているか（= 前の generate から持ち越した DiT がある）。 */
  readonly held: boolean;
};

/**
 * DiT 段の Session をどこから取るか。
 *
 * MUST: 常駐 DiT が既にあれば、実効値に関わらずそれを使う（読み直さない — 実効値の意味は「この
 * generate の**後に**持つか」であって「この generate で使うか」ではない）。
 */
export const transformerSource = (facts: ResidencyFacts): TransformerSource => {
  if (facts.held) return "resident";
  return facts.effective === "transformer" && !facts.downgraded ? "build-resident" : "per-stage";
};

/** DiT 段の結末（持ち越した DiT の OOM は別の経路 — {@link TransformerResidency} が退避してやり直す）。 */
export type TransformerOutcome = "success" | "failure";

/** 席の DiT をどうするかの決定。 */
export type TransformerDecision = {
  /**
   * `retain` = 席に載せて持つ / `release` = 手放して畳む / `evict` = 途中で捨てて畳む（GPU 側の扱いは
   * `release` と同じで、差は名乗る理由だけ）。
   */
  readonly action: "retain" | "release" | "evict";
  /** 発するイベント（既定の段ごと運転で常駐と無関係なら無し）。 */
  readonly notice?: ResidencyNotice;
};

/**
 * DiT 段を抜けるときに、段で使った DiT をどうするかを決める（`facts` は段の**開始時**の事実 —
 * 出所は {@link transformerSource} で導く）。状態機械はこの `action` をそのまま適用する（真実はここ 1 か所）。
 *
 * - 失敗: 持ち越した DiT（`resident`）は捨てて名乗る（壊れた Session を次の generate へ持ち越さない —
 *   OOM や device 消失の後の Session は使えるとは限らない）。この generate で作った DiT は段ごと運転の
 *   失敗と同じで、畳むだけで名乗らない（席に載る前の失敗 — 利用者から見て常駐は何も変わっていない）。
 * - 成功: 実効値 `transformer` で格下げでなければ持つ。持たないときは理由を名乗る。常駐と無関係な
 *   段ごと運転だけは名乗らない（既定のイベント列を変えない）。
 *
 * MUST: 到達しない組は既定値へ落とさず投げる（黙って「名乗らずに手放す」を返すと、常駐しない判断は
 * 必ず名乗る MUST と食い違う値が残る）。席に DiT があるまま格下げ済みにはならない — 格下げは退避で
 * 席を空けるときにだけ立ち、格下げ後は席に載せない。
 */
export const decideAfterTransformer = (
  facts: ResidencyFacts,
  outcome: TransformerOutcome,
): TransformerDecision => {
  if (facts.held && facts.downgraded) {
    throw new Error(
      "anima: 常駐の席に DiT があるのに格下げ済み（到達しない状態 — 状態機械の破れ）",
    );
  }
  const source = transformerSource(facts);
  if (outcome === "failure") {
    return source === "resident"
      ? { action: "evict", notice: { action: "evicted", reason: "failure" } }
      : { action: "release" };
  }
  switch (source) {
    case "build-resident":
      return { action: "retain", notice: { action: "retained", reason: "request" } };
    case "resident":
      // 実効値 per-stage で持ち越した DiT を使ったら、手放したことを名乗る（連続生成の最後の 1 枚を
      // per-stage にして解放する使い方の観測点）。
      return facts.effective === "transformer"
        ? { action: "retain", notice: { action: "retained", reason: "request" } }
        : { action: "release", notice: { action: "released", reason: "request" } };
    case "per-stage":
      // 席が空で実効値 transformer なのに段ごと運転 = 格下げ済み（transformerSource の対偶）。
      return facts.effective === "transformer"
        ? { action: "release", notice: { action: "released", reason: "downgraded" } }
        : { action: "release" };
  }
};

/**
 * generate が DiT 段の外で失敗したとき、席の DiT をどうするかを決める（席に DiT があるときだけ呼ぶ）。
 *
 * - 実効値 `per-stage`: 手放して `released` / `request` を名乗る — 「この generate の後に持たない」指示は
 *   generate が失敗しても効かせる（効かせないと、手放したつもりの VRAM が次の generate か dispose まで残る）。
 * - 実効値 `transformer`: 持ったまま名乗らない（DiT 段を抜けていれば `retained` は名乗り済み、DiT 段の前なら
 *   常駐の状態は何も変わっていない）。
 */
export const decideAfterFailedGenerate = (effective: AnimaResidency): TransformerDecision =>
  effective === "per-stage"
    ? { action: "release", notice: { action: "released", reason: "request" } }
    : { action: "retain" };

/**
 * 空きの試し確保に足す余裕（512 MiB）。
 *
 * WHY: 見積り（`AdmissionReport.peakAccountedBytes`）+ 最大 part の staging は「勘定に入れた分」で、
 * 実際の確保はそれを超えうる — ①割り当て器（gpu-allocator）のブロックは 256 MiB で、ブロック未満の
 * 確保が新しいブロックを切ると試し確保が見た量より最大 256 MiB 多く取られる ②wgpu は submit / poll の
 * 後に予算の 99% を超えると device を失う（確保の OOM 線は 97%）— この差が 9.93 GiB の機（Arc B570）で
 * 約 178 MiB ③見積りの `unaccounted`（params・量子化の一時など）は勘定の外。①+② を覆い ③ に少し残す
 * 値として 512 MiB を置く（常駐の既定席 1024² で常駐ぶん +2,646 MiB — ADR 0112 — に比べて小さい）。
 */
export const HEADROOM_MARGIN_BYTES = 512 * 1024 * 1024;

/**
 * 段 1 本が常駐 DiT の上に要るバイト数 = 見積りのピーク + 最大 part の staging + 余裕。
 *
 * - `report.peakAccountedBytes`: 重み + 保持集合の上限（`ModelComponent.estimate` — 束縛と device の
 *   上限はその段の実引数と同じもの）。
 * - `maxPartBytes`: 構築中に同時に残る staging（`ModelComponent.maxPartBytes` の WHY）。
 * - {@link HEADROOM_MARGIN_BYTES}: 勘定の外の上乗せ。
 *
 * NOTE: `export` は GPU 無しで算術を縛るテストのため（公開面には出さない）。
 */
export const stageMemoryNeed = (
  report: Pick<AdmissionReport, "peakAccountedBytes">,
  maxPartBytes: number,
): number => report.peakAccountedBytes + maxPartBytes + HEADROOM_MARGIN_BYTES;

/**
 * 続けて走る段の組（text 段の前の量りでは text_encoder / text_conditioner）が常駐 DiT の上に要る
 * バイト数 = 段ごとの必要量の**最大**。
 *
 * WHY 和でなく最大: 段は 1 本ずつ順に走り、各段の Session は次の段を張る前に畳む（pipeline の
 * `withSession` の finally）。同時に居るのは常駐 DiT（text 段の間は一定 — 試し確保の時点で既に載っている）
 * と段 1 本だけなので、最も重い段 1 本が入れば残りも入る。DiT 段を挟むと常駐 DiT の backing が育ちうる
 * ので、VAE 段はこの組に入れず DiT 段の後に別に量る（モジュール doc「退避の 2 本の線」）。
 *
 * NOTE: `export` は GPU 無しで算術を縛るテストのため（公開面には出さない）。
 */
export const generateMemoryNeed = (needs: readonly number[]): number => Math.max(0, ...needs);

/**
 * OOM の退避に当たる失敗か（退避する相手 = 常駐の席の DiT があり、失敗が `GpuOutOfMemoryError`）。
 *
 * MUST: 型で判定する（文言を見ない）— errorScope が捕まえた余力切れだけが退避で直りうる失敗で、
 * validation や device 消失を退避でやり直すと本当の原因が 2 度目の失敗に埋もれる。
 */
export const evictsForMemory = (error: unknown, held: boolean): boolean =>
  held && error instanceof GpuOutOfMemoryError;

/** 状態機械が GPU 側へ頼む口（本番は Session と GpuContext の薄い包み）。 */
export type ResidencyHooks<T> = {
  /** 常駐させていた資源を破棄する。 */
  readonly dispose: (resource: T) => Promise<void> | void;
  /**
   * 退避（OOM / 空き不足）で破棄した後、解放が device に届くのを待つ。Intel / wgpu は `destroy()` の解放が
   * 次の device poll まで遅れる（docs/known-issues.md「Intel Arc B570」節）ので、待たずに
   * やり直すと同じ OOM を踏みうる（待ちの形の根拠と未検証の範囲は pipeline 側の実装の doc）。
   * この待ちは、致命的な OOM で既に無効化された device の消失も表面化させる（pipeline 側の実装が待ちの
   * 前に空の submit を出す — 有効性を検査する呼び出し）ので、死んだ device は次の段より前にここで
   * 消失の例外になる。先回りの退避（{@link TransformerResidency.ensureHeadroom}）にはやり直しが無い —
   * 待ちの後は次の段をそのまま張る。
   */
  readonly settleRelease: () => Promise<void>;
};

/** 通知の受け口（pipeline の `onEvent` へ写す）。 */
export type ResidencyNotify = (notice: ResidencyNotice) => Promise<void>;

/** DiT 段 1 回ぶんの入力。 */
export type TransformerStage<T, R> = {
  readonly effective: AnimaResidency;
  /** DiT の Session を作る。 */
  readonly open: () => Promise<T>;
  /** 段の本体（denoise ループ）。やり直しでは最初から呼び直す。 */
  readonly body: (resource: T) => Promise<R>;
  readonly notify: ResidencyNotify;
};

/**
 * 常駐の席 1 本（DiT）の状態機械。pipeline に 1 つ持ち、`generate` / `dispose` の直列化鎖の
 * 内側からだけ呼ぶ（並行呼び出しを想定しない — 鎖が 1 本ずつに並べる）。
 *
 * 席をどうするかは純関数（{@link decideAfterTransformer} / {@link decideAfterFailedGenerate}）が決め、
 * ここはその `action` を適用して通知を出すだけ。例外は退避（OOM の反応 {@link evictsForMemory} /
 * 空き不足の先回り {@link TransformerResidency.ensureHeadroom}）で、決定ではなく GPU の事実への
 * 反応なのでここに置く。
 */
export class TransformerResidency<T> {
  readonly #hooks: ResidencyHooks<T>;
  #held: T | undefined;
  #downgraded = false;

  constructor(hooks: ResidencyHooks<T>) {
    this.#hooks = hooks;
  }

  /** 常駐中の DiT があるか（診断とテスト用）。 */
  get holding(): boolean {
    return this.#held !== undefined;
  }

  /** 退避（OOM / 空き不足）で格下げ済みか（診断とテスト用）。 */
  get downgraded(): boolean {
    return this.#downgraded;
  }

  /**
   * DiT 以外の段 1 本を回す。常駐 DiT がある状態で段が `GpuOutOfMemoryError` を投げたら、
   * 常駐 DiT を退避してから**段を 1 回だけ**やり直す（`attempt` を呼び直す — Session の構築から）。
   * やり直しの失敗はそのまま投げる（退避する相手はもう居ない）。
   */
  async runStage<R>(attempt: () => Promise<R>, notify: ResidencyNotify): Promise<R> {
    try {
      return await attempt();
    } catch (error) {
      if (!evictsForMemory(error, this.holding)) throw error;
      await this.#evict("out-of-memory", notify, { error });
      return await attempt();
    }
  }

  /**
   * DiT 段を回す（Session の出所は {@link transformerSource}・抜けるときは
   * {@link decideAfterTransformer} の `action` を適用する）。
   *
   * 持ち越した DiT の run が `GpuOutOfMemoryError` を投げたら、それを退避して段ごと運転で
   * **段を最初から 1 回だけ**やり直す。denoise の状態は段の本体が作り直すので、出る値は変わらない
   * （同じ seed・同じ入力・同じグラフ）。
   *
   * この generate で作る DiT は、段が成功して `retain` と決まるまで席に載せない。その run の失敗は
   * OOM でも退避しない（格下げも通知もせず、畳んで元の例外を投げる）。WHY: そのとき pipeline が持つ
   * GPU 資源はこの DiT 1 本だけ（text 段は畳んだ後）で、VRAM の構成は段ごと運転と同じ — 常駐が原因の
   * OOM ではないので、格下げして GB 級を読み直しても同じ構成をもう一度試すだけになる。
   * DiT の構築そのものの OOM も同じく退避しない（退避する相手が居ない）。
   */
  async runTransformer<R>(stage: TransformerStage<T, R>): Promise<R> {
    const facts = this.#facts(stage.effective);
    const carried = this.#held;
    const resource = carried ?? await stage.open();
    let result: R;
    try {
      result = await stage.body(resource);
    } catch (error) {
      if (evictsForMemory(error, carried !== undefined)) {
        await this.#evict("out-of-memory", stage.notify, { error });
        // 退避で席は空き格下げが立ったので、やり直しは段ごと運転になり、そこでは退避しない
        // （退避する相手が居ない = やり直しは構造上 1 回だけ）。
        return await this.runTransformer(stage);
      }
      throw await this.#settleFailure(
        error,
        decideAfterTransformer(facts, "failure"),
        resource,
        stage.notify,
      );
    }
    const decision = decideAfterTransformer(facts, "success");
    await this.#apply(decision.action, resource);
    if (decision.notice !== undefined) await stage.notify(decision.notice);
    return result;
  }

  /**
   * generate が DiT 段の外で失敗したときに呼ぶ（pipeline の `#generate` の失敗経路 — DiT 段の中の失敗は
   * {@link runTransformer} で決着済み）。席に DiT があれば {@link decideAfterFailedGenerate} を適用する。
   * 返り値は投げるべき例外（{@link #settleFailure} の MUST）。
   */
  async releaseIfRequested(
    error: unknown,
    effective: AnimaResidency,
    notify: ResidencyNotify,
  ): Promise<unknown> {
    const held = this.#held;
    if (held === undefined) return error;
    return await this.#settleFailure(error, decideAfterFailedGenerate(effective), held, notify);
  }

  /**
   * 先回りの退避（モジュール doc「退避の 2 本の線」の 1）: 常駐 DiT の上に `need` バイトが入るかを
   * `probe` で量り、入らなければ 格下げ → 常駐 DiT を破棄 → 解放が届くのを待つ → `evicted` /
   * `headroom` を名乗る。pipeline は次の段を張る前（text 段の前 / VAE 段の前）に呼ぶ。
   *
   * - 常駐 DiT が無い: 何もしない（`probe` も呼ばない — 退避する相手が居ないので量る意味が無い）。
   * - `probe` が true: 何もしない（常駐を続ける）。
   * - `probe` が false: 退避する。後始末（破棄・解放待ち）が失敗したら、名乗ってから後始末の失敗を
   *   並べた `AggregateError` を投げる — 先頭に元の失敗が無い点だけが OOM の退避と違う。
   * - `probe` が投げた: そのまま投げる（格下げも破棄もせず、常駐 DiT を持ったまま）。WHY: validation
   *   （バグ）や device 消失は「入らない」ではない — 退避で片付けると本当の原因が埋もれる。席の DiT の
   *   扱いは generate の失敗経路（{@link releaseIfRequested}）が決める。
   */
  async ensureHeadroom(
    need: number,
    probe: (bytes: number) => Promise<boolean>,
    notify: ResidencyNotify,
  ): Promise<void> {
    if (!this.holding) return;
    if (await probe(need)) return;
    await this.#evict("headroom", notify, undefined);
  }

  /**
   * 常駐 DiT を手放す（pipeline の `dispose` から — 鎖の中で GPU を破棄する**前**に呼ぶ。
   * flush-before-destroy）。常駐が無ければ何もしない。
   */
  async dispose(): Promise<void> {
    await this.#release();
  }

  #facts(effective: AnimaResidency): ResidencyFacts {
    return { effective, downgraded: this.#downgraded, held: this.holding };
  }

  /**
   * 決定の `action` を DiT に適用する（`retain` = 席に載せる — 持ち越しなら既に載っている /
   * `release`・`evict` = 席から外して破棄する）。
   *
   * MUST: 席は破棄の**前**に空ける（破棄に失敗した Session を次の generate が拾わない）。
   */
  async #apply(action: TransformerDecision["action"], resource: T): Promise<void> {
    if (action === "retain") {
      this.#held = resource;
      return;
    }
    if (this.#held === resource) this.#held = undefined;
    await this.#hooks.dispose(resource);
  }

  /**
   * 失敗の後始末: 決定を適用してから名乗り、投げるべき例外を返す。
   *
   * MUST: 元の失敗を上書きしない。後始末（破棄・通知）の失敗は `AggregateError` に並べる。
   * 通知の購読側が**元の失敗と同じ例外**を投げ直したとき（`signal.throwIfAborted()` を毎回呼ぶ
   * 中断の書き方）は新しい失敗ではないので並べない — 並べると消費側が
   * `error === signal.reason` で自分の中断を識別できなくなる。
   */
  async #settleFailure(
    error: unknown,
    decision: TransformerDecision,
    resource: T,
    notify: ResidencyNotify,
  ): Promise<unknown> {
    const notice = decision.notice;
    return await withCleanupFailures(error, [
      () => this.#apply(decision.action, resource),
      ...(notice === undefined ? [] : [() => notify(notice)]),
    ]);
  }

  /**
   * 退避（OOM の反応 / 空き不足の先回り — 共有の 1 本）: 格下げ → 常駐 DiT を破棄 → 解放が届くのを
   * 待つ → `evicted` / `reason` を名乗る。`cause` は反応の退避の元の失敗（先回りでは undefined）。
   *
   * MUST: 格下げを最初に立てる（破棄が失敗しても、以後この pipeline が常駐を試みない）。
   * MUST: 席は破棄の**前**に空ける（破棄に失敗した Session を次の generate が拾わない）。
   * MUST: 後始末（破棄・解放待ち）が失敗しても `evicted` を名乗ってから投げる — 格下げは立ち席は
   * 空いたので、名乗らないと利用者は常駐を失ったことに次の generate まで気付けない。そのときの通知の
   * 失敗も並べる（`AggregateError` — 元の失敗があればそれを先頭に置く）。後始末が通れば通知の throw は
   * そのまま投げる（中断の手段）。
   */
  async #evict(
    reason: "out-of-memory" | "headroom",
    notify: ResidencyNotify,
    cause: { readonly error: unknown } | undefined,
  ): Promise<void> {
    this.#downgraded = true;
    // 元の失敗が無い（先回り）ときは「元の失敗と同じ例外の投げ直し」も無い — 一意な印で比べる。
    const original = cause === undefined ? NO_ORIGINAL_FAILURE : cause.error;
    const failures = await cleanupFailures(original, [
      () => this.#release(),
      () => this.#hooks.settleRelease(),
    ]);
    const notice: ResidencyNotice = { action: "evicted", reason };
    if (failures.length === 0) {
      await notify(notice);
      return;
    }
    failures.push(...await cleanupFailures(original, [() => notify(notice)]));
    throw new AggregateError(
      cause === undefined ? failures : [cause.error, ...failures],
      CLEANUP_FAILED,
    );
  }

  async #release(): Promise<void> {
    const held = this.#held;
    this.#held = undefined;
    if (held !== undefined) await this.#hooks.dispose(held);
  }
}

/** 元の失敗が無い退避（先回り）で {@link cleanupFailures} に渡す印（どの例外とも一致しない）。 */
const NO_ORIGINAL_FAILURE: unique symbol = Symbol("元の失敗なし");

/** 後始末の失敗を並べた `AggregateError` の文言（段ごと運転の DiT にも常駐 DiT にも使う）。 */
const CLEANUP_FAILED = "anima: DiT の後始末が失敗した";

/**
 * 後始末を全て走らせ、失敗を集める。元の失敗と同じ例外の投げ直しは並べない
 * （{@link TransformerResidency} の `#settleFailure` の MUST）。
 */
const cleanupFailures = async (
  error: unknown,
  cleanups: readonly (() => Promise<void> | void)[],
): Promise<unknown[]> => {
  const failures: unknown[] = [];
  for (const cleanup of cleanups) {
    try {
      await cleanup();
    } catch (cleanupError) {
      if (cleanupError !== error) failures.push(cleanupError);
    }
  }
  return failures;
};

/** 後始末を全て走らせ、元の失敗を先頭に並べた例外を返す（後始末が全て通れば元の失敗そのもの）。 */
const withCleanupFailures = async (
  error: unknown,
  cleanups: readonly (() => Promise<void> | void)[],
): Promise<unknown> => {
  const failures = await cleanupFailures(error, cleanups);
  return failures.length === 0 ? error : new AggregateError([error, ...failures], CLEANUP_FAILED);
};
