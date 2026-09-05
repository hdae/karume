// 配布形（safetensors の 1 本または shard 列 + __metadata__ 埋め込みのグラフ JSON）の結合検証。
// グラフ単体の規則は ir.ts が済ませている前提で、ここは宣言と実テンソルの突合、および
// ランタイム対応表との突合だけを持つ。

import { groupScaleShape } from "./i4.ts";
import { type IrDtype, type IrGraph, type IrStorageDtype, parseIrGraph } from "./ir.ts";
import {
  declaredByteLength,
  parseSafetensors,
  type SafetensorsDtype,
  type SafetensorsFile,
  type TensorView,
} from "./safetensors.ts";

/** グラフ JSON を載せる __metadata__ のキー。 */
export const IR_METADATA_KEY = "karume_ir";

export class ContainerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContainerError";
  }
}

export type KarumeModel = {
  readonly graph: IrGraph;
  readonly file: SafetensorsFile;
};

/** op ごとの実行可能条件。op 名だけでは dtype と attrs の差が表せない。 */
export type OpSupport = {
  /**
   * 実行できる意味論 dtype の**和**（op ごとに違う — 契約表 src/ops.ts が正本）。
   * スロット別契約の op ではこの和が実際の受理より広いので、入力の突合には使わない。
   */
  readonly dtypes: ReadonlySet<IrDtype>;
  /**
   * **入力スロット別**の受理集合（並びは契約のアリティぶん）。uniform 契約では全スロットが
   * 同じ集合になる。
   *
   * MUST: 入力の突合はこちらで行う。和だけで見ると gather / embedding / masked_fill の
   * スロット取り違え（`gather(index, src)` のように値と添字を逆に渡した形）が
   * 「どちらの dtype も和には入っている」として列挙門を素通りする — 契約検査（plan.ts）まで
   * 落ちて初めて 1 件ずつ止まるので、「非対応は全件列挙」の意図が壊れる。
   */
  readonly slotDtypes: readonly ReadonlySet<IrDtype>[];
  /**
   * **出力 slot 別**に現れうる意味論 dtype（契約表の出力 dtype 写像の値域 — 並びは契約が
   * 宣言する出力数ぶん。ADR 0068 決定 1）。
   *
   * MUST: 入力スロット 0 の受理集合で代用しない。比較（f32 → bool）・bool の `sum`（→ i32）・
   * `where`（bool → f32）は入力と出力の dtype が違うため、スロット 0 で突き合わせると
   * **正しいグラフが列挙門で落ちる**。逆に恒等な op では両者が一致するので、専用の欄を
   * 持たせても既存の判定は変わらない。
   */
  readonly outDtypes: readonly ReadonlySet<IrDtype>[];
  /** 実装済みの attr キー（契約表の attrs スキーマが宣言するキーそのもの）。 */
  readonly attrKeys: ReadonlySet<string>;
};

/** ランタイムが実行できる op（dtype / attrs 込み）と格納 dtype の対応表。 */
export type RuntimeSupport = {
  readonly ops: ReadonlyMap<string, OpSupport>;
  /**
   * 実行できる格納 dtype。
   *
   * NOTE: f16 は「適格な重みスロットなら圧縮のまま GPU 常駐、それ以外はロード時に CPU で
   * f32 展開」（ADR 0018）で、どちらの経路でも実行できるためこの集合に入る。適格判定の
   * 結果は実行の可否ではなく VRAM の効き方を変えるだけなので、ここでは分岐しない
   * （適格 0MB を沈黙させないのは Session の診断の役目 — ADR 0006）。
   */
  readonly storage: ReadonlySet<IrStorageDtype>;
  /**
   * グラフ入力として転送できる意味論 dtype。
   *
   * MUST: op 起点の突合とは別軸で要る。実行器はどのノードも消費しない入力も含めて
   * `graph.inputs` を全件転送するため、転送層の制約は「その入力を使うノードがあるか」と
   * 無関係に実在する。この軸が無いと、未使用入力の dtype 違反だけが列挙門を素通りする。
   */
  readonly io: ReadonlySet<IrDtype>;
};

/**
 * 格納 dtype → safetensors dtype。f32 / f16 / bf16 / i8 / i4 は意味論 f32 の符号化で、
 * `i32` だけが生の int32（ADR 0010 の明示的な例外）。
 *
 * NOTE: この表は「宣言と実テンソルの突合」専用で、実行できるかどうかとは別軸
 * （実行可否の正本は {@link RuntimeSupport.storage} の実値 = src/ops.ts の `RUNTIME_SUPPORT`。
 * ここで数え上げると同じ事実を 2 箇所で持つことになり、増えたときに片方だけ腐る）。
 */
const STORAGE_ENCODING: Readonly<Record<IrStorageDtype, SafetensorsDtype>> = {
  f32: "F32",
  f16: "F16",
  bf16: "BF16",
  i8: "I8",
  // packed 4bit（ADR 0069 決定 2）。shape は論理形のままで、バイト数だけが bit 幅から決まる。
  i4: "I4",
  i32: "I32",
};

/**
 * companion scale の格納 dtype（ADR 0019）。scale を f16 のビット列として読むと全チャネルが
 * 桁違いの値になるため、形の検査（{@link assertScaleTensor}）もバイト数の導出
 * （{@link declaredScaleBytes}）もこの 1 本から引く。
 */
const SCALE_DTYPE: SafetensorsDtype = "F32";

/**
 * 格納 dtype と要素数から決まる payload のバイト長（**宣言由来** — 実テンソルを見ない）。
 *
 * 実バイトとの一致は突合門が保証している: {@link ShardValidator.intake} が dtype と shape を
 * 実テンソルと突き合わせ、safetensors パーサが「実バイト長 = 宣言由来バイト長」を見ているので、
 * 開けたモデルではこの値が現物と厳密に一致する。
 */
export const declaredPayloadBytes = (
  storage: IrStorageDtype,
  count: number,
  where: string,
): number => declaredByteLength(STORAGE_ENCODING[storage], count, where);

/** companion scale の宣言由来バイト長（要素数 → バイト数 — {@link declaredPayloadBytes} の scale 版）。 */
export const declaredScaleBytes = (count: number, where: string): number =>
  declaredByteLength(SCALE_DTYPE, count, where);

export const openModel = (buffer: ArrayBuffer): KarumeModel =>
  openModelFile(parseSafetensors(buffer));

/**
 * 解析済み safetensors 1 本からモデルを開く。
 *
 * MUST: 検査は shard 進行検証（{@link createShardValidator}）の「1 本受けて読了」に載せる —
 * 単一ファイル面と shard 列面で検査を二重実装すると、片方だけに規則が足されて受理集合が
 * 割れる（ADR 0070 決定 1 が「同一集合の shard 横断版」と言っているのはこの一本化のこと）。
 */
const openModelFile = (file: SafetensorsFile): KarumeModel => {
  const graph = extractIrGraph(file);
  const validator = createShardValidator(graph);
  validator.intake(file);
  validator.finish();
  return { graph, file };
};

/**
 * グラフ shard の `__metadata__` から IR を取り出す。
 *
 * shard 列でも「グラフ shard が最初の 1 本」という契約（ADR 0070 決定 3）なので、
 * 「karume_ir があるか」を見るのはこの 1 箇所だけ — 重み shard は metadata を見ない
 * （{@link ShardValidator.intake} が metadata に触らないのはこのため）。
 */
export const extractIrGraph = (file: SafetensorsFile): IrGraph => {
  const json = file.metadata.get(IR_METADATA_KEY);
  if (json === undefined) {
    throw new ContainerError(`__metadata__.${IR_METADATA_KEY} が無い（Karume モデルではない）`);
  }
  return parseIrGraph(json);
};

/**
 * piece（テンソル分割）キーの規則 — `<親名>#NNNNN-of-NNNNN`（5 桁ゼロ詰め・index は 1 始まり・
 * ADR 0090 の読み手契約）。
 *
 * 単独で 1 shard に収まらない大テンソルは**先頭次元（行）の連続範囲**へ割って、連続する
 * shard へ 1 本ずつ配られる。親名が宣言（`initializer.tensor`）に在るときだけ piece と
 * 解釈する MUST — 素のテンソル名に偶然この綴りが現れても、宣言に無ければ従来どおり余剰
 * （どの initializer からも参照されないテンソル）として落ちる。
 */
const PIECE_KEY = /^(.+)#(\d{5})-of-(\d{5})$/;

/** piece キーの分解結果（{@link parsePieceKey}）。 */
export type PieceKey = {
  /** 親テンソルのキー（= `initializer.tensor`）。 */
  readonly name: string;
  /** 1 始まりの位置。 */
  readonly index: number;
  /** 列の総数（2 以上）。 */
  readonly count: number;
};

/**
 * piece キーを親名と位置に分解する（形が違えば undefined = piece ではない）。
 *
 * MUST: `count ≥ 2` と `1 ≤ index ≤ count` をここで見る。範囲外の綴り（`#00003-of-00002` や
 * 1 本しかない piece 列）を通すと、それが進行状態の初期値として入り込んで違反の帰属が
 * 「piece 列の並び」へ移る。実際にはそのキー自体が配布形の誤りなので、piece と解釈せずに
 * 余剰として落とすほうが直す側にとって決定的になる。
 */
export const parsePieceKey = (key: string): PieceKey | undefined => {
  const matched = PIECE_KEY.exec(key);
  if (matched === null) return undefined;
  const index = Number(matched[2]);
  const count = Number(matched[3]);
  if (count < 2 || index < 1 || index > count) return undefined;
  return { name: matched[1], index, count };
};

/** 分割テンソルの 1 本ぶんの位置（{@link ReadyInitializer.piece}）。 */
export type InitializerPiece = {
  /** この piece が始まる行（先頭次元）。行数は `view.shape[0]`。 */
  readonly rowOffset: number;
  /** piece 1（バッファ確保と companion scale の転送を担う席）。 */
  readonly first: boolean;
  /** piece n（末尾整列の詰め物を担う席）。 */
  readonly last: boolean;
};

/** その shard で**実体が確定した** initializer（payload と、あれば companion scale の view）。 */
export type ReadyInitializer = {
  readonly name: string;
  /** payload の実体 view（file 内）。piece 列では**その piece だけ**の view。 */
  readonly view: TensorView;
  /** scale の実体 view（`storage.scale` を持つ initializer のみ・piece 列では piece 1 だけ）。 */
  readonly scale?: TensorView;
  /**
   * 分割テンソルの位置（丸ごと 1 本で来たときは undefined — ADR 0090）。
   *
   * 消費側は `first` でバッファを確保して scale を上げ、`last` でだけ末尾整列の詰め物を掛け、
   * 各 piece を `rowOffset` から決まるバイト位置へ書く（中間 piece に詰め物を掛けると、
   * 詰め物が次の piece の先頭バイトを潰す沈黙誤値になる）。
   */
  readonly piece?: InitializerPiece;
  /** view / scale が指す shard。逐次消費側は転送後にこの参照を手放す（ADR 0070 決定 3）。 */
  readonly file: SafetensorsFile;
};

/** piece 列の進行状態（親テンソル 1 本ぶん — 完了した親は表から消える）。 */
type PieceProgress = {
  readonly count: number;
  /** 次に来るべき piece の index。 */
  readonly nextIndex: number;
  /** 受理済みの行数（= 次の piece の行オフセット）。 */
  readonly rows: number;
};

/** その shard に来た piece 1 本（キーの分解結果 + 実体 view）。 */
type ShardPiece = PieceKey & {
  readonly key: string;
  readonly view: TensorView;
};

/**
 * shard 中の piece キーを親ごとに集める（宣言順に処理するための前処理 — 走査は 1 回）。
 *
 * 親が宣言に無いキーは集めない = 突合集合の外に留まるので、従来どおり余剰として落ちる。
 */
const collectShardPieces = (
  file: SafetensorsFile,
  declaredTensors: ReadonlySet<string>,
): {
  readonly byParent: ReadonlyMap<string, readonly ShardPiece[]>;
  readonly keys: ReadonlySet<string>;
} => {
  const byParent = new Map<string, ShardPiece[]>();
  const keys = new Set<string>();
  for (const [key, view] of file.tensors) {
    const parsed = parsePieceKey(key);
    if (parsed === undefined || !declaredTensors.has(parsed.name)) continue;
    keys.add(key);
    const piece: ShardPiece = { ...parsed, key, view };
    const siblings = byParent.get(parsed.name);
    if (siblings === undefined) byParent.set(parsed.name, [piece]);
    else siblings.push(piece);
  }
  return { byParent, keys };
};

/**
 * companion scale の実体を突き合わせる（実在 = co-shard MUST・dtype・形）。
 *
 * MUST: 丸ごとの実体と piece 1 で**同じ規則**を掛ける。分割された重みでも scale は 1 本きりで、
 * 置き場所は「実体（piece 列なら piece 1）と同じ shard」だけが許される — 逐次消費は weight と
 * scale を同時に要するので、別 shard に置くと「転送したら参照を手放す」契約と両立しない。
 */
const resolveScale = (
  name: string,
  initializer: IrGraph["initializers"][string],
  weightShape: readonly (number | string)[],
  file: SafetensorsFile,
): TensorView | undefined => {
  const scaleKey = initializer.storage.scale;
  if (scaleKey === undefined) return undefined;
  const scale = file.tensors.get(scaleKey);
  if (scale === undefined) {
    throw new ContainerError(
      `initializer '${name}': scale テンソル '${scaleKey}' がファイルに無い（実体 '${initializer.tensor}' と同じ shard に置く MUST — companion scale の co-shard 契約・ADR 0070 決定 1）`,
    );
  }
  // group 形の scale を要求するのは格納 i4 だけ（ADR 0069 決定 3）。i8 に付いた
  // group_size は語彙としては通る（実行できないことは assertRuntimeSupport が列挙する）
  // ので、scale の形の分岐は group_size の有無ではなく**格納 dtype**で決める。
  const groupSize = initializer.storage.dtype === "i4" ? initializer.storage.groupSize : undefined;
  assertScaleTensor(name, scaleKey, scale, weightShape, groupSize);
  return scale;
};

/**
 * shard を 1 本ずつ受けて進行的に検査する器（ADR 0070 決定 1）。
 *
 * 突合集合は `initializer.tensor` と `storage.scale` が指す名前の**和**（+ 親が宣言に在る
 * piece キー — {@link parsePieceKey}）。shard ごとに決まること（余剰・重複・dtype / shape・
 * scale の形・co-shard・piece の並びと行範囲）は {@link intake} が即座に、全 shard 揃って
 * 初めて決まること（欠け・未完の piece 列）は {@link finish} が見る。
 */
export type ShardValidator = {
  /** shard を 1 本受理し、この shard で実体が確定した initializer 群を返す（fail loudly）。 */
  intake(file: SafetensorsFile): readonly ReadyInitializer[];
  /** 全 shard 読了後の完全性検査（欠けを全件列挙して fail loudly）。 */
  finish(): void;
};

export const createShardValidator = (graph: IrGraph): ShardValidator => {
  // グラフ単体で決まる規則は shard を 1 本も見ないうちに落とす（構築時 1 回）。
  assertNoScaleKeyCollision(graph);
  // 突合集合（ADR 0070 決定 1）。scale は IR の値ではないので initializer 集合だけを正本に
  // すると i8 / i4 資産の scale が全て「余剰」になる。
  const declaredNames = new Set<string>();
  // piece キーの親として認める名前（実体キーだけ — scale は分割しない）。
  const declaredTensors = new Set<string>();
  for (const initializer of Object.values(graph.initializers)) {
    declaredNames.add(initializer.tensor);
    declaredTensors.add(initializer.tensor);
    if (initializer.storage.scale !== undefined) declaredNames.add(initializer.storage.scale);
  }
  const seen = new Set<string>();
  /** 進行中の piece 列（キー = `initializer.tensor`）。 */
  const progress = new Map<string, PieceProgress>();

  return {
    intake(file: SafetensorsFile): readonly ReadyInitializer[] {
      // 検査順は「宣言 → 実体」を先、「実体 → 宣言」を後（単一ファイル面の従来順そのまま）。
      // 逆にすると、実体を残したまま宣言だけ改名した形の帰属が余剰へ移り、直す側は
      // 「余っている名前」だけを見せられて改名先が分からなくなる。
      const ready: ReadyInitializer[] = [];
      // piece は親ごとに集めてから宣言順に処理する（走査は 1 回）— 宣言順（`graph.initializers`
      // の並び）を保つのは、消費側の GPU 転送順が shard のヘッダ並びに依存すると同一資産でも
      // 配布形の詰め方でアリーナ配置が変わるため。
      const pieces = collectShardPieces(file, declaredTensors);
      // 進行状態の更新は**全検査を通り抜けた後**にまとめて適用する（`seen` と同じ規律）。
      const advanced: { readonly tensor: string; readonly next: PieceProgress | undefined }[] = [];
      for (const [name, initializer] of Object.entries(graph.initializers)) {
        const where = `initializer '${name}'`;
        // 意味論 dtype と格納 dtype の組（f32 の符号化語彙 / i32 は生の int32）と数値 shape は
        // parseIrGraph が保証済み（グラフ単体で決まる規則はパーサに一本化 — docs/ir-v1.md）。
        // ここは実テンソルとの突合だけを見る。
        const declared = graph.values[name];
        const expected = STORAGE_ENCODING[initializer.storage.dtype];
        const view = file.tensors.get(initializer.tensor);
        const shardPieces = pieces.byParent.get(initializer.tensor);
        const state = progress.get(initializer.tensor);
        if (shardPieces === undefined) {
          // MUST: piece 列の途中でこの親の piece が来ない shard は違反。分割は「連続する shard
          // へ 1 本ずつ」なので、途切れた列は後続 shard でも埋まらない — 欠けとして読了まで
          // 持ち越すと、どの shard から崩れたのかが失われる。
          if (state !== undefined) {
            throw new ContainerError(
              `${where}: テンソル '${initializer.tensor}' の piece 列がこの shard で途切れた` +
                `（${state.nextIndex - 1}/${state.count} まで受理・次は index ${state.nextIndex} ` +
                "が要る — piece は連続する shard に 1 本ずつ置く MUST）",
            );
          }
          // この shard に来ていないだけ（後続 shard で来る）— 欠けの判定は finish の担当。
          if (view === undefined) continue;
          if (view.dtype !== expected) {
            throw new ContainerError(
              `${where}: 格納 dtype '${initializer.storage.dtype}' に対し safetensors 側が ${view.dtype}（${expected} が必要）`,
            );
          }
          if (
            declared.shape.length !== view.shape.length ||
            declared.shape.some((dim, index) => dim !== view.shape[index])
          ) {
            throw new ContainerError(
              `${where}: 宣言 shape [${declared.shape.join(",")}] ≠ 実テンソル [${
                view.shape.join(",")
              }]`,
            );
          }
          ready.push({
            name,
            view,
            scale: resolveScale(name, initializer, declared.shape, file),
            file,
          });
          continue;
        }
        // ここから piece 列。1 テンソルは「丸ごと」か「piece 列」のどちらか一方で、混在は
        // 再定義（どちらのバイトが勝つかが転送順で決まる沈黙誤値）と同じ機序になる。
        if (view !== undefined) {
          throw new ContainerError(
            `${where}: テンソル '${initializer.tensor}' が丸ごとと piece '${
              shardPieces[0].key
            }' の両方でこの shard に在る（1 テンソルは丸ごとか piece 列のどちらか一方）`,
          );
        }
        if (shardPieces.length > 1) {
          const keys = shardPieces.map((piece) => piece.key).sort().join(", ");
          throw new ContainerError(
            `${where}: 同じ shard に piece が ${shardPieces.length} 本ある（${keys}）— piece は連続する shard に 1 本ずつ置く MUST`,
          );
        }
        const piece = shardPieces[0];
        if (state === undefined && seen.has(initializer.tensor)) {
          throw new ContainerError(
            `${where}: テンソル '${initializer.tensor}' は別の shard で実体が確定しているのに piece '${piece.key}' が来た（1 テンソルは丸ごとか piece 列のどちらか一方）`,
          );
        }
        const expectedIndex = state?.nextIndex ?? 1;
        if (piece.index !== expectedIndex) {
          throw new ContainerError(
            `${where}: piece '${piece.key}' の index ${piece.index} が期待 ${expectedIndex} と違う（piece 1..n を shard 順に 1 本ずつ）`,
          );
        }
        if (state !== undefined && piece.count !== state.count) {
          throw new ContainerError(
            `${where}: piece '${piece.key}' の総数 ${piece.count} が先行 piece の ${state.count} と違う`,
          );
        }
        if (piece.view.dtype !== expected) {
          throw new ContainerError(
            `${where}: 格納 dtype '${initializer.storage.dtype}' に対し piece '${piece.key}' が ${piece.view.dtype}（${expected} が必要）`,
          );
        }
        // piece は親の**先頭次元の連続範囲**（dtype は親と同一・残りの次元は宣言と同値）。
        const rows = piece.view.shape[0];
        const sameTail = declared.shape.length === piece.view.shape.length &&
          declared.shape.every((dim, axis) => axis === 0 || dim === piece.view.shape[axis]);
        if (!sameTail || rows === undefined || rows < 1) {
          throw new ContainerError(
            `${where}: piece '${piece.key}' の shape [${piece.view.shape.join(",")}] が宣言 [${
              declared.shape.join(",")
            }] の行範囲でない（先頭次元は 1 以上の行数・残りの次元は宣言と同値）`,
          );
        }
        const declaredRows = Number(declared.shape[0]);
        const rowOffset = state?.rows ?? 0;
        const covered = rowOffset + rows;
        if (covered > declaredRows) {
          throw new ContainerError(
            `${where}: piece '${piece.key}' までの累積行数 ${covered} が宣言 shape の先頭次元 ${declaredRows} を超える`,
          );
        }
        const last = piece.index === piece.count;
        if (last && covered !== declaredRows) {
          throw new ContainerError(
            `${where}: piece 列を読み切っても累積行数が ${covered} 行で宣言 shape の先頭次元 ${declaredRows} 行に届かない`,
          );
        }
        // MUST: 末尾以外の piece はバイト長が 4 の倍数。続きの piece は行オフセット由来の
        // バイト位置へ書かれ、`queue.writeBuffer` の書き込み先オフセットは 4 バイト整列を
        // 要求する（破れると validation で no-op = 重みが欠けたまま走り出す）。末尾 piece
        // だけは任意長でよい（消費側の末尾詰め物が整列を作る）。
        if (!last && piece.view.byteLength % 4 !== 0) {
          throw new ContainerError(
            `${where}: 末尾でない piece '${piece.key}' が ${piece.view.byteLength} バイト（4 の倍数が必要 — 続きの piece が書かれるオフセットの整列条件）`,
          );
        }
        ready.push({
          name,
          view: piece.view,
          // companion scale は piece 1 と同じ shard（co-shard MUST の piece 版）。2 本目以降の
          // shard に scale があれば「別 shard で定義済み」として再定義検査が落とす。
          scale: piece.index === 1
            ? resolveScale(name, initializer, declared.shape, file)
            : undefined,
          piece: { rowOffset, first: piece.index === 1, last },
          file,
        });
        advanced.push({
          tensor: initializer.tensor,
          next: last
            ? undefined
            : { count: piece.count, nextIndex: piece.index + 1, rows: covered },
        });
      }
      // 実体 → 宣言の 3 本。孤立 scale を余剰より先に見るのは帰属の問題（下の doc）。重複は
      // 突合集合に入っている名前でしか起きない（= 余剰と交わらない）ので、順は結果を変えない。
      assertNoOrphanScale(graph, file, pieces.byParent, seen);
      assertNoSurplusTensors(declaredNames, pieces.keys, file);
      assertNoRedefinedTensors(seen, file);
      // 記録は全検査を通り抜けた後（途中で落ちた shard は「見た」ことにしない — 失敗した
      // 構築は部分 Session を公開せず全て捨てる契約 = ADR 0070 決定 3 と同じ境界）。
      for (const name of file.tensors.keys()) seen.add(name);
      for (const step of advanced) {
        if (step.next !== undefined) {
          progress.set(step.tensor, step.next);
          continue;
        }
        progress.delete(step.tensor);
        // 親名を「見た」ことにするのは**最後の piece を受理したとき**だけ（欠け検査の突合先
        // であり、以後の丸ごと再定義を落とす印でもある）。
        seen.add(step.tensor);
      }
      return ready;
    },

    finish(): void {
      // 欠けは**全件列挙**する（1 件ずつ落とすと、配布形を組む側が何本足りないのか分からない）。
      const missing: string[] = [];
      for (const [name, initializer] of Object.entries(graph.initializers)) {
        const where = `initializer '${name}'`;
        const state = progress.get(initializer.tensor);
        if (state !== undefined) {
          // 未完の piece 列も欠けの一種。どこまで来て何行残っているかまで言う（配布形を
          // 組み直す側は、この 2 つでどの piece から作り直すかを決める）。
          missing.push(
            `${where}: テンソル '${initializer.tensor}' の piece 列が未完（piece ${
              state.nextIndex - 1
            }/${state.count} まで受理・残り ${
              Number(graph.values[name].shape[0]) - state.rows
            } 行）`,
          );
        } else if (!seen.has(initializer.tensor)) {
          missing.push(`${where}: テンソル '${initializer.tensor}' がファイルに無い`);
        }
        const scaleKey = initializer.storage.scale;
        if (scaleKey !== undefined && !seen.has(scaleKey)) {
          missing.push(`${where}: scale テンソル '${scaleKey}' がファイルに無い`);
        }
      }
      if (missing.length > 0) {
        throw new ContainerError(
          `宣言に対して不足するテンソル (${missing.length}): ${missing.join(" / ")}`,
        );
      }
    },
  };
};

/**
 * initializer どうしのキーの取り違え 3 種を落とす: scale キーが**どの** initializer の実体キーと
 * も衝突しないこと、scale キーが 2 本の initializer で**共有されていない**こと、そして
 * **実体キーが共有されていない**こと。
 *
 * MUST: 別の initializer の実体を scale として読むと、dtype も shape も偶然合う組で沈黙誤値に
 * なる。共有も同じ機序で、チャネル数（i4 なら行数と group 数）さえ揃えば形検査を両方が通り、
 * 後発の重みが先発の scale で逆量子化される。IR v1 は重み tying を表現する語彙を持たない
 * （`storage.scale` はキー 1 本きり）ので、共有形は取り違えだけを意味する。
 * MUST: 実体キーの共有も同じ理由で落とす（**1 実体 1 initializer**）。エクスポータ側が
 * 1:1 を MUST として発行している規則の読み手側の鏡像で、通すと実行層が initializer 名ごとに
 * 確保・転送するため同じバイト列が 2 度 GPU へ上がり（無診断の VRAM 倍化）、i8 / i4 では
 * **同じ量子化バイトが 2 つの別 scale で逆量子化される**。
 * グラフ単体で決まる規則なので shard を見る前（validator 構築時）に 1 回だけ掛ける —
 * shard ごとに掛けると、衝突相手が別 shard にいる配布形で検出が「たまたま同居したときだけ」に
 * なる。
 */
const assertNoScaleKeyCollision = (graph: IrGraph): void => {
  // 実体キー / scale キーの持ち主を記録しながら 1 走査で両向きを見る（宣言順のどちらが先でも
  // 同じ帰属で落ちる）。どの診断も**相手の initializer 名**を名乗る MUST — 名前が出ないと
  // 直す側はどちらを改名するか決められない。
  const entityOwner = new Map<string, string>();
  const scaleOwner = new Map<string, string>();
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const earlierScale = scaleOwner.get(initializer.tensor);
    if (earlierScale !== undefined) {
      throw new ContainerError(
        `initializer '${earlierScale}': scale テンソル '${initializer.tensor}' が initializer '${name}' の実体と同じキー`,
      );
    }
    const earlierEntity = entityOwner.get(initializer.tensor);
    if (earlierEntity !== undefined) {
      throw new ContainerError(
        `initializer '${name}': 実体テンソル '${initializer.tensor}' が initializer '${earlierEntity}' と共有されている（1 実体 1 initializer MUST）`,
      );
    }
    entityOwner.set(initializer.tensor, name);
    const scaleKey = initializer.storage.scale;
    if (scaleKey === undefined) continue;
    const entity = entityOwner.get(scaleKey);
    if (entity !== undefined) {
      throw new ContainerError(
        `initializer '${name}': scale テンソル '${scaleKey}' が initializer '${entity}' の実体と同じキー`,
      );
    }
    const sharedWith = scaleOwner.get(scaleKey);
    if (sharedWith !== undefined) {
      throw new ContainerError(
        `initializer '${name}': scale テンソル '${scaleKey}' が initializer '${sharedWith}' と共有されている（1 重み 1 scale MUST）`,
      );
    }
    scaleOwner.set(scaleKey, name);
  }
};

/**
 * co-shard 違反のうち「scale だけが来て実体が同じ shard に無い」向きを検出する。
 *
 * MUST: 余剰検査より**先**にこの帰属で言う。scale 名は突合集合に入っているので余剰では
 * 拾えず、実体側の co-shard 検査は「実体が来た shard」でしか回らないため、この順でないと
 * 「余剰でも欠けでもない孤立 scale」が読了まで沈黙する。
 *
 * 実体の同居は「丸ごと」か「piece 1」のどちらかで足りる（piece 2..n の shard に scale を
 * 置く形は、その scale が既に piece 1 の shard で確定済み = 再定義として落ちる — 帰属は
 * 「孤立」より「別 shard で定義済み」のほうが正確）。
 */
const assertNoOrphanScale = (
  graph: IrGraph,
  file: SafetensorsFile,
  pieces: ReadonlyMap<string, readonly ShardPiece[]>,
  seen: ReadonlySet<string>,
): void => {
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const scaleKey = initializer.storage.scale;
    if (scaleKey === undefined) continue;
    if (!file.tensors.has(scaleKey) || seen.has(scaleKey)) continue;
    const firstPiece = pieces.get(initializer.tensor)?.some((piece) => piece.index === 1) === true;
    if (file.tensors.has(initializer.tensor) || firstPiece) continue;
    throw new ContainerError(
      `initializer '${name}': scale テンソル '${scaleKey}' だけが shard にあり実体 '${initializer.tensor}' が無い（companion scale は weight と同一 shard に置く MUST — ADR 0070 決定 1）`,
    );
  }
};

/**
 * shard 中の全テンソルがどこかから参照されていることを検査する（宣言 → 実体の逆向き）。
 *
 * MUST: fail loudly（黙って受理しない）。宣言側の走査だけでは「使われなくなった重みが
 * 配布形に残っている」形が素通りし、配布物が数十 MB〜GB 級であることを踏まえると
 * 「黙って太った配布形」がロード時に検出されないまま公開されうる。
 *
 * NOTE: 余剰は**全件列挙**する（1 件ずつ落とすと、削る側が何本余っているのか分からない —
 * {@link assertRuntimeSupport} と同じ型）。
 *
 * `pieceKeys` は「親が宣言に在る piece キー」= 突合集合の一員。親が宣言に無い piece 綴りは
 * ここに来ないので、従来どおり余剰として落ちる（{@link parsePieceKey} の doc）。
 */
const assertNoSurplusTensors = (
  declaredNames: ReadonlySet<string>,
  pieceKeys: ReadonlySet<string>,
  file: SafetensorsFile,
): void => {
  const surplus = [...file.tensors.keys()]
    .filter((name) => !declaredNames.has(name) && !pieceKeys.has(name))
    .sort();
  if (surplus.length > 0) {
    throw new ContainerError(
      `どの initializer からも参照されないテンソル (${surplus.length}): ${surplus.join(", ")}`,
    );
  }
};

/**
 * 既に別の shard で実体が確定した名前の再登場を検出する。
 *
 * MUST: shard 横断でしか見えない違反。同名テンソルが 2 本の shard にあると「後から来た方が
 * 勝つ / 先に来た方が勝つ」が転送順で決まる沈黙誤値になり、配布形を組み直すまで気づけない。
 */
const assertNoRedefinedTensors = (seen: ReadonlySet<string>, file: SafetensorsFile): void => {
  const redefined = [...file.tensors.keys()].filter((name) => seen.has(name)).sort();
  if (redefined.length > 0) {
    throw new ContainerError(
      `別の shard で既に定義されたテンソル (${redefined.length}): ${redefined.join(", ")}`,
    );
  }
};

/**
 * 量子化格納の scale テンソル（ADR 0019）の**形**を実テンソルと突き合わせる。
 *
 * MUST: scale は IR の値ではなく safetensors の**素のテンソル**なので、宣言完全性の検査
 * （parseIrGraph）が 1 つも掛からない — ここだけが門になる。
 *
 * 1. **F32**（scale を f16 のビット列として読むと全チャネルが桁違いの値になる）
 * 2. 形（`groupSize` の有無で 2 通り — ADR 0069 決定 3）
 *    - per-channel（i8）: 重みと**同 rank の keepdim broadcast 形**（各軸は 1 か重みと同値。
 *      1 軸だけがチャネル軸として残る形 — `torch.amax(..., keepdim=True)` の出力そのもの）
 *    - group（i4）: **rank 非依存の rank 2 形** {@link groupScaleShape}。keepdim broadcast 形
 *      とは受理集合が交わらないので**別分岐**にする（broadcast 形の規則で見ると group 数の
 *      取り違えが「1 でも同値でもない軸」として落ちるだけで、正しい group 形も一緒に落ちる）
 *
 * NOTE: 実在と名前衝突は別の層が持つ — 実在は co-shard 検査（{@link assertNoOrphanScale} と
 * intake の実体側）、衝突は {@link assertNoScaleKeyCollision}。どちらも「形」より前に決まる。
 * 「非 1 の軸が消費側 op のチャネル軸と一致するか」は op を知らないと決まらないので
 * ここでは見ない（GPU 常駐経路の平坦添字が掛かる条件 — src/runtime/executor.ts が見る）。
 */
const assertScaleTensor = (
  name: string,
  scaleKey: string,
  view: TensorView,
  weightShape: readonly (number | string)[],
  /** group 量子化（格納 i4）の group 長。per-channel（i8）では undefined。 */
  groupSize: number | undefined,
): void => {
  const where = `initializer '${name}'`;
  if (view.dtype !== SCALE_DTYPE) {
    throw new ContainerError(
      `${where}: scale テンソル '${scaleKey}' が ${view.dtype}（${SCALE_DTYPE} が必要）`,
    );
  }
  if (groupSize !== undefined) {
    // 行（先頭次元）× group 数ちょうどの rank 2。行長が group で割り切れることは
    // parseIrGraph が保証済み（ADR 0069 決定 2）。
    const expected = groupScaleShape(weightShape, groupSize);
    if (view.shape.length !== expected.length) {
      throw new ContainerError(
        `${where}: scale [${view.shape.join(",")}] の rank が group 形 [${
          expected.join(",")
        }] と違う（group 形は重みの rank に依らず rank 2）`,
      );
    }
    if (view.shape.some((dim, axis) => dim !== expected[axis])) {
      throw new ContainerError(
        `${where}: scale [${view.shape.join(",")}] が重み [${weightShape.join(",")}] の group 形 [${
          expected.join(",")
        }]（group_size=${groupSize}）でない`,
      );
    }
    return;
  }
  if (view.shape.length !== weightShape.length) {
    throw new ContainerError(
      `${where}: scale [${view.shape.join(",")}] の rank が重み [${
        weightShape.join(",")
      }] と違う（keepdim broadcast 形が必要）`,
    );
  }
  const broadcastable = view.shape.every((dim, axis) => dim === 1 || dim === weightShape[axis]);
  if (!broadcastable) {
    throw new ContainerError(
      `${where}: scale [${view.shape.join(",")}] が重み [${
        weightShape.join(",")
      }] へ broadcast できない`,
    );
  }
  // MUST: 残る非 1 軸は**高々 1 本**（= keepdim 形の「チャネル軸だけが残る」）。broadcast 可能性
  // だけでは重みと同形の per-element scale（`[O,I]`）も通り、GPU 常駐経路は `wscale[チャネル]` の
  // 平坦添字で先頭要素しか読まない = 沈黙誤値になる。全軸 1（単一チャネルの退化形）は
  // `torch.amax(..., keepdim=True)` の正当な出力なので受理する。
  const channelAxes = view.shape.filter((dim) => dim !== 1).length;
  if (channelAxes > 1) {
    throw new ContainerError(
      `${where}: scale [${
        view.shape.join(",")
      }] の非 1 軸が ${channelAxes} 本（per-channel scale は 1 本まで — チャネル軸だけが残る keepdim 形）`,
    );
  }
};

/**
 * ノードが触る値の宣言 dtype。ins / outs が inputs[] か values{} のちょうど 1 箇所で
 * 宣言されることは parseIrGraph が保証済み（checkDefinitions / checkDeclarations）。
 */
const declaredDtype = (graph: IrGraph, name: string): IrDtype => {
  const input = graph.inputs.find((spec) => spec.name === name);
  return input !== undefined ? input.dtype : graph.values[name].dtype;
};

/**
 * ランタイム対応表と突合する。
 *
 * MUST: op 名だけでなく**意味論 dtype と attrs まで**見る。名前だけの突合は「対応表には
 * あるのに実行時に落ちる」を作る（docs/research の recon §3-9 が名指しした形 — 先行実験で
 * 実バグとして観測されている）。
 * 非対応は**全件列挙して** fail loudly する（1 件ずつ落とすと、対応表を埋める側が
 * 何本足りないのか分からない）。
 *
 * NOTE: 同じ規則を plan.ts の `validateGraphContracts` も見るが層が違う — こちらは
 * 「モデル作者へ capability 不足を一度に列挙する門」、あちらは「GPU 非依存に毎回通る
 * 契約検査」。両者とも src/ops.ts の契約表由来なので規則が割れることはない。
 */
export const assertRuntimeSupport = (graph: IrGraph, support: RuntimeSupport): void => {
  const missingOps = new Set<string>();
  // MUST: dtype 違反は**宣言（値名）単位に重複除去**する。エイリアス入力（`add(h,h)`）や、
  // 定義ノードの outs と消費ノードの ins の両方に現れる値は同じ宣言を何度も踏むため、素朴に
  // 積むと件数 N が「直すべき宣言の本数」より多く出て、列挙の指標としての意味が薄れる。
  const badDtypes = new Map<string, IrDtype>();
  const badAttrs: string[] = [];
  // 転送層の軸。ノード起点の突合より先に見る（宣言順 = inputs が先）。
  for (const spec of graph.inputs) {
    if (!support.io.has(spec.dtype)) badDtypes.set(spec.name, spec.dtype);
  }
  graph.nodes.forEach((node, index) => {
    const op = support.ops.get(node.op);
    if (op === undefined) {
      missingOps.add(node.op);
      return;
    }
    const where = `nodes[${index}] (${node.op})`;
    node.ins.forEach((name, slot) => {
      // 契約より入力が多い形（アリティ違反）は契約検査の担当。ここは列挙門なので、
      // 対応するスロットが無いぶんは和で見て 1 件でも多く拾う。
      const accept = op.slotDtypes[slot] ?? op.dtypes;
      const dtype = declaredDtype(graph, name);
      if (!accept.has(dtype)) badDtypes.set(name, dtype);
    });
    node.outs.forEach((name, slot) => {
      // 出力は契約表の写像の値域を**出力 slot 別に**見る（cast は attrs.to で決まるので
      // 語彙全体）。入力側の集合で代用すると、比較や bool の sum のように dtype が変わる op で
      // 正しいグラフが落ちる。契約より出力が多い形（出力数違反）は入力側と同様に契約検査の
      // 担当で、余ったぶんは全 slot の和で見て 1 件でも多く拾う。
      const accept = op.outDtypes[slot] ??
        new Set(op.outDtypes.flatMap((slotAccept) => [...slotAccept]));
      const dtype = declaredDtype(graph, name);
      if (!accept.has(dtype)) badDtypes.set(name, dtype);
    });
    const unknown = Object.keys(node.attrs).filter((key) => !op.attrKeys.has(key)).sort();
    if (unknown.length > 0) badAttrs.push(`${where}: ${unknown.join(", ")}`);
  });

  const missingStorage = new Map<IrStorageDtype, string[]>();
  // group 量子化を受理する格納は **i4 だけ**（ADR 0069 決定 2）。他の格納 dtype に付いた
  // group_size は実行経路が無く、黙って無視すると group ごとの scale を per-channel として
  // 読む沈黙誤値になるので、capability 不足で落とす。
  const groupQuantized: string[] = [];
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const dtype = initializer.storage.dtype;
    if (!support.storage.has(dtype)) {
      const users = missingStorage.get(dtype) ?? [];
      users.push(name);
      missingStorage.set(dtype, users);
      continue;
    }
    if (dtype !== "i4" && initializer.storage.groupSize !== undefined) groupQuantized.push(name);
  }

  if (
    missingOps.size === 0 && badDtypes.size === 0 && badAttrs.length === 0 &&
    missingStorage.size === 0 && groupQuantized.length === 0
  ) return;

  const diagnostics: string[] = [];
  if (missingOps.size > 0) {
    diagnostics.push(`非対応 op (${missingOps.size}): ${[...missingOps].sort().join(", ")}`);
  }
  if (badDtypes.size > 0) {
    const listed = [...badDtypes].map(([name, dtype]) => `値 '${name}': ${dtype}`);
    diagnostics.push(`非対応 意味論 dtype (${badDtypes.size}): ${listed.join(", ")}`);
  }
  if (badAttrs.length > 0) {
    diagnostics.push(`未実装 attrs (${badAttrs.length}): ${badAttrs.join(", ")}`);
  }
  for (const [dtype, users] of [...missingStorage].sort((a, b) => a[0].localeCompare(b[0]))) {
    diagnostics.push(`非対応 格納 dtype '${dtype}' (${users.length}): ${users.sort().join(", ")}`);
  }
  if (groupQuantized.length > 0) {
    diagnostics.push(
      `非対応 group 量子化 (${groupQuantized.length}): ${
        groupQuantized.sort().join(", ")
      }（group 量子化の格納は i4 のみ — ADR 0069）`,
    );
  }
  throw new ContainerError(`ランタイムの capability 不足 — ${diagnostics.join(" / ")}`);
};
